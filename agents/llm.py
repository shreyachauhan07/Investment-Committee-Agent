"""LLM client abstraction.

One interface, two implementations:

  * OpenAILLMClient  - real calls to any OpenAI-compatible API endpoint.
  * MockLLMClient    - deterministic offline stand-in used when no API key
    is configured (or LLM_MOCK=1). It returns structurally valid output
    derived from the prompt, so the whole graph runs without network access.

`get_llm_client()` picks the implementation from environment config.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol, TypeVar

import openai
from pydantic import BaseModel, ValidationError
from openai import OpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MOCK,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)
from models import (
    ADVISOR_NAMES,
    CommitteeOpinion,
    ConsensusOutput,
    PlannerPlan,
)

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """What every agent needs from an LLM: validated JSON in, model out."""

    def complete_json(
        self, system: str, user: str, response_model: type[T]
    ) -> T: ...


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce valid structured output."""


# ---------------------------------------------------------------------------
# Real implementation
# ---------------------------------------------------------------------------


class OpenAILLMClient:
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=LLM_API_KEY or "not-set",
            base_url=LLM_BASE_URL,
            timeout=LLM_TIMEOUT,
        )
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.max_retries = LLM_MAX_RETRIES

    def complete_json(
        self, system: str, user: str, response_model: type[T]
    ) -> T:
        """Ask the model for JSON and validate it against `response_model`.

        Retry strategy:
          - validation failure -> send the validation error back to the model
            and ask it to fix the JSON (the "repair loop")
          - API failure -> retry with exponential backoff
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                content = self._chat(messages)
                data = json.loads(content)
                return response_model.model_validate(data)
            except ValidationError as exc:
                log.warning("LLM output failed validation (attempt %d): %s", attempt, exc)
                if attempt == self.max_retries:
                    raise LLMError(
                        f"Model could not produce valid {response_model.__name__}: {exc}"
                    ) from exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON was invalid. Fix it to match the "
                            f"schema. Validation error: {exc}. Return ONLY valid JSON."
                        ),
                    }
                )
            except openai.AuthenticationError as exc:
                raise LLMError(
                    "Invalid LLM API key. Set LLM_API_KEY in your .env file."
                ) from exc
            except openai.RateLimitError as exc:
                log.warning("Rate limited (attempt %d); backing off", attempt)
                self._backoff(attempt)
                if attempt == self.max_retries:
                    raise LLMError("LLM rate limit exceeded") from exc
            except openai.APITimeoutError as exc:
                log.warning("LLM timed out (attempt %d)", attempt)
                self._backoff(attempt)
                if attempt == self.max_retries:
                    raise LLMError("LLM request timed out") from exc
            except openai.APIConnectionError as exc:
                log.warning("LLM connection error (attempt %d)", attempt)
                self._backoff(attempt)
                if attempt == self.max_retries:
                    raise LLMError("Could not connect to LLM API") from exc

        raise LLMError("Unexpectedly exhausted retries")  # pragma: no cover

    def _chat(self, messages: list[dict]) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": 42,  # deterministic ordering within a given prompt
        }
        try:
            # Not every OpenAI-compatible provider supports this flag.
            kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**kwargs)
        except (openai.BadRequestError, openai.APIStatusError, TypeError):
            # Some providers reject response_format -> retry without it.
            log.info("Provider rejected response_format; retrying without it")
            kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2 ** (attempt - 1), 8))


# ---------------------------------------------------------------------------
# Deterministic mock (offline / tests)
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Generates valid structured output from the prompt alone.

    It reads the structured context blocks that `agents/prompts.py` embeds
    into every prompt (question, evidence, opinions), so its answers vary
    with the input. Reasoning quality is intentionally shallow — it exists
    to exercise the pipeline, not to give investment advice.
    """

    def complete_json(
        self, system: str, user: str, response_model: type[T]
    ) -> T:
        payload: dict = {}
        if response_model is PlannerPlan:
            payload = self._plan(user)
        elif response_model is CommitteeOpinion:
            payload = self._opinion(system, user)
        elif response_model is ConsensusOutput:
            payload = self._consensus(user)
        else:  # pragma: no cover
            raise LLMError(f"Mock LLM has no generator for {response_model.__name__}")
        return response_model.model_validate(payload)

    # -- generators -------------------------------------------------------

    @staticmethod
    def _plan(user: str) -> dict:
        q = _block(user, "question")
        tools = ["portfolio_analyzer", "fund_metadata"]
        if any(k in q.lower() for k in ("redeem", "sell", "exit", "add", "new fund", "debt")):
            tools += ["historical_returns", "risk_metrics"]
        if any(k in q.lower() for k in ("risk", "volatile", "underperform", "drawdown", "sharpe")):
            tools += ["historical_returns", "risk_metrics"]
        tools = sorted(set(tools))
        return {
            "intent": q,
            "relevant_tools": tools,
            "required_information": [
                "Current fund allocations and categories",
                "Expense ratios of the funds involved",
                "Historical return and risk behaviour",
            ],
            "specialists": list(ADVISOR_NAMES),
        }

    @staticmethod
    def _opinion(system: str, user: str) -> dict:
        advisor = _block(user, "role") or "conservative"
        q = _block(user, "question")
        stance = _stance(q, advisor)
        evidence = _evidence_block(user)
        evidence_refs = list(evidence.keys())[:3] if isinstance(evidence, dict) else []
        summary = f"{_role_word(advisor)} stance on '{q}': {stance}."
        return {
            "advisor": advisor,
            "stance": stance,
            "recommendation_summary": summary,
            "key_points": [
                f"Considers the question '{q}' through a {advisor} lens.",
                "Points to the available evidence to support its stance.",
            ],
            "concerns": [f"Risks that a {advisor} advisor flags are emphasized."],
            "evidence": evidence_refs,
        }

    @staticmethod
    def _consensus(user: str) -> dict:
        opinions = _opinions_block(user)
        stances = [o.get("stance", "") for o in opinions]
        stance_counts: dict[str, int] = {}
        for s in stances:
            stance_counts[s] = stance_counts.get(s, 0) + 1
        majority = max(stance_counts, key=stance_counts.get) if stances else "hold"
        majority_count = stance_counts.get(majority, 0)
        minority = [
            s for s in set(stances) if s != majority and stance_counts[s] > 0
        ]
        agreements = (
            [f"{majority_count} of {len(stances)} advisors converge on '{majority}'."]
            if majority_count >= 2
            else ["The advisors are split; no clear majority stance."]
        )
        disagreements = (
            [f"Advisors differ: {majority} vs {', '.join(minority)}."]
            if minority
            else []
        )
        return {
            "agreements": agreements,
            "disagreements": disagreements,
            "final_recommendation": (
                f"On balance, the committee recommends to {majority} given the "
                "available evidence."
            ),
            "preferred_stance": majority,
        }


# ---------------------------------------------------------------------------
# Prompt context block helpers (used by both the real and mock clients)
# ---------------------------------------------------------------------------


def _block(text: str, tag: str) -> str:
    """Extract <tag>...</tag> content from a prompt."""
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return m.group(1) if m else ""


def _evidence_block(user: str) -> dict:
    raw = _block(user, "evidence")
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _opinions_block(user: str) -> list[dict]:
    raw = _block(user, "opinions")
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _stance(q: str, advisor: str) -> str:
    """Role-aware mock stance so the committee genuinely disagrees.

    Each advisor answers from its mandate: the conservative resists risk,
    the growth advisor resists giving up return, the cost advisor hunts
    fees/overlap, and the devil's advocate challenges the direction.
    """
    low = q.lower()
    if advisor == "devils_advocate":
        return "challenge"
    # Debt/bond questions must be caught before the generic "add/increase"
    # rule, otherwise a conservative would wrongly resist adding safer debt.
    if any(k in low for k in ("debt", "bond", "fixed income")):
        return "add" if advisor == "conservative" else "hold"
    if any(k in low for k in ("redeem", "sell", "exit")):
        return (
            "redeem"
            if advisor == "conservative"
            else "hold" if advisor == "growth" else "consolidate"
        )
    if any(k in low for k in ("add", "increase", "new fund", "small cap")):
        return (
            "add"
            if advisor == "growth"
            else "hold" if advisor == "conservative" else "consolidate"
        )
    if any(k in low for k in ("over-diversified", "consolidate", "simplify", "overlap")):
        return "consolidate" if advisor == "cost_efficiency" else "hold"
    if any(k in low for k in ("risk", "volatile", "drawdown", "dangerous")):
        return "reduce risk" if advisor == "conservative" else "hold"
    if any(k in low for k in ("underperform", "lag", "behind")):
        return "investigate" if advisor == "devils_advocate" else "hold"
    return "hold"


def _role_word(advisor: str) -> str:
    return {
        "conservative": "The conservative advisor",
        "growth": "The growth advisor",
        "cost_efficiency": "The cost and efficiency advisor",
        "devils_advocate": "The devil's advocate",
    }.get(advisor, advisor)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_client() -> OpenAILLMClient | MockLLMClient:
    if LLM_MOCK or not LLM_API_KEY:
        log.info("Using mock LLM client (set LLM_API_KEY for real model calls)")
        return MockLLMClient()
    log.info("Using OpenAI-compatible LLM client (model=%s)", LLM_MODEL)
    return OpenAILLMClient()
