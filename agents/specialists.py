"""Specialist committee members.

Each specialist receives the same evidence but evaluates it through its
own role definition (see SPECIALIST_SYSTEMS in agents/prompts.py). Running
them independently is what produces genuine disagreement.

On LLM failure a specialist still reports — its stance is recorded as
"insufficient evidence" so the committee can see a member could not
conclude, instead of silently losing that member's view.
"""

from __future__ import annotations

import logging

from agents.llm import LLMError, LLMClient
from agents.prompts import specialist_prompt
from models import (
    ADVISOR_NAMES,
    CommitteeOpinion,
    CommitteeOpinions,
    Portfolio,
    ToolResults,
)

log = logging.getLogger(__name__)

_STANCE_ON_FAILURE = "insufficient evidence"


def _fallback_opinion(role: str) -> CommitteeOpinion:
    return CommitteeOpinion(
        advisor=role,
        stance=_STANCE_ON_FAILURE,
        recommendation_summary=(
            f"{role} could not reach a conclusion (evidence unavailable or "
            "model failure)."
        ),
        key_points=[],
        concerns=["No reliable opinion could be formed."],
        evidence=[],
    )


def run_specialist(
    role: str,
    question: str,
    portfolio: Portfolio,
    tool_results: ToolResults,
    llm: LLMClient,
) -> CommitteeOpinion:
    system, user = specialist_prompt(role, question, portfolio, tool_results, CommitteeOpinion)
    try:
        opinion = llm.complete_json(system, user, CommitteeOpinion)
    except LLMError:
        log.exception("Specialist '%s' failed; recording fallback opinion", role)
        return _fallback_opinion(role)

    # Defensive: the model must report as its assigned role, whatever it typed.
    if opinion.advisor != role:
        opinion = opinion.model_copy(update={"advisor": role})
    log.info(
        "Specialist '%s' stance=%s | summary=%s",
        role, opinion.stance, opinion.recommendation_summary[:80],
    )
    return opinion


def run_committee(
    question: str,
    portfolio: Portfolio,
    tool_results: ToolResults,
    llm: LLMClient,
) -> CommitteeOpinions:
    """Run all four specialists and collect their independent opinions."""
    opinions = [
        run_specialist(role, question, portfolio, tool_results, llm)
        for role in ADVISOR_NAMES
    ]
    return CommitteeOpinions(
        conservative=opinions[0],
        growth=opinions[1],
        cost_efficiency=opinions[2],
        devils_advocate=opinions[3],
    )
