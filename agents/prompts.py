"""Prompt construction for every agent.

Each prompt embeds structured context inside clear <tag> blocks:
    <question>, <evidence>, <opinions>, ...
The real LLM reads these as ordinary text; the mock LLM parses the same
blocks to build deterministic output. Keeping one prompt format means the
two LLM implementations always see identical information.

The specialist system prompts encode genuinely different incentives, which
is what produces disagreement on the committee.
"""

from __future__ import annotations

import json
from typing import Any

from config import RISK_FREE_RATE_PCT
from models import Portfolio, ToolResults
from tools.registry import TOOL_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Committee role definitions (system prompts)
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the planning agent of an investment committee.

Your job:
1. Understand what the user is asking.
2. Decide which tools the committee needs. Choose ONLY from the available tools list.
3. State what information is required to answer well.
4. Decide which specialist committee members should weigh in.

Return ONLY a JSON object with keys: intent, relevant_tools,
required_information, specialists."""

SPECIALIST_SYSTEMS = {
    "conservative": """You are a CONSERVATIVE advisor on an investment committee.

Your mandate is:
- Capital preservation above all else
- Strong diversification and downside risk protection
- Skepticism of concentrated positions, high volatility, and funds with
  large drawdowns

Evaluate the evidence strictly from this perspective. Be specific: cite
the actual numbers in the evidence (volatility, drawdowns, exposures).
You may disagree with the growth-oriented advisors — that is expected.

Return ONLY a JSON object with keys: advisor, stance, recommendation_summary,
key_points, concerns, evidence.""",

    "growth": """You are a GROWTH advisor on an investment committee.

Your mandate is:
- Maximizing long-term returns, especially for a young investor with a
  long horizon (retirement, child education)
- Higher risk tolerance as the price of compounding
- Challenging excessive caution when the investor has decades to recover

Evaluate the evidence strictly from this perspective. Be specific: cite
the actual numbers (CAGR, Sharpe ratio, volatility). You may disagree with
the conservative advisor — that is expected.

Return ONLY a JSON object with keys: advisor, stance, recommendation_summary,
key_points, concerns, evidence.""",

    "cost_efficiency": """You are the COST & EFFICIENCY advisor on an investment committee.

Your mandate is:
- Portfolio simplicity
- Eliminating fund overlap and unnecessary funds
- Minimizing expense ratios (fees erode returns silently)
- Avoiding redundant positions that do not add diversification

Evaluate the evidence strictly from this perspective. Be specific: cite
the actual numbers (expense ratios, overlap scores, weighted category costs).

Return ONLY a JSON object with keys: advisor, stance, recommendation_summary,
key_points, concerns, evidence.""",

    "devils_advocate": """You are the DEVIL'S ADVOCATE on an investment committee.

Your job is to challenge the other advisors and the assumptions behind their
recommendations. Actively look for:
- Why their advice might fail
- Hidden risks they overlooked
- Assumptions that may not hold (e.g. past returns repeating, single
  benchmark comparisons, ignoring the investor's actual goals and age)

You are not trying to block decisions; you are stress-testing them. Be
specific and reference the evidence.

Return ONLY a JSON object with keys: advisor, stance, recommendation_summary,
key_points, concerns, evidence.""",
}

CONSENSUS_SYSTEM = """You are the consensus agent chairing an investment committee.

You receive independent opinions from four advisors. Your job:
1. Aggregate the viewpoints.
2. Identify genuine agreements and genuine disagreements.
3. Explain WHY a disagreement exists (different priorities/assumptions).
4. Produce ONE final recommendation. Do NOT force agreement — if the
   advisors truly disagree, say so and explain the trade-off, then state
   which side you ultimately prefer and the assumptions that drive that
   preference.

Return ONLY a JSON object with keys: agreements, disagreements,
final_recommendation, preferred_stance."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _schema(response_model: type) -> str:
    return json.dumps(response_model.model_json_schema(), indent=1)


def _snippet(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def planner_prompt(
    question: str, portfolio: Portfolio, response_model: type
) -> tuple[str, str]:
    user = f"""<question>{_snippet(question)}</question>

<portfolio>{json.dumps(portfolio.model_dump())}</portfolio>

<available_tools>{json.dumps(TOOL_DESCRIPTIONS)}</available_tools>

Respond with JSON only, matching this schema:
{_schema(response_model)}"""
    return PLANNER_SYSTEM, user


def specialist_prompt(
    role: str,
    question: str,
    portfolio: Portfolio,
    tool_results: ToolResults,
    response_model: type,
) -> tuple[str, str]:
    user = f"""<role>{role}</role>
<question>{_snippet(question)}</question>

<profile>{json.dumps(portfolio.profile.model_dump())}</profile>

<evidence>{json.dumps(evidence_digest(tool_results))}</evidence>

<tool_status>{json.dumps(tool_status_digest(tool_results))}</tool_status>

Notes:
- {RISK_FREE_RATE_PCT}% annual risk-free rate is assumed for Sharpe ratios.
- Metrics marked insufficient_data are not reliable — do not over-rely on them.
- 'stance' should be a short verb like "hold", "redeem", "add", "consolidate".

Respond with JSON only, matching this schema:
{_schema(response_model)}"""
    return SPECIALIST_SYSTEMS[role], user


def consensus_prompt(
    question: str,
    opinions: list[dict],
    tool_results: ToolResults,
    response_model: type,
) -> tuple[str, str]:
    user = f"""<question>{_snippet(question)}</question>

<opinions>{json.dumps(opinions)}</opinions>

<tool_status>{json.dumps(tool_status_digest(tool_results))}</tool_status>

Respond with JSON only, matching this schema:
{_schema(response_model)}"""
    return CONSENSUS_SYSTEM, user


# ---------------------------------------------------------------------------
# Evidence digest — the curated facts the committee reasons over
# ---------------------------------------------------------------------------


def evidence_digest(results: ToolResults) -> dict[str, Any]:
    digest: dict[str, Any] = {}

    if results.portfolio_analysis:
        pa = results.portfolio_analysis
        digest["allocation"] = {
            fa.ticker: {
                "name": fa.name,
                "category": fa.category,
                "weight_pct": fa.weight_pct,
                "expense_ratio_pct": fa.expense_ratio_pct,
            }
            for fa in pa.fund_allocations
        }
        digest["categories"] = [ca.model_dump() for ca in pa.category_allocations]
        digest["overlap"] = [
            p.model_dump() for p in pa.overlap_pairs[:10]
        ]
        if pa.diversification:
            digest["diversification"] = pa.diversification.model_dump()

    if results.fund_metadata:
        digest["metadata"] = {
            r.ticker: {
                "category": r.category,
                "benchmark": r.benchmark,
                "expense_ratio_pct": r.expense_ratio_pct,
                "aum_million": r.aum_million,
            }
            for r in results.fund_metadata.records
        }

    if results.historical_returns:
        digest["historical"] = {
            r.ticker: {
                "cagr_pct": r.cagr_pct,
                "annualized_volatility_pct": r.annualized_volatility_pct,
                "max_drawdown_pct": r.max_drawdown_pct,
                "insufficient_data": r.insufficient_data,
            }
            for r in results.historical_returns.records
        }

    if results.risk_metrics:
        digest["risk"] = {
            r.ticker: {
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown_pct": r.max_drawdown_pct,
                "annualized_volatility_pct": r.annualized_volatility_pct,
                "avg_12m_rolling_return_pct": (
                    round(
                        sum(x.return_pct for x in r.rolling_12m_returns)
                        / len(r.rolling_12m_returns),
                        2,
                    )
                    if r.rolling_12m_returns
                    else None
                ),
                "insufficient_data": r.insufficient_data,
            }
            for r in results.risk_metrics.records
        }

    return digest


def tool_status_digest(results: ToolResults) -> dict[str, str]:
    return {name: status.status for name, status in results.status.items()}
