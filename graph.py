"""LangGraph orchestration for the Investment Committee Agent.

Flow:
    plan -> tools -> committee -> consensus -> finalize -> END

Every node reads and updates a shared `CommitteeState`. The graph is
compiled once and reused, so runs are cheap and the flow is easy to
visualize (LangGraph can render the same structure that the README diagram
shows).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.consensus import reach_consensus
from agents.llm import get_llm_client
from agents.planner import plan
from agents.specialists import run_committee
from confidence import compute_confidence
from config import RISK_FREE_RATE_PCT
from models import (
    ADVISOR_NAMES,
    CommitteeOpinions,
    ConsensusOutput,
    EvidenceItem,
    FinalReport,
    PlannerPlan,
    Portfolio,
    ToolResults,
)
from tools.registry import run_relevant_tools


class CommitteeState(TypedDict, total=False):
    question: str
    portfolio: Portfolio
    plan: PlannerPlan
    tool_results: ToolResults
    opinions: CommitteeOpinions
    consensus: ConsensusOutput
    report: FinalReport
    _llm: object


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def plan_node(state: CommitteeState) -> dict:
    return {"plan": plan(state["question"], state["portfolio"], state["_llm"])}


def tools_node(state: CommitteeState) -> dict:
    tool_names = state["plan"].relevant_tools
    return {"tool_results": run_relevant_tools(state["portfolio"], tool_names)}


def committee_node(state: CommitteeState) -> dict:
    return {
        "opinions": run_committee(
            state["question"],
            state["portfolio"],
            state["tool_results"],
            state["_llm"],
        )
    }


def consensus_node(state: CommitteeState) -> dict:
    return {
        "consensus": reach_consensus(
            state["question"],
            state["opinions"],
            state["tool_results"],
            state["_llm"],
        )
    }


def finalize_node(state: CommitteeState) -> dict:
    confidence = compute_confidence(
        state["consensus"], state["opinions"], state["tool_results"]
    )
    report = _build_report(state, confidence)
    return {"report": report}


# ---------------------------------------------------------------------------
# Report assembly (deterministic — driven by tool outputs, not the LLM)
# ---------------------------------------------------------------------------


def _build_report(state: CommitteeState, confidence: float) -> FinalReport:
    opinions, consensus, tool_results = (
        state["opinions"],
        state["consensus"],
        state["tool_results"],
    )
    tool_errors = [
        f"{name}: {status.error}"
        for name, status in tool_results.status.items()
        if status.status == "error" and status.error
    ]
    tool_warnings = [
        f"{name}: {warning}"
        for name, status in tool_results.status.items()
        for warning in status.warnings
    ]

    return FinalReport(
        question=state["question"],
        committee_opinions={
            advisor: getattr(opinions, advisor).recommendation_summary
            for advisor in ADVISOR_NAMES
        },
        agreements=consensus.agreements,
        disagreements=consensus.disagreements,
        confidence_score=confidence,
        final_recommendation=consensus.final_recommendation,
        evidence=_build_evidence(tool_results),
        assumptions=_build_assumptions(state),
        tool_errors=tool_errors,
        tool_warnings=tool_warnings,
    )


def _build_evidence(tool_results: ToolResults) -> list[EvidenceItem]:
    """Every evidence item comes straight from a tool's typed output."""
    evidence: list[EvidenceItem] = []

    if tool_results.portfolio_analysis:
        pa = tool_results.portfolio_analysis
        div = pa.diversification
        if div:
            evidence.append(
                EvidenceItem(
                    source="portfolio_analyzer",
                    metric="diversification",
                    value=(
                        f"{div.num_funds} funds / {div.num_categories} categories, "
                        f"effective count {div.effective_number_of_funds}, "
                        f"avg overlap {div.average_pairwise_overlap}"
                    ),
                    description="Portfolio diversification metrics.",
                )
            )
        for pair in pa.overlap_pairs[:3]:
            evidence.append(
                EvidenceItem(
                    source="portfolio_analyzer",
                    metric="overlap",
                    value=f"{pair.fund_a} x {pair.fund_b} = {pair.overlap_score}",
                    description=f"Shared holdings: {', '.join(pair.shared_holdings)}",
                )
            )
        for cat in pa.category_allocations[:4]:
            evidence.append(
                EvidenceItem(
                    source="portfolio_analyzer",
                    metric="category_exposure",
                    value=f"{cat.category} = {cat.weight_pct}%",
                    description=(
                        f"weighted expense ratio {cat.weighted_expense_ratio_pct}%"
                    ),
                )
            )

    if tool_results.fund_metadata:
        for record in tool_results.fund_metadata.records:
            evidence.append(
                EvidenceItem(
                    source="fund_metadata",
                    metric=f"{record.ticker} expense_ratio",
                    value=f"{record.expense_ratio_pct}%",
                    description=(
                        f"{record.name} ({record.category}, benchmark "
                        f"{record.benchmark}, AUM ${record.aum_million / 1000:.1f}B)"
                    ),
                    kind="dataset",
                )
            )

    if tool_results.historical_returns:
        for record in tool_results.historical_returns.records:
            if record.insufficient_data:
                continue
            evidence.append(
                EvidenceItem(
                    source="historical_returns",
                    metric=f"{record.ticker} cagr",
                    value=f"{record.cagr_pct}%",
                    description=f"volatility {record.annualized_volatility_pct}%, "
                    f"max drawdown {record.max_drawdown_pct}%",
                )
            )

    if tool_results.risk_metrics:
        for record in tool_results.risk_metrics.records:
            if record.insufficient_data:
                continue
            evidence.append(
                EvidenceItem(
                    source="risk_metrics",
                    metric=f"{record.ticker} sharpe_ratio",
                    value=str(record.sharpe_ratio),
                    description=(
                        f"max drawdown {record.max_drawdown_pct}%, "
                        f"assumed risk-free {record.risk_free_rate_pct}%"
                    ),
                )
            )

    return evidence


def _build_assumptions(state: CommitteeState) -> list[str]:
    return [
        f"Risk-free rate assumed at {RISK_FREE_RATE_PCT}% for Sharpe ratios.",
        "Historical metrics computed on 60 months of monthly returns (synthetic).",
        "All fund and market data is synthetic and for demonstration only, not "
        "a basis for real investment decisions.",
        "Committee opinions are AI-generated interpretations of tool output.",
    ]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(llm=None):
    graph = StateGraph(CommitteeState)
    graph.add_node("plan", plan_node)
    graph.add_node("tools", tools_node)
    graph.add_node("committee", committee_node)
    graph.add_node("consensus", consensus_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "tools")
    graph.add_edge("tools", "committee")
    graph.add_edge("committee", "consensus")
    graph.add_edge("consensus", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_investment_committee(
    question: str,
    portfolio: Portfolio,
    llm=None,
) -> FinalReport:
    """Public entry point: answer a portfolio question end to end."""
    llm = llm or get_llm_client()
    app = build_graph(llm)
    state: CommitteeState = {
        "question": question,
        "portfolio": portfolio,
        "_llm": llm,
    }
    result = app.invoke(state)
    return result["report"]
