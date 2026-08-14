"""Pydantic schemas for the Investment Committee Agent.

Every object that crosses a pipeline boundary is typed and validated here:
  - user input (portfolio, profile)
  - the planner's structured plan
  - tool outputs
  - committee opinions
  - the final report

This is the system's contract. A tool or LLM that returns something that
does not fit these schemas is treated as a failure *at the boundary*, not
silently passed along.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    age: int = Field(ge=18, le=100, description="Investor age in years")
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    goals: list[str] = []


class Fund(BaseModel):
    """A single position in the portfolio. `allocation` is a fraction (0-1)."""

    ticker: str
    name: str = ""
    allocation: float = Field(ge=0.0, le=1.5)


class Portfolio(BaseModel):
    profile: UserProfile
    funds: list[Fund] = []


# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------


class PlannerPlan(BaseModel):
    """Structured decision from the Planner Agent."""

    intent: str = Field(description="One-line summary of what the user is asking")
    relevant_tools: list[str] = Field(description="Tool names the committee needs")
    required_information: list[str] = Field(
        description="What facts are needed to answer well"
    )
    specialists: list[str] = Field(
        description="Committee members that should weigh in"
    )


# ---------------------------------------------------------------------------
# Tool outputs
# ---------------------------------------------------------------------------


class FundAllocation(BaseModel):
    ticker: str
    name: str
    category: str
    weight_pct: float = Field(ge=0.0, le=100.0)
    expense_ratio_pct: float = Field(ge=0.0)


class CategoryAllocation(BaseModel):
    category: str
    weight_pct: float = Field(ge=0.0, le=100.0)
    num_funds: int = Field(ge=0)
    weighted_expense_ratio_pct: float = Field(ge=0.0)


class OverlapPair(BaseModel):
    fund_a: str
    fund_b: str
    overlap_score: float = Field(ge=0.0, le=1.0)
    shared_holdings: list[str] = []


class DiversificationMetrics(BaseModel):
    num_funds: int = Field(ge=0)
    num_categories: int = Field(ge=0)
    max_fund_weight_pct: float = Field(ge=0.0)
    max_category_weight_pct: float = Field(ge=0.0)
    herfindahl_index: float = Field(ge=0.0, le=1.0)
    effective_number_of_funds: float = Field(ge=0.0)
    average_pairwise_overlap: float = Field(ge=0.0)


class PortfolioAnalysis(BaseModel):
    fund_allocations: list[FundAllocation] = []
    category_allocations: list[CategoryAllocation] = []
    overlap_pairs: list[OverlapPair] = []
    diversification: DiversificationMetrics | None = None


class FundMetadataRecord(BaseModel):
    ticker: str
    name: str
    category: str
    benchmark: str
    expense_ratio_pct: float = Field(ge=0.0)
    aum_million: float = Field(ge=0.0)


class FundMetadataReport(BaseModel):
    records: list[FundMetadataRecord] = []


class HistoricalReturnRecord(BaseModel):
    ticker: str
    name: str
    cagr_pct: float
    annualized_volatility_pct: float = Field(ge=0.0)
    max_drawdown_pct: float = Field(ge=0.0, description="Positive number, e.g. 22.5 = -22.5%")
    months_of_data: int = Field(ge=0)
    insufficient_data: bool = False


class HistoricalReturnsReport(BaseModel):
    records: list[HistoricalReturnRecord] = []


class RollingReturn(BaseModel):
    end_month: str
    return_pct: float


class RiskMetricRecord(BaseModel):
    ticker: str
    sharpe_ratio: float
    max_drawdown_pct: float = Field(ge=0.0)
    annualized_volatility_pct: float = Field(ge=0.0)
    rolling_12m_returns: list[RollingReturn] = []
    risk_free_rate_pct: float = Field(ge=0.0)
    insufficient_data: bool = False


class RiskMetricsReport(BaseModel):
    records: list[RiskMetricRecord] = []


class ToolStatus(BaseModel):
    status: Literal["ok", "partial", "error"]
    warnings: list[str] = []
    error: str | None = None


class ToolResults(BaseModel):
    """Aggregated tool outputs plus per-tool status for the graph state."""

    portfolio_analysis: PortfolioAnalysis | None = None
    fund_metadata: FundMetadataReport | None = None
    historical_returns: HistoricalReturnsReport | None = None
    risk_metrics: RiskMetricsReport | None = None
    status: dict[str, ToolStatus] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Committee opinions
# ---------------------------------------------------------------------------


ADVISOR_NAMES = ("conservative", "growth", "cost_efficiency", "devils_advocate")


class CommitteeOpinion(BaseModel):
    advisor: Literal["conservative", "growth", "cost_efficiency", "devils_advocate"]
    stance: str = Field(description='e.g. "hold", "redeem", "add"')
    recommendation_summary: str = Field(
        description="Short conclusion the advisor reaches"
    )
    key_points: list[str] = []
    concerns: list[str] = []
    evidence: list[str] = Field(
        default_factory=list, description="Metric/evidence references used"
    )


class CommitteeOpinions(BaseModel):
    conservative: CommitteeOpinion
    growth: CommitteeOpinion
    cost_efficiency: CommitteeOpinion
    devils_advocate: CommitteeOpinion


class ConsensusOutput(BaseModel):
    """Structured verdict from the Consensus Agent."""

    agreements: list[str] = []
    disagreements: list[str] = []
    final_recommendation: str = ""
    preferred_stance: str = ""


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    source: str = Field(description="Tool or dataset that produced the fact")
    metric: str
    value: str
    description: str = ""
    kind: Literal["calculated", "dataset", "assumption"] = "calculated"


class FinalReport(BaseModel):
    question: str
    committee_opinions: dict[str, str] = Field(
        description="advisor name -> recommendation summary (assignment format)"
    )
    agreements: list[str] = []
    disagreements: list[str] = []
    confidence_score: float = Field(ge=0.0, le=1.0)
    final_recommendation: str
    evidence: list[EvidenceItem] = []
    assumptions: list[str] = []
    tool_errors: list[str] = []
    tool_warnings: list[str] = []
