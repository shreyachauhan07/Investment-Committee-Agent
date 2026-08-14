"""End-to-end graph tests using the deterministic mock LLM.

These verify the pipeline produces the assignment's required structure:
planner runs, relevant tools are called, all four specialists form
independent opinions, disagreements are preserved, confidence is scored,
and the final report carries tool-backed evidence.
"""

from agents.llm import MockLLMClient
from graph import run_investment_committee
from models import ADVISOR_NAMES


def _run(question: str, portfolio):
    return run_investment_committee(question, portfolio, llm=MockLLMClient())


def test_full_pipeline_produces_valid_report(sample_portfolio):
    report = _run("Should I redeem the Pinnacle Small Cap fund?", sample_portfolio)
    assert set(report.committee_opinions) == set(ADVISOR_NAMES)
    assert report.final_recommendation
    assert 0.0 <= report.confidence_score <= 1.0
    assert report.tool_errors == []


def test_evidence_is_nonempty_and_traceable(sample_portfolio):
    report = _run("Should I add a Small Cap fund?", sample_portfolio)
    assert len(report.evidence) > 0
    sources = {item.source for item in report.evidence}
    assert {"portfolio_analyzer", "fund_metadata"} <= sources
    assert all(item.kind in ("calculated", "dataset", "assumption") for item in report.evidence)


def test_disagreement_is_preserved(sample_portfolio):
    report = _run("Should I redeem the Pinnacle Small Cap fund?", sample_portfolio)
    # With the mock, the four advisors genuinely differ on redemption.
    assert report.disagreements, "Expected the committee to disagree"
    assert report.agreements is not None


def test_planner_triggers_relevant_tools(sample_portfolio):
    report = _run("Why did my portfolio underperform?", sample_portfolio)
    # Underperformance questions must pull in return/risk tools.
    sources = {item.source for item in report.evidence}
    assert "historical_returns" in sources
    assert "risk_metrics" in sources


def test_all_four_opinions_present(sample_portfolio):
    report = _run("Is my allocation consistent with my age and goals?", sample_portfolio)
    for advisor in ADVISOR_NAMES:
        assert report.committee_opinions[advisor]


def test_overdiversification_question(sample_portfolio):
    report = _run("Am I over-diversified?", sample_portfolio)
    assert report.final_recommendation
    assert any(item.source == "portfolio_analyzer" for item in report.evidence)


def test_repeat_runs_are_deterministic(sample_portfolio):
    q = "Should I increase my debt allocation?"
    first = _run(q, sample_portfolio)
    second = _run(q, sample_portfolio)
    assert first.model_dump() == second.model_dump()
