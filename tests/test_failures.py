"""Failure-handling tests: the system must degrade gracefully, never crash."""

import pytest

from agents.llm import LLMError, MockLLMClient
from graph import run_investment_committee
from main import load_portfolio
from models import Fund, Portfolio


class FailingLLM:
    """Simulates a total LLM outage."""

    def complete_json(self, system, user, response_model):
        raise LLMError("simulated API outage")


class InvalidOutputLLM:
    """Simulates a model that returns garbage that never validates."""

    def complete_json(self, system, user, response_model):
        return object()


def test_llm_outage_produces_explicit_fallback(sample_portfolio):
    report = run_investment_committee(
        "Should I redeem Fund X?", sample_portfolio, llm=FailingLLM()
    )
    # Every specialist reports it could not conclude...
    for opinion in report.committee_opinions.values():
        assert "could not reach a conclusion" in opinion
    # ...and the failure is surfaced, not hidden.
    assert any("could not be computed" in d.lower() for d in report.disagreements)
    assert report.final_recommendation
    assert 0.0 <= report.confidence_score <= 1.0


def test_unknown_fund_portfolio_still_reports(sample_portfolio):
    pf = Portfolio(
        profile=sample_portfolio.profile,
        funds=[Fund(ticker="GHOST", allocation=0.3), Fund(ticker="CREL", allocation=0.7)],
    )
    report = run_investment_committee("Should I redeem the unknown fund?", pf, llm=MockLLMClient())
    assert report.final_recommendation
    assert not report.tool_errors  # warnings are partial, not hard errors
    # Evidence still covers the recognized fund.
    assert any("CREL" in e.metric for e in report.evidence)


def test_empty_portfolio_degrades_gracefully(sample_portfolio):
    pf = Portfolio(profile=sample_portfolio.profile, funds=[])
    report = run_investment_committee("Am I over-diversified?", pf, llm=MockLLMClient())
    assert report.final_recommendation
    assert 0.0 <= report.confidence_score <= 1.0
    # The lack of data must be visible in the report, not silently hidden.
    assert report.tool_warnings
    assert any("empty" in w.lower() for w in report.tool_warnings)


def test_missing_portfolio_file_exits_cleanly(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(SystemExit):
        load_portfolio(str(missing))


def test_invalid_portfolio_json_exits_cleanly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_portfolio(str(bad))


def test_invalid_portfolio_fields_exit_cleanly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"profile": {"age": 35}, "funds": [{"ticker": "CREL", "allocation": -5}]}',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_portfolio(str(bad))
