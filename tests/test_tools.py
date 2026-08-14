"""Tool correctness and failure-behaviour tests (no LLM involved)."""

import pytest

from models import Fund, Portfolio
from tools.portfolio_analyzer import analyze_portfolio
from tools.fund_metadata import get_fund_metadata
from tools.historical_returns import historical_returns
from tools.risk_metrics import risk_metrics
from tools.registry import run_relevant_tools


def test_allocation_breakdown_sums_to_100(sample_portfolio):
    analysis, _ = analyze_portfolio(sample_portfolio)
    total = sum(fa.weight_pct for fa in analysis.fund_allocations)
    assert total == pytest.approx(100.0, abs=0.01)


def test_category_rollup_matches_fund_level(sample_portfolio):
    analysis, _ = analyze_portfolio(sample_portfolio)
    cat_total = sum(ca.weight_pct for ca in analysis.category_allocations)
    assert cat_total == pytest.approx(100.0, abs=0.01)
    assert len(analysis.category_allocations) == 6


def test_known_overlap_pair_has_expected_holdings(sample_portfolio):
    analysis, _ = analyze_portfolio(sample_portfolio)
    by_pair = {(p.fund_a, p.fund_b): p for p in analysis.overlap_pairs}
    pair = by_pair[("CREL", "SUMFLEX")]
    assert pair.overlap_score > 0.5
    assert {"MEGTECH", "NATBANK"}.issubset(pair.shared_holdings)


def test_diversification_metrics_sane(sample_portfolio):
    analysis, _ = analyze_portfolio(sample_portfolio)
    d = analysis.diversification
    assert d.num_funds == 6
    assert d.effective_number_of_funds > 1
    assert 0 <= d.herfindahl_index <= 1
    assert d.max_fund_weight_pct == 25.0


def test_unknown_fund_skipped_with_warning(sample_portfolio):
    pf = Portfolio(
        profile=sample_portfolio.profile,
        funds=[Fund(ticker="GHOST", allocation=1.0)],
    )
    analysis, warnings = analyze_portfolio(pf)
    assert analysis.fund_allocations == []
    assert any("GHOST" in w for w in warnings)


def test_empty_portfolio_degrades_gracefully(sample_portfolio):
    pf = Portfolio(profile=sample_portfolio.profile, funds=[])
    results = run_relevant_tools(pf, ["portfolio_analyzer", "fund_metadata"])
    assert results.status["portfolio_analyzer"].status == "partial"
    assert results.portfolio_analysis is not None


def test_zero_allocation_portfolio_no_crash(sample_portfolio):
    pf = Portfolio(
        profile=sample_portfolio.profile,
        funds=[Fund(ticker="CREL", allocation=0.0)],
    )
    analysis, warnings = analyze_portfolio(pf)
    assert analysis.diversification is not None
    assert analysis.diversification.herfindahl_index == 0.0


def test_metadata_records_correct_fields(sample_portfolio):
    report, _ = get_fund_metadata(sample_portfolio)
    by_ticker = {r.ticker: r for r in report.records}
    crel = by_ticker["CREL"]
    assert crel.category == "Large Cap Equity"
    assert crel.expense_ratio_pct == 1.05
    assert crel.benchmark == "LargeCap 250 Index"
    assert crel.aum_million > 0


def test_historical_returns_plausible(sample_portfolio):
    report, _ = historical_returns(sample_portfolio)
    by_ticker = {r.ticker: r for r in report.records}
    crel = by_ticker["CREL"]
    assert crel.cagr_pct > 5.0
    assert 5.0 < crel.annualized_volatility_pct < 30.0
    assert 0.0 < crel.max_drawdown_pct < 40.0
    assert not crel.insufficient_data


def test_insufficient_history_flag(sample_portfolio):
    pf = Portfolio(
        profile=sample_portfolio.profile,
        funds=[Fund(ticker="GHOST", allocation=1.0)],
    )
    report, warnings = historical_returns(pf)
    assert report.records[0].insufficient_data
    assert any("insufficient" in w.lower() for w in warnings)


def test_risk_metrics_include_sharpe_and_rolling(sample_portfolio):
    report, _ = risk_metrics(sample_portfolio)
    by_ticker = {r.ticker: r for r in report.records}
    crel = by_ticker["CREL"]
    assert -5.0 < crel.sharpe_ratio < 5.0
    assert len(crel.rolling_12m_returns) == 49  # 60 months - 12 + 1
    assert crel.risk_free_rate_pct > 0.0


def test_run_relevant_tools_records_status(sample_portfolio):
    results = run_relevant_tools(sample_portfolio, ["portfolio_analyzer", "risk_metrics"])
    assert results.status["portfolio_analyzer"].status == "ok"
    assert results.status["risk_metrics"].status == "ok"
    assert results.portfolio_analysis is not None
    assert results.risk_metrics is not None


def test_run_relevant_tools_handles_unknown_tool_name(sample_portfolio):
    results = run_relevant_tools(sample_portfolio, ["not_a_tool"])
    assert results.status["not_a_tool"].status == "error"
