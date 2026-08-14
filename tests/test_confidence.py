"""Confidence scoring formula tests.

Verify the score responds correctly to each signal:
agreement, tool health, data health, disagreement and tool-error penalties.
"""

import pytest

from confidence import compute_confidence
from models import (
    CommitteeOpinion,
    CommitteeOpinions,
    ConsensusOutput,
    HistoricalReturnRecord,
    HistoricalReturnsReport,
    ToolResults,
    ToolStatus,
)


def _opinions(stances: list[str]) -> CommitteeOpinions:
    def make(advisor, stance):
        return CommitteeOpinion(
            advisor=advisor,
            stance=stance,
            recommendation_summary=stance,
            key_points=[],
            concerns=[],
            evidence=[],
        )

    return CommitteeOpinions(
        conservative=make("conservative", stances[0]),
        growth=make("growth", stances[1]),
        cost_efficiency=make("cost_efficiency", stances[2]),
        devils_advocate=make("devils_advocate", stances[3]),
    )


def _healthy_tools(n: int = 2) -> ToolResults:
    results = ToolResults()
    for i in range(n):
        results.status[f"tool_{i}"] = ToolStatus(status="ok")
    return results


def _tools_with_errors(n_ok: int, n_error: int) -> ToolResults:
    results = ToolResults()
    for i in range(n_ok):
        results.status[f"ok_{i}"] = ToolStatus(status="ok")
    for i in range(n_error):
        results.status[f"bad_{i}"] = ToolStatus(status="error", error="boom")
    return results


def test_unanimous_agreement_high_confidence():
    consensus = ConsensusOutput(preferred_stance="hold")
    opinions = _opinions(["hold", "hold", "hold", "hold"])
    # 0.5*1.0 + 0.3*1.0 + 0.2*0.5(neutral data) = 0.9, no penalties.
    score = compute_confidence(consensus, opinions, _healthy_tools())
    assert score == 0.9


def test_split_agreement_lowers_confidence():
    consensus = ConsensusOutput(preferred_stance="hold")
    opinions = _opinions(["hold", "hold", "redeem", "challenge"])
    score = compute_confidence(consensus, opinions, _healthy_tools())
    assert score < 0.9


def test_no_support_for_stance_gives_zero_agreement():
    consensus = ConsensusOutput(preferred_stance="redeem")
    opinions = _opinions(["hold", "hold", "hold", "hold"])
    # 0.0 agreement + 0.3 tools + 0.1 data, no penalties.
    score = compute_confidence(consensus, opinions, _healthy_tools())
    assert score == 0.4


def test_disagreement_penalty():
    consensus = ConsensusOutput(preferred_stance="hold", disagreements=["split"])
    opinions = _opinions(["hold", "hold", "hold", "hold"])
    with_penalty = compute_confidence(consensus, opinions, _healthy_tools())
    consensus_clean = ConsensusOutput(preferred_stance="hold")
    no_penalty = compute_confidence(consensus_clean, opinions, _healthy_tools())
    assert with_penalty == pytest.approx(no_penalty - 0.1)


def test_tool_error_penalty():
    consensus = ConsensusOutput(preferred_stance="hold")
    opinions = _opinions(["hold", "hold", "hold", "hold"])
    clean = compute_confidence(consensus, opinions, _healthy_tools())
    errored = compute_confidence(consensus, opinions, _tools_with_errors(2, 1))
    assert errored < clean


def test_tool_health_matters_partial_vs_ok():
    consensus = ConsensusOutput(preferred_stance="hold")
    opinions = _opinions(["hold", "hold", "hold", "hold"])
    all_ok = _healthy_tools(4)
    partially = ToolResults()
    for i in range(4):
        partially.status[f"t{i}"] = ToolStatus(status="partial")
    assert compute_confidence(consensus, opinions, all_ok) > compute_confidence(
        consensus, opinions, partially
    )


def test_data_health_uses_historical_records():
    consensus = ConsensusOutput(preferred_stance="hold")
    opinions = _opinions(["hold", "hold", "hold", "hold"])
    results = _healthy_tools(2)
    results.historical_returns = HistoricalReturnsReport(
        records=[
            HistoricalReturnRecord(ticker="A", name="A", cagr_pct=10.0, annualized_volatility_pct=10.0,
                                   max_drawdown_pct=5.0, months_of_data=60, insufficient_data=False),
            HistoricalReturnRecord(ticker="B", name="B", cagr_pct=0.0, annualized_volatility_pct=0.0,
                                   max_drawdown_pct=0.0, months_of_data=3, insufficient_data=True),
        ]
    )
    score = compute_confidence(consensus, opinions, results)
    assert score < 1.0  # 50% data health pulls it down


def test_score_stays_in_range_for_extreme_inputs():
    consensus = ConsensusOutput(preferred_stance="unknown")
    opinions = _opinions(["challenge", "challenge", "challenge", "challenge"])
    results = ToolResults()  # no tools -> tool health 0
    score = compute_confidence(consensus, opinions, results)
    assert 0.0 <= score <= 1.0
