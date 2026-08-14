"""Schema validation tests: invalid input must be caught at the boundary."""

import pytest
from pydantic import ValidationError

from models import (
    CommitteeOpinion,
    ConsensusOutput,
    FinalReport,
    Fund,
    PlannerPlan,
    Portfolio,
    UserProfile,
)


def test_valid_portfolio_parses(sample_portfolio):
    assert len(sample_portfolio.funds) == 6
    assert sample_portfolio.profile.age == 35
    assert sample_portfolio.profile.risk_profile == "moderate"


def test_fund_rejects_negative_allocation():
    with pytest.raises(ValidationError):
        Fund(ticker="X", allocation=-0.1)


def test_fund_rejects_allocation_above_limit():
    with pytest.raises(ValidationError):
        Fund(ticker="X", allocation=5.0)


def test_user_profile_rejects_invalid_risk_profile():
    with pytest.raises(ValidationError):
        UserProfile(age=35, risk_profile="mega-aggressive")


def test_user_profile_rejects_unrealistic_age():
    with pytest.raises(ValidationError):
        UserProfile(age=10)


def test_empty_funds_allowed_but_detectable(sample_portfolio):
    empty = Portfolio(profile=sample_portfolio.profile, funds=[])
    assert empty.funds == []


def test_planner_plan_accepts_valid_tools():
    plan = PlannerPlan(
        intent="Should I redeem a fund?",
        relevant_tools=["fund_metadata", "historical_returns"],
        required_information=["Expense ratio"],
        specialists=["conservative", "growth", "cost_efficiency", "devils_advocate"],
    )
    assert len(plan.relevant_tools) == 2


def test_committee_opinion_requires_advisor():
    with pytest.raises(ValidationError):
        CommitteeOpinion(stance="hold", recommendation_summary="keep")


def test_consensus_output_allows_empty_lists():
    out = ConsensusOutput()
    assert out.agreements == []
    assert out.final_recommendation == ""


def test_final_report_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        FinalReport(
            question="q",
            committee_opinions={},
            confidence_score=1.5,
            final_recommendation="x",
        )
