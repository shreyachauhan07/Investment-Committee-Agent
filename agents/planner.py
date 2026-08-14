"""Planner Agent.

Decides what the user is asking, which tools are relevant, what
information is required, and which specialists should weigh in.

On LLM failure it falls back to a safe default plan (run every tool,
consult every specialist) rather than crashing the pipeline.
"""

from __future__ import annotations

import logging

from agents.llm import LLMError, LLMClient
from agents.prompts import planner_prompt
from models import ADVISOR_NAMES, PlannerPlan, Portfolio
from tools.registry import TOOL_DESCRIPTIONS

log = logging.getLogger(__name__)


def plan(question: str, portfolio: Portfolio, llm: LLMClient) -> PlannerPlan:
    system, user = planner_prompt(question, portfolio, PlannerPlan)
    try:
        plan_out = llm.complete_json(system, user, PlannerPlan)
        log.info("Planner intent: %s | tools=%s", plan_out.intent, plan_out.relevant_tools)
        return plan_out
    except LLMError:
        log.exception("Planner failed; using default plan (all tools, all specialists)")
        return PlannerPlan(
            intent=question,
            relevant_tools=list(TOOL_DESCRIPTIONS),
            required_information=["No plan could be produced; using all available evidence."],
            specialists=list(ADVISOR_NAMES),
        )
