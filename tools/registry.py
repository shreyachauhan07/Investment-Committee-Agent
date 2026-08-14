"""Tool registry and execution.

`run_relevant_tools` is the single place the graph calls to execute tools.
It maps a tool name to its function, catches any exception, and records a
per-tool status so the rest of the pipeline can degrade gracefully instead
of crashing (or, worse, reporting misleading results).
"""

from __future__ import annotations

import logging

from models import Portfolio, ToolResults, ToolStatus
from tools.fund_metadata import get_fund_metadata
from tools.historical_returns import historical_returns
from tools.portfolio_analyzer import analyze_portfolio
from tools.risk_metrics import risk_metrics

log = logging.getLogger(__name__)

# name -> (callable, description) — the planner picks names from here.
TOOL_FUNCS = {
    "portfolio_analyzer": analyze_portfolio,
    "fund_metadata": get_fund_metadata,
    "historical_returns": historical_returns,
    "risk_metrics": risk_metrics,
}

TOOL_DESCRIPTIONS = {
    "portfolio_analyzer": (
        "Analyze the portfolio: allocation, category exposure, fund overlap, "
        "and diversification metrics."
    ),
    "fund_metadata": (
        "Return category, benchmark, expense ratio and AUM for each fund."
    ),
    "historical_returns": (
        "Return CAGR, annualized volatility and maximum drawdown per fund."
    ),
    "risk_metrics": (
        "Return Sharpe ratio, max drawdown and trailing 12-month rolling "
        "returns per fund."
    ),
}

# Where each tool's typed result is stored inside ToolResults.
_RESULT_ATTRS = {
    "portfolio_analyzer": "portfolio_analysis",
    "fund_metadata": "fund_metadata",
    "historical_returns": "historical_returns",
    "risk_metrics": "risk_metrics",
}


def run_relevant_tools(portfolio: Portfolio, tool_names: list[str]) -> ToolResults:
    results = ToolResults()

    for name in tool_names:
        fn = TOOL_FUNCS.get(name)
        if fn is None:
            results.status[name] = ToolStatus(
                status="error", error=f"Unknown tool requested: {name}"
            )
            continue

        try:
            data, warnings = fn(portfolio)
            results.status[name] = ToolStatus(
                status="ok" if not warnings else "partial", warnings=warnings
            )
            setattr(results, _RESULT_ATTRS[name], data)
        except Exception as exc:  # noqa: BLE001 — tool failure must not kill pipeline
            log.exception("Tool '%s' failed", name)
            results.status[name] = ToolStatus(status="error", error=str(exc))

    return results
