"""Risk Metrics Tool.

Returns Sharpe ratio, maximum drawdown, annualized volatility and
trailing 12-month rolling returns per fund.
"""

from __future__ import annotations

from config import RISK_FREE_RATE_PCT
from models import (
    Portfolio,
    RiskMetricRecord,
    RiskMetricsReport,
    RollingReturn,
)
from tools.data_loader import load_returns
from tools.metrics import (
    annualized_volatility,
    max_drawdown,
    rolling_returns,
    sharpe_ratio,
)

MIN_MONTHS = 12


def risk_metrics(portfolio: Portfolio) -> tuple[RiskMetricsReport, list[str]]:
    warnings: list[str] = []
    returns = load_returns()
    records: list[RiskMetricRecord] = []

    for f in portfolio.funds:
        subset = returns[returns["ticker"] == f.ticker].sort_values("month")
        monthly = subset["return_pct"].to_numpy(dtype=float) / 100.0 if not subset.empty else None

        if monthly is None or len(monthly) < MIN_MONTHS:
            warnings.append(
                f"Fund '{f.ticker}' has insufficient return history to compute "
                "risk metrics."
            )
            records.append(
                RiskMetricRecord(
                    ticker=f.ticker,
                    sharpe_ratio=0.0,
                    max_drawdown_pct=0.0,
                    annualized_volatility_pct=0.0,
                    risk_free_rate_pct=RISK_FREE_RATE_PCT,
                    insufficient_data=True,
                )
            )
            continue

        rolling = rolling_returns(monthly, window=12)
        rolling_rows = [
            RollingReturn(
                end_month=str(subset.iloc[11 + k]["month"]),
                return_pct=round(r * 100, 2),
            )
            for k, r in enumerate(rolling)
        ]

        records.append(
            RiskMetricRecord(
                ticker=f.ticker,
                sharpe_ratio=round(sharpe_ratio(monthly, RISK_FREE_RATE_PCT), 2),
                max_drawdown_pct=round(max_drawdown(monthly) * 100, 2),
                annualized_volatility_pct=round(
                    annualized_volatility(monthly) * 100, 2
                ),
                rolling_12m_returns=rolling_rows,
                risk_free_rate_pct=RISK_FREE_RATE_PCT,
                insufficient_data=False,
            )
        )

    return RiskMetricsReport(records=records), warnings
