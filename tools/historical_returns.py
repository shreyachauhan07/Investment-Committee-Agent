"""Historical Return Tool.

Returns CAGR, annualized volatility and maximum drawdown per fund
from the monthly return history. Funds with < 12 months of data are
flagged as having insufficient history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import HistoricalReturnRecord, HistoricalReturnsReport, Portfolio
from tools.data_loader import load_returns
from tools.metrics import annualized_return, annualized_volatility, max_drawdown

MIN_MONTHS = 12


def historical_returns(
    portfolio: Portfolio,
) -> tuple[HistoricalReturnsReport, list[str]]:
    warnings: list[str] = []
    returns = load_returns()
    records: list[HistoricalReturnRecord] = []

    for f in portfolio.funds:
        monthly = _monthly_series(returns, f.ticker)
        if monthly is None or len(monthly) < MIN_MONTHS:
            warnings.append(
                f"Fund '{f.ticker}' has insufficient return history to compute "
                "historical returns."
            )
            records.append(
                HistoricalReturnRecord(
                    ticker=f.ticker,
                    name=f.name or f.ticker,
                    cagr_pct=0.0,
                    annualized_volatility_pct=0.0,
                    max_drawdown_pct=0.0,
                    months_of_data=0 if monthly is None else len(monthly),
                    insufficient_data=True,
                )
            )
            continue

        records.append(
            HistoricalReturnRecord(
                ticker=f.ticker,
                name=f.name or f.ticker,
                cagr_pct=round(annualized_return(monthly) * 100, 2),
                annualized_volatility_pct=round(
                    annualized_volatility(monthly) * 100, 2
                ),
                max_drawdown_pct=round(max_drawdown(monthly) * 100, 2),
                months_of_data=len(monthly),
                insufficient_data=False,
            )
        )

    return HistoricalReturnsReport(records=records), warnings


def _monthly_series(
    returns: pd.DataFrame, ticker: str
) -> np.ndarray | None:
    subset = returns[returns["ticker"] == ticker].sort_values("month")
    if subset.empty:
        return None
    return subset["return_pct"].to_numpy(dtype=float) / 100.0
