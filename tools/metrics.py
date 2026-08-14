"""Financial calculation helpers shared across tools."""

from __future__ import annotations

import numpy as np

SQRT_12 = np.sqrt(12)


def annualized_return(monthly: np.ndarray) -> float:
    """Annualized (CAGR-style) return from a series of monthly returns."""
    if len(monthly) == 0:
        return 0.0
    total = float(np.prod(1.0 + monthly))
    if total <= 0:
        return -1.0
    return float(total ** (12 / len(monthly)) - 1.0)


def annualized_volatility(monthly: np.ndarray) -> float:
    """Annualized standard deviation of monthly returns."""
    if len(monthly) < 2:
        return 0.0
    return float(np.std(monthly, ddof=1) * SQRT_12)


def max_drawdown(monthly: np.ndarray) -> float:
    """Maximum peak-to-trough decline, returned as a positive percentage (e.g. 22.5)."""
    if len(monthly) == 0:
        return 0.0
    wealth = np.cumprod(1.0 + monthly)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    return float(-np.min(drawdown))


def rolling_returns(monthly: np.ndarray, window: int = 12) -> list[float]:
    """Trailing `window`-period compounded returns for each complete window."""
    out = []
    for i in range(window, len(monthly) + 1):
        out.append(float(np.prod(1.0 + monthly[i - window : i]) - 1.0))
    return out


def sharpe_ratio(monthly: np.ndarray, annual_risk_free_pct: float) -> float:
    """Excess return over the risk-free rate per unit of volatility."""
    vol = annualized_volatility(monthly)
    if vol == 0.0 or len(monthly) < 2:
        return 0.0
    ann_return = float(np.mean(monthly) * 12)
    excess = ann_return * 100 - annual_risk_free_pct
    return float(excess / (vol * 100))
