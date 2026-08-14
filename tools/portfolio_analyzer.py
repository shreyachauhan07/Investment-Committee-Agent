"""Portfolio Analyzer Tool.

Given a portfolio, produce:
  - per-fund allocation
  - per-category allocation (with weighted expense ratio)
  - pairwise fund overlap (sum of shared-holding weight intersections)
  - diversification metrics (HHI, effective fund count, max exposures)
"""

from __future__ import annotations

from models import (
    CategoryAllocation,
    DiversificationMetrics,
    FundAllocation,
    OverlapPair,
    Portfolio,
    PortfolioAnalysis,
)
from tools.data_loader import load_holdings, load_metadata


def analyze_portfolio(portfolio: Portfolio) -> tuple[PortfolioAnalysis, list[str]]:
    warnings: list[str] = []

    if not portfolio.funds:
        return PortfolioAnalysis(), ["Portfolio is empty; no analysis possible."]

    meta = load_metadata().set_index("ticker")
    holdings_df = load_holdings()

    fund_allocations: list[FundAllocation] = []
    for f in portfolio.funds:
        if f.ticker not in meta.index:
            warnings.append(f"Fund '{f.ticker}' has no metadata; skipped from analysis.")
            continue
        row = meta.loc[f.ticker]
        fund_allocations.append(
            FundAllocation(
                ticker=f.ticker,
                name=f.name or str(row["name"]),
                category=str(row["category"]),
                weight_pct=round(f.allocation * 100, 2),
                expense_ratio_pct=float(row["expense_ratio_pct"]),
            )
        )

    if not fund_allocations:
        return PortfolioAnalysis(fund_allocations=[]), (
            warnings + ["No recognized funds in portfolio."]
        )

    # Category roll-up: weight, fund count, weighted expense ratio.
    categories: dict[str, list[FundAllocation]] = {}
    for fa in fund_allocations:
        categories.setdefault(fa.category, []).append(fa)
    category_allocations = []
    for cat, members in categories.items():
        total_weight = sum(fa.weight_pct for fa in members)
        category_allocations.append(
            CategoryAllocation(
                category=cat,
                weight_pct=round(total_weight, 2),
                num_funds=len(members),
                weighted_expense_ratio_pct=round(
                    sum(fa.weight_pct * fa.expense_ratio_pct for fa in members)
                    / total_weight
                    if total_weight > 0
                    else 0.0,
                    2,
                ),
            )
        )
    category_allocations.sort(key=lambda ca: -ca.weight_pct)

    # Pairwise overlap from the holdings dataset.
    holdings_by_fund = {
        ticker: dict(zip(g["holding_ticker"], g["weight"]))
        for ticker, g in holdings_df.groupby("fund_ticker")
    }
    tickers = [fa.ticker for fa in fund_allocations]
    overlap_pairs = _compute_overlap(tickers, holdings_by_fund)

    div = _diversification_metrics(fund_allocations, category_allocations, overlap_pairs)

    total_alloc = sum(f.allocation for f in portfolio.funds)
    if abs(total_alloc - 1.0) > 0.01:
        warnings.append(f"Fund allocations sum to {total_alloc:.2f}, not 1.0.")

    return (
        PortfolioAnalysis(
            fund_allocations=fund_allocations,
            category_allocations=category_allocations,
            overlap_pairs=overlap_pairs,
            diversification=div,
        ),
        warnings,
    )


def _compute_overlap(
    tickers: list[str], holdings_by_fund: dict[str, dict[str, float]]
) -> list[OverlapPair]:
    """Overlap score between two funds = sum of min(weight_a, weight_b) over
    shared holdings. 0.0 = no shared names, 1.0 = identical portfolios."""
    pairs: list[OverlapPair] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            ha, hb = holdings_by_fund.get(a, {}), holdings_by_fund.get(b, {})
            shared = set(ha) & set(hb)
            if not shared:
                continue
            score = sum(min(ha[h], hb[h]) for h in shared)
            pairs.append(
                OverlapPair(
                    fund_a=a,
                    fund_b=b,
                    overlap_score=round(score, 4),
                    shared_holdings=sorted(shared),
                )
            )
    return sorted(pairs, key=lambda p: -p.overlap_score)


def _diversification_metrics(
    fund_allocations: list[FundAllocation],
    category_allocations: list[CategoryAllocation],
    overlap_pairs: list[OverlapPair],
) -> DiversificationMetrics:
    weights = [fa.weight_pct / 100 for fa in fund_allocations]
    hhi = sum(w * w for w in weights)
    avg_overlap = (
        sum(p.overlap_score for p in overlap_pairs) / len(overlap_pairs)
        if overlap_pairs
        else 0.0
    )
    return DiversificationMetrics(
        num_funds=len(fund_allocations),
        num_categories=len(category_allocations),
        max_fund_weight_pct=round(max(fa.weight_pct for fa in fund_allocations), 2),
        max_category_weight_pct=round(
            max(ca.weight_pct for ca in category_allocations), 2
        ),
        herfindahl_index=round(hhi, 4),
        effective_number_of_funds=round(1 / hhi if hhi > 0 else 0.0, 2),
        average_pairwise_overlap=round(avg_overlap, 4),
    )
