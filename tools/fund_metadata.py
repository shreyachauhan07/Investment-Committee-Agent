"""Fund Metadata Tool.

Returns category, benchmark, expense ratio and AUM for each fund
in the portfolio. Unknown funds are skipped with a warning.
"""

from __future__ import annotations

from models import FundMetadataRecord, FundMetadataReport, Portfolio
from tools.data_loader import load_metadata


def get_fund_metadata(
    portfolio: Portfolio,
) -> tuple[FundMetadataReport, list[str]]:
    warnings: list[str] = []
    meta = load_metadata().set_index("ticker")
    records: list[FundMetadataRecord] = []

    for f in portfolio.funds:
        if f.ticker not in meta.index:
            warnings.append(
                f"Fund '{f.ticker}' is unknown to the dataset; metadata unavailable."
            )
            continue
        row = meta.loc[f.ticker]
        records.append(
            FundMetadataRecord(
                ticker=f.ticker,
                name=str(row["name"]),
                category=str(row["category"]),
                benchmark=str(row["benchmark"]),
                expense_ratio_pct=float(row["expense_ratio_pct"]),
                aum_million=float(row["aum_million"]),
            )
        )

    if not records:
        warnings.insert(0, "No fund metadata could be retrieved.")

    return FundMetadataReport(records=records), warnings
