"""Generate the synthetic dataset used by the tools.

Run:  python scripts/make_dataset.py

Everything is derived from a single random seed, so the dataset is fully
reproducible. The generator encodes simple financial realism:
  - each fund has an annual expected return and volatility by category
  - all equity funds are driven by one shared "market factor" so they are
    correlated (this is what makes overlap analysis meaningful)
  - debt/liquid funds have low volatility and a tiny market exposure
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

SEED = 42
N_MONTHS = 60  # five years of monthly returns

# fund -> (name, category, benchmark, expense_ratio_pct, aum_million,
#          annual_return_pct, annual_volatility_pct, market_beta)
FUNDS = {
    "CREL": ("Crest Large Cap Fund", "Large Cap Equity", "LargeCap 250 Index", 1.05, 8600, 12.0, 15.0, 1.00),
    "ATL500": ("Atlas Index 500 Fund", "Large Cap Equity", "Broad 500 Index", 0.35, 15400, 12.0, 16.0, 1.00),
    "SUMFLEX": ("Summit Flexi Cap Fund", "Flexi Cap Equity", "Flexi Cap Index", 1.10, 6300, 13.0, 16.5, 1.05),
    "HORIZMID": ("Horizon Mid Cap Fund", "Mid Cap Equity", "MidCap 150 Index", 1.35, 4100, 14.0, 20.0, 1.15),
    "PINSC": ("Pinnacle Small Cap Fund", "Small Cap Equity", "SmallCap 250 Index", 1.55, 2300, 15.0, 24.0, 1.20),
    "NOVAGL": ("Nova Global Equity Fund", "International Equity", "MSCI World Index", 1.40, 2900, 11.0, 15.0, 0.80),
    "BALADV": ("Balance Advantage Fund", "Balanced Hybrid", "Hybrid 40:60 Index", 1.20, 4800, 10.0, 9.0, 0.45),
    "SECCORP": ("Secure Corporate Bond Fund", "Corporate Bond", "Corporate Bond Index", 0.60, 6700, 7.8, 3.5, 0.08),
    "PRUGILT": ("Prudent Gilt Fund", "Government Bond", "G-Sec Index", 0.75, 3100, 7.2, 5.0, 0.05),
    "LIQTR": ("Liquid Treasury Fund", "Liquid", "Treasury Bill Index", 0.25, 9900, 6.8, 1.2, 0.02),
}

# ticker -> (name, sector)  -- fictional issuers so no real facts are asserted
HOLDINGS = {
    "MEGTECH": ("MegaTech Ltd", "Technology"),
    "NATBANK": ("National Bank Ltd", "Banking"),
    "FIRSTBNK": ("First National Bank", "Banking"),
    "SOLENER": ("Solar Energy Corp", "Energy"),
    "LIFEPH": ("LifePharma Ltd", "Pharmaceuticals"),
    "AUTODRV": ("AutoDrive Ltd", "Automobile"),
    "TELCONE": ("TelecomOne Ltd", "Telecom"),
    "CONSSTAR": ("ConsumerStar Ltd", "Consumer"),
    "INFRBLD": ("InfraBuild Ltd", "Infrastructure"),
    "METLWRK": ("MetalWorks Ltd", "Metals & Mining"),
    "FINEDGE": ("FinEdge Ltd", "Financial Services"),
    "CHEMPLS": ("ChemPlus Ltd", "Chemicals"),
    "RETAILHUB": ("RetailHub Ltd", "Retail"),
    "ENGNCO": ("EngineCo Ltd", "Industrials"),
    "GOVT": ("Government Securities", "Government"),
}

# fund -> list of (holding, weight) ; weights within a fund sum to ~1
FUND_HOLDINGS = {
    "CREL": [("MEGTECH", 0.22), ("NATBANK", 0.20), ("FIRSTBNK", 0.16),
             ("SOLENER", 0.12), ("LIFEPH", 0.14), ("TELCONE", 0.16)],
    "ATL500": [("MEGTECH", 0.15), ("NATBANK", 0.14), ("FIRSTBNK", 0.12),
               ("SOLENER", 0.10), ("LIFEPH", 0.09), ("TELCONE", 0.08),
               ("CONSSTAR", 0.08), ("INFRBLD", 0.07), ("METLWRK", 0.06),
               ("FINEDGE", 0.06), ("CHEMPLS", 0.03), ("RETAILHUB", 0.02)],
    "SUMFLEX": [("MEGTECH", 0.18), ("NATBANK", 0.17), ("FIRSTBNK", 0.13),
                ("SOLENER", 0.10), ("AUTODRV", 0.12), ("TELCONE", 0.10),
                ("CONSSTAR", 0.09), ("FINEDGE", 0.11)],
    "HORIZMID": [("AUTODRV", 0.18), ("TELCONE", 0.16), ("CONSSTAR", 0.14),
                 ("INFRBLD", 0.13), ("METLWRK", 0.12), ("FINEDGE", 0.12),
                 ("ENGNCO", 0.15)],
    "PINSC": [("METLWRK", 0.20), ("ENGNCO", 0.18), ("RETAILHUB", 0.17),
              ("CHEMPLS", 0.16), ("CONSSTAR", 0.15), ("INFRBLD", 0.14)],
    "NOVAGL": [("MEGTECH", 0.25), ("SOLENER", 0.20), ("LIFEPH", 0.15),
               ("TELCONE", 0.15), ("FINEDGE", 0.15), ("RETAILHUB", 0.10)],
    "BALADV": [("NATBANK", 0.14), ("FIRSTBNK", 0.11), ("MEGTECH", 0.10),
               ("GOVT", 0.45), ("CHEMPLS", 0.08), ("INFRBLD", 0.12)],
    "SECCORP": [("GOVT", 0.35), ("NATBANK", 0.25), ("FIRSTBNK", 0.20),
                ("FINEDGE", 0.20)],
    "PRUGILT": [("GOVT", 1.00)],
    "LIQTR": [("GOVT", 1.00)],
}


def generate_monthly_returns(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return {ticker: np.ndarray of N_MONTHS monthly returns}."""
    market_factor = rng.normal(loc=0.008, scale=0.042, size=N_MONTHS)
    monthly: dict[str, np.ndarray] = {}
    for ticker, (_, _, _, _, _, ann_ret, ann_vol, beta) in FUNDS.items():
        # All annual figures are percentages -> convert to decimal monthly.
        monthly_vol = ann_vol / 100 / np.sqrt(12)
        idio_var = max(monthly_vol**2 - (beta * 0.042) ** 2, 0.0)
        idio_std = max(np.sqrt(idio_var), 0.0005)
        alpha = ann_ret / 100 / 12 - beta * 0.008
        idio = rng.normal(loc=0.0, scale=idio_std, size=N_MONTHS)
        monthly[ticker] = alpha + beta * market_factor + idio
    return monthly


def build_metadata() -> pd.DataFrame:
    rows = [
        {
            "ticker": t,
            "name": n,
            "category": cat,
            "benchmark": bench,
            "expense_ratio_pct": exp,
            "aum_million": aum,
        }
        for t, (n, cat, bench, exp, aum, *_rest) in FUNDS.items()
    ]
    return pd.DataFrame(rows)


def build_holdings() -> pd.DataFrame:
    rows = [
        {"fund_ticker": fund, "holding_ticker": h, "holding_name": HOLDINGS[h][0],
         "sector": HOLDINGS[h][1], "weight": w}
        for fund, holdings in FUND_HOLDINGS.items()
        for h, w in holdings
        if w > 0
    ]
    return pd.DataFrame(rows)


def build_returns(monthly: dict[str, np.ndarray]) -> pd.DataFrame:
    months = pd.period_range(start="2021-08", periods=N_MONTHS, freq="M")
    rows = []
    for ticker, series in monthly.items():
        for i, ret in enumerate(series):
            rows.append({"ticker": ticker, "month": str(months[i]), "return_pct": round(ret * 100, 4)})
    return pd.DataFrame(rows)


SAMPLE_PORTFOLIO = {
    "profile": {"age": 35, "risk_profile": "moderate",
                "goals": ["retirement", "child education"]},
    "funds": [
        {"ticker": "CREL", "name": "Crest Large Cap Fund", "allocation": 0.25},
        {"ticker": "SUMFLEX", "name": "Summit Flexi Cap Fund", "allocation": 0.15},
        {"ticker": "BALADV", "name": "Balance Advantage Fund", "allocation": 0.20},
        {"ticker": "SECCORP", "name": "Secure Corporate Bond Fund", "allocation": 0.15},
        {"ticker": "HORIZMID", "name": "Horizon Mid Cap Fund", "allocation": 0.15},
        {"ticker": "PINSC", "name": "Pinnacle Small Cap Fund", "allocation": 0.10},
    ],
}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    build_metadata().to_csv(DATA / "fund_metadata.csv", index=False)
    build_holdings().to_csv(DATA / "fund_holdings.csv", index=False)
    build_returns(generate_monthly_returns(rng)).to_csv(DATA / "fund_returns.csv", index=False)
    (DATA / "sample_portfolio.json").write_text(
        json.dumps(SAMPLE_PORTFOLIO, indent=2), encoding="utf-8"
    )
    print(f"Dataset written to {DATA}")


if __name__ == "__main__":
    main()
