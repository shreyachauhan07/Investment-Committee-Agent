"""Central access point for the CSV dataset (cached in memory)."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config import HOLDINGS_FILE, METADATA_FILE, RETURNS_FILE


@lru_cache(maxsize=1)
def load_metadata() -> pd.DataFrame:
    return pd.read_csv(METADATA_FILE)


@lru_cache(maxsize=1)
def load_holdings() -> pd.DataFrame:
    return pd.read_csv(HOLDINGS_FILE)


@lru_cache(maxsize=1)
def load_returns() -> pd.DataFrame:
    return pd.read_csv(RETURNS_FILE)
