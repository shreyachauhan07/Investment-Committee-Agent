"""Configuration: environment variables, data paths, logging.

Centralizing config keeps secrets out of the codebase and makes the
pipeline easy to run in different environments (local, CI, interview demo).
"""

import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# langchain-core (a LangGraph dependency) warns that its optional pydantic-v1
# path is unsupported on Python 3.14. It is unused here and the warning fires
# at import time, so filter it module-level rather than in setup_logging().
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")

# LLM settings (OpenAI-compatible). Empty base URL = OpenAI's default endpoint.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
# Force the deterministic mock LLM even when an API key is present
# (useful for demos and offline evaluation).
LLM_MOCK = os.getenv("LLM_MOCK", "0") == "1"

# Data files
PORTFOLIO_FILE = DATA_DIR / "sample_portfolio.json"
METADATA_FILE = DATA_DIR / "fund_metadata.csv"
HOLDINGS_FILE = DATA_DIR / "fund_holdings.csv"
RETURNS_FILE = DATA_DIR / "fund_returns.csv"

# Financial assumptions used by the risk tool
RISK_FREE_RATE_PCT = 6.5  # annual, Indian 10-year G-Sec proxy (synthetic context)
ANNUALIZATION_FACTOR = 12  # monthly returns -> annualized


def setup_logging() -> None:
    """Configure a single, simple console logger for the whole app."""
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
