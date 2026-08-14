"""Command-line entry point for the Investment Committee Agent.

Usage:
    python main.py "Should I redeem the Pinnacle Small Cap fund?"
    python main.py "Am I over-diversified?" --portfolio data/sample_portfolio.json

Set LLM_API_KEY in .env for real model calls; without it the deterministic
mock LLM is used (great for demos and offline runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pydantic import ValidationError

from config import DATA_DIR, setup_logging
from graph import run_investment_committee
from models import Fund, Portfolio, UserProfile

log = logging.getLogger(__name__)


def load_portfolio(path: str) -> Portfolio:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return Portfolio(
            profile=UserProfile(**data["profile"]),
            funds=[Fund(**fund) for fund in data["funds"]],
        )
    except FileNotFoundError:
        log.error("Portfolio file not found: %s", path)
        sys.exit(f"Portfolio file not found: {path}")
    except json.JSONDecodeError as exc:
        log.error("Portfolio file is not valid JSON: %s", exc)
        sys.exit(f"Portfolio file is not valid JSON: {exc}")
    except KeyError as exc:
        log.error("Portfolio file is missing required field: %s", exc)
        sys.exit(f"Portfolio file is missing required field: {exc}")
    except ValidationError as exc:
        log.error("Portfolio failed validation: %s", exc)
        sys.exit(f"Portfolio failed validation: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Ask the investment committee a portfolio question.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask the investment committee",
    )
    parser.add_argument(
        "--portfolio",
        default=str(DATA_DIR / "sample_portfolio.json"),
        help="Path to a portfolio JSON file",
    )
    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        print(
            "\nExamples:\n"
            '  python main.py "Should I redeem the Pinnacle Small Cap fund?"\n'
            '  python main.py "Am I over-diversified?"\n'
            '  python main.py "Should I consolidate my portfolio?"'
        )
        sys.exit(0)

    setup_logging()
    portfolio = load_portfolio(args.portfolio)
    report = run_investment_committee(args.question, portfolio)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
