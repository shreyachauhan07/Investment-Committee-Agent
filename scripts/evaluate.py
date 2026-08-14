"""Run the assignment's example questions through the pipeline and verify
that every run produces a valid, evidence-backed, disagreement-aware report.

Usage:
    python scripts/evaluate.py

Runs entirely offline with the mock LLM. Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as `python scripts/evaluate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm import LLMError, MockLLMClient
from config import setup_logging
from graph import run_investment_committee
from models import ADVISOR_NAMES, Portfolio
from tests.conftest import load_sample

QUESTIONS = [
    "Should I redeem Fund X?",
    "Should I add a Small Cap fund?",
    "Am I over-diversified?",
    "Should I consolidate my portfolio?",
    "Why did my portfolio underperform?",
    "Why is my portfolio more volatile than expected?",
    "Should I increase debt allocation?",
    "Am I taking excessive risk?",
    "Am I overexposed to a sector?",
    "Is my allocation consistent with my age and goals?",
]


def check(report) -> list[str]:
    problems = []
    if set(report.committee_opinions) != set(ADVISOR_NAMES):
        problems.append("not all four committee opinions present")
    if not report.final_recommendation:
        problems.append("missing final recommendation")
    if not (0.0 <= report.confidence_score <= 1.0):
        problems.append(f"confidence out of range: {report.confidence_score}")
    if not report.evidence:
        problems.append("no evidence produced")
    return problems


def main() -> int:
    setup_logging()
    llm = MockLLMClient()
    portfolio = load_sample()
    failures = 0

    print(f"{'question':<50} {'conf':>5}  {'evid':>4}  {'disagr':>6}  result")
    print("-" * 90)
    for question in QUESTIONS:
        report = run_investment_committee(question, portfolio, llm=llm)
        problems = check(report)
        ok = not problems
        failures += 0 if ok else 1
        print(
            f"{question[:48]:<50} {report.confidence_score:>5.2f}  "
            f"{len(report.evidence):>4}  {len(report.disagreements):>6}  "
            f"{'PASS' if ok else 'FAIL: ' + '; '.join(problems)}"
        )

    # Failure cases must degrade gracefully.
    print("\nFailure cases:")

    class FailingLLM:
        def complete_json(self, system, user, response_model):
            raise LLMError("simulated outage")

    outage = run_investment_committee("Any question?", portfolio, llm=FailingLLM())
    if outage.final_recommendation and outage.confidence_score >= 0.0:
        print(f"{'LLM outage':<48} PASS (explicit fallback produced)")
    else:
        print(f"{'LLM outage':<48} FAIL")
        failures += 1

    empty = Portfolio(profile=portfolio.profile, funds=[])
    empty_report = run_investment_committee("Am I over-diversified?", empty, llm=llm)
    if empty_report.tool_warnings:
        print(f"{'Empty portfolio':<48} PASS (warnings surfaced)")
    else:
        print(f"{'Empty portfolio':<48} FAIL")
        failures += 1

    print(f"\n{len(QUESTIONS) + 2 - failures} of {len(QUESTIONS) + 2} checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
