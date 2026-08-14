"""Transparent, deterministic confidence scoring.

The confidence score is COMPUTED, not asked for. The LLM never says "I'm
82% sure"; instead we combine measurable signals:

    score = 0.5 * agreement
          + 0.3 * tool_health
          + 0.2 * data_health
          - 0.1 if the committee reported disagreements
          - 0.1 if any tool errored

  * agreement    - fraction of advisors whose stance matches the consensus
                   preferred stance. 1.0 = unanimous, 0.0 = no support.
  * tool_health  - mean health of the tools the planner requested
                   (ok=1.0, partial=0.5, error=0.0).
  * data_health  - fraction of funds with sufficient return history.
                   0.5 (neutral) when history was not requested.

Result is clamped to [0, 1]. The formula is documented in the README so
anyone can recompute and verify a given score.
"""

from __future__ import annotations

from models import (
    ADVISOR_NAMES,
    CommitteeOpinions,
    ConsensusOutput,
    ToolResults,
)

# Weights sum to 1.0 across the three health signals.
W_AGREEMENT = 0.5
W_TOOLS = 0.3
W_DATA = 0.2
PENALTY_DISAGREEMENT = 0.1
PENALTY_TOOL_ERROR = 0.1

_STATUS_WEIGHTS = {"ok": 1.0, "partial": 0.5, "error": 0.0}


def compute_confidence(
    consensus: ConsensusOutput,
    opinions: CommitteeOpinions,
    tool_results: ToolResults,
) -> float:
    agreement = _agreement(consensus, opinions)
    tool_health = _tool_health(tool_results)
    data_health = _data_health(tool_results)

    score = (
        W_AGREEMENT * agreement
        + W_TOOLS * tool_health
        + W_DATA * data_health
    )

    if consensus.disagreements:
        score -= PENALTY_DISAGREEMENT
    if any(s.status == "error" for s in tool_results.status.values()):
        score -= PENALTY_TOOL_ERROR

    return round(min(max(score, 0.0), 1.0), 2)


def _agreement(consensus: ConsensusOutput, opinions: CommitteeOpinions) -> float:
    preferred = consensus.preferred_stance
    if not preferred or preferred == "unknown":
        return 0.0
    stances = [getattr(opinions, advisor).stance for advisor in ADVISOR_NAMES]
    if not stances:
        return 0.0
    return sum(1 for s in stances if s == preferred) / len(stances)


def _tool_health(tool_results: ToolResults) -> float:
    statuses = list(tool_results.status.values())
    if not statuses:
        return 0.0
    return sum(_STATUS_WEIGHTS.get(s.status, 0.0) for s in statuses) / len(statuses)


def _data_health(tool_results: ToolResults) -> float:
    historical = tool_results.historical_returns
    if historical and historical.records:
        healthy = sum(0.0 if r.insufficient_data else 1.0 for r in historical.records)
        return healthy / len(historical.records)
    return 0.5  # neutral: history was not part of this analysis
