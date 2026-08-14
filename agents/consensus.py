"""Consensus Agent.

Aggregates the four independent opinions: names genuine agreements and
disagreements, explains the trade-offs, and produces a single final
recommendation. Disagreements are preserved, not smoothed away.

On LLM failure it returns a conservative fallback that says so explicitly.
"""

from __future__ import annotations

import logging

from agents.llm import LLMError, LLMClient
from agents.prompts import consensus_prompt
from models import CommitteeOpinions, ConsensusOutput, ToolResults

log = logging.getLogger(__name__)


def _opinions_as_dicts(opinions: CommitteeOpinions) -> list[dict]:
    return [
        opinions.conservative.model_dump(),
        opinions.growth.model_dump(),
        opinions.cost_efficiency.model_dump(),
        opinions.devils_advocate.model_dump(),
    ]


def reach_consensus(
    question: str,
    opinions: CommitteeOpinions,
    tool_results: ToolResults,
    llm: LLMClient,
) -> ConsensusOutput:
    system, user = consensus_prompt(
        question, _opinions_as_dicts(opinions), tool_results, ConsensusOutput
    )
    try:
        consensus = llm.complete_json(system, user, ConsensusOutput)
    except LLMError:
        log.exception("Consensus agent failed; returning explicit fallback")
        return ConsensusOutput(
            agreements=[],
            disagreements=["Consensus could not be computed due to a system failure."],
            final_recommendation=(
                "No final recommendation was produced: the consensus agent failed. "
                "Review the committee opinions and tool evidence manually."
            ),
            preferred_stance="unknown",
        )

    log.info(
        "Consensus: %d agreement(s), %d disagreement(s), stance=%s",
        len(consensus.agreements),
        len(consensus.disagreements),
        consensus.preferred_stance,
    )
    return consensus
