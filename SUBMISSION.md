# Investment Committee Agent — Submission Write-up

**Repo:** https://github.com/shreyachauhan07/Investment-Committee-Agent
**Headline commit:** `507cbf3` · **Tests:** 44/44 passing · **Evaluation:** 12/12 checks passing

---

## What it is

An explainable, multi-agent financial advisory system. A user asks a portfolio
question ("Should I redeem the Pinnacle Small Cap fund?"), and a graph of
specialized agents answers it with:

- a **deterministic confidence score**,
- a **final recommendation**,
- **traceable evidence** for every claim, and
- **preserved disagreement** between advisors.

Orchestration is done with **LangGraph**; every intermediate and final output is a
**validated Pydantic object**; all financial math is **computed by tools** (numpy/pandas),
never guessed by the model.

## Architecture

```
User Question
    │
    ▼
┌────────────┐    selects tools    ┌────────────┐
│   Planner  │ ──────────────────▶ │   Tools    │  portfolio_analyzer, fund_metadata,
└────────────┘                     └────────────┘  historical_returns, risk_metrics
    │                                   │
    │                                   ▼
    │                            evidence digest
    ▼
┌────────────────────────────────────────────┐
│            Specialist Committee           │
│  conservative · growth · cost_efficiency  │
│  devils_advocate                          │
└────────────────────────────────────────────┘
    │            │
    ▼            ▼
┌────────────┐  ┌──────────────┐
│ Consensus  │  │  Confidence  │  deterministic formula over agreement + tool health
└────────────┘  └──────────────┘
    │
    ▼
┌────────────┐
│ Finalize   │  FinalReport (JSON) with evidence, assumptions, disagreements
└────────────┘
```

Pipeline in `graph.py`: `plan → tools → committee → consensus → finalize`, run as a
LangGraph `StateGraph` with a single shared `CommitteeState`.

## How every assignment requirement is met

| Requirement | Implementation |
|---|---|
| Python + LangGraph | `graph.py` — `StateGraph` with `CommitteeState`, typed nodes, conditional edges |
| LLM API call | `agents/llm.py` — OpenAI-compatible client (`LLM_API_KEY`/`LLM_BASE_URL`), JSON-schema validated via Pydantic |
| Planner Agent | `agents/planner.py` — reads the question, picks the smallest relevant tool subset |
| Conservative Advisor | `agents/specialists.py` + `agents/prompts.py` (capital preservation mandate) |
| Growth Advisor | same module, growth mandate |
| Cost & Efficiency Advisor | same module, fees/overlap/simplicity mandate |
| Devil's Advocate | same module, challenges the consensus |
| Consensus Agent | `agents/consensus.py` — aggregates, records agreements + disagreements |
| Disagreements preserved | surfaced in output as `disagreements[]`, never averaged away |
| Confidence scoring | `confidence.py` — deterministic, explainable, never LLM-generated |
| Portfolio Analyzer | `tools/portfolio_analyzer.py` — diversification, overlap, category exposure |
| Fund Metadata | `tools/fund_metadata.py` — expense ratios, categories, AUM, benchmarks |
| Historical return analysis | `tools/historical_returns.py` — CAGR, volatility, max drawdown |
| Risk metrics | `tools/risk_metrics.py` — Sharpe ratio (risk-free = 6.5%), drawdowns |
| Sample portfolio | `data/sample_portfolio.json` + reproducible generator `scripts/make_dataset.py` (seed 42) |
| Structured output | `models.py` — `PlannerPlan`, `CommitteeOpinion`, `ConsensusOutput`, `FinalReport` |
| Environment variables for keys | `.env` via `config.py` (`LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_MOCK`, …) |
| Git/GitHub-compatible structure | standard layout, `.gitignore` (excludes `.env`, caches), committed and pushed |
| Failure handling | retry + repair loop; per-agent fallbacks; tool status; graceful degradation |
| Evaluation | `tests/` (44 tests) + `scripts/evaluate.py` (12 checks incl. LLM outage) |

## Design decisions worth knowing

1. **Tools compute, LLMs interpret.** Sharpe ratios, drawdowns, overlap (Jaccard),
   category exposure, and effective diversification are *calculated* in code. The
   model only reads and explains the numbers — which is what makes the evidence
   trustworthy and auditable.
2. **A mock LLM keeps the system testable and free.** With no `LLM_API_KEY`,
   `get_llm_client()` returns a deterministic `MockLLMClient` that parses the exact
   prompt blocks the real client uses. All tests and the demo run offline and are
   reproducible; set a key to get real model opinions.
3. **Disagreement is structural, not accidental.** Each advisor has its own system
   prompt and mandate and answers independently on the same evidence digest. The
   confidence score then *measures* agreement instead of assuming it.
4. **Confidence is deterministic and explainable:**
   `0.5·agreement + 0.3·tool_health + 0.2·data_health − 0.1 if disagreements − 0.1 if tool error`,
   clamped to [0,1]. It degrades sensibly when tools fail or advisors split.
5. **Provider-agnostic.** Any OpenAI-compatible endpoint works (OpenAI, Azure,
   Ollama, LM Studio) via `LLM_BASE_URL`.

## Reliability & failure handling

- LLM calls: retries with exponential backoff; the response is re-validated against
  the Pydantic model and re-requested on malformed output (repair loop).
- If the planner fails → a safe default plan (all tools, all specialists) is used.
- If a specialist fails → an explicit fallback opinion is recorded; the run continues.
- If consensus fails → a computed fallback (stance from the opinions, disagreement
  preserved) is returned.
- Tools report status (`ok`/`partial`/`error`); partial or failed tools feed the
  confidence score and surface as `tool_warnings`/`tool_errors` in the report.

## Evaluation results

- `python -m pytest -q` → **44 passed** (models, tools, confidence, graph end-to-end, failure paths).
- `python scripts/evaluate.py` → **12/12 checks passed**: all 10 assignment example
  questions produce a well-formed report with evidence and preserved disagreement,
  a simulated total LLM outage degrades to an explicit fallback, and an empty
  portfolio surfaces warnings.
- `python -m pyflakes .` → clean. Reproducible because the mock LLM is deterministic.

## How to run

```bash
pip install -r requirements.txt          # langgraph, openai, pandas, pydantic, ...
python main.py "Should I redeem the Pinnacle Small Cap fund?"
python -m pytest -q
python scripts/evaluate.py
# optional: copy .env.example to .env, set LLM_API_KEY (and LLM_BASE_URL for local models)
```

## Key files

```
graph.py                LangGraph wiring + evidence/assumption builders
agents/llm.py           OpenAI client + MockLLMClient + retry/repair loop
agents/prompts.py       role system prompts, evidence digest
agents/{planner,specialists,consensus}.py
tools/                  the four financial tools + reliability registry
models.py               all Pydantic schemas incl. FinalReport
confidence.py           deterministic confidence formula
config.py               env vars, paths, logging
data/                   synthetic dataset (seed 42, 10 funds, 60 months)
tests/                  44 tests
scripts/evaluate.py     12-check evaluation harness
```

## Notes & limitations

- All fund, holdings, and market data are **synthetic** (fictional fund names) and
  for demonstration only — not a basis for real investment decisions.
- The mock LLM produces plausible-but-template-based opinions; use a real model for
  genuine qualitative reasoning.
- Advisors run sequentially; running them in parallel is a straightforward
  extension and would cut latency further (currently ~0.5s for all 10 questions
  with the mock).
