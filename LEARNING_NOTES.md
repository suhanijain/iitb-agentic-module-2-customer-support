# HW2 Support Agent — Learning Notes

## Executive outcome

The supplied controlled ablation is **98.33/100**:
route = 1.000, actions = 1.000, facts = 0.9583, citations = 0.9583.

The two partial rows are:
- `dev-001`: 0.85 — citation component 0.00 because an untrusted `community` citation surfaced.
- `dev-004`: 0.75 — the current 30-day rule was omitted from the generated answer.

If those two rows become 1.00, the arithmetic gain is:

`(1.00 - 0.85)/24 + (1.00 - 0.75)/24 = 1.6667`

so the same 24-case ablation would reach **100.00**, assuming no regression.

A fresh provider-backed run is required to confirm the final score. The review environment did not
have the project's LangChain/BM25 dependencies installed, so no end-to-end LLM score was invented.

## Code changes in this review

1. **Grounding completeness — `support_agent/graph.py`**
   The response prompt now explicitly says that a directly supported number, date, limit or duration
   should be included. This is a general rule, not a query-ID special case.

2. **Citation telemetry — `graph.py`, `state.py`, `agent.py`**
   Invalid/untrusted citations are still removed before submission, but are recorded as
   `dropped_citations` so citation errors become measurable.

3. **Default LangGraph memory — `agent.py`**
   `MemorySaver` is used when available if the caller does not supply another checkpointer.

4. **Regression test — `tests/test_assignment_features.py`**
   Added a test proving invalid citations are excluded and recorded.

No new framework, query-ID branching, copied gold answers, or practice-answer lookup was added.

## TODO map and score impact

| TODO | Files | Rubric | Max | Evidence | Possible impact |
|---|---|---|---:|---|---|
| 1 | `kb.py` | Retrieval | 25 | Structured 1,200/top-6 retained; BM25 recall 25/29 | Better topic boundaries; limited ceiling on a small handbook |
| 2 | `retrieval.py` | Retrieval | 25 | BM25 + RRF + metadata filtering | Better exact-code retrieval and source safety; measure optional rerank/translation |
| 3 | `graph.py` | Generation/faithfulness | 20 | 0.75 verification gate + fail-closed escalation | Reduces unsupported answers; costs an extra model call |
| 4 | `graph.py` | Tools/safety | part of 25 | Bounded tool loop + dependent calls + error handling | Improves action reliability; loop cap controls cost/runaway behaviour |
| 5 | `policy.py` | Tools/safety | part of 25 | Detection + untrusted fencing + code-owned approval | Prevents injection from becoming authority |
| 6 | `graph.py`, `agent.py` | Orchestration | 30 | Branches + multi-turn + HITL + checkpointing | Improves routing/state continuity; human approval adds latency |

Rubric marks are category maxima, not a guaranteed additive gain from each change.

## Experiments and variations

| Experiment | Result / learning |
|---|---|
| Supplied trace | 70.31 |
| First combined implementation | 69.00 — later model rewrite could discard actions/facts |
| Fixed 600 / top-4 | BM25 recall 20/29 = 0.690 |
| Structured 600 / top-4 | 17/29 = 0.586 |
| Structured 1,200 / top-4 | 23/29 = 0.793 |
| Structured 1,200 / top-6 | 25/29 = 0.862; retained |
| Controlled orchestration ablation | 98.33; route/actions = 1.000 |
| Translation | Compare ON/OFF on the same 24 cases; record score, tokens, latency |
| Reranking | Compare ON/OFF against added LLM cost |
| Verification | Compare ON/OFF; record unsupported-answer rate, tokens, latency |
| Tool cap | Try 3/6/8 and inspect loops vs completion |
| RRF k | Try 20/60/100 |
| TOP_K | Try 4/6/8 with `--retrieval-only` first |

## The key software-development learning

**Use the model for uncertainty; use code for authority.**

Examples:
- order status → record lookup
- refund authority → Python guardrail
- unsupported policy question → retrieval + verification + escalation
- unknown tool → return an observation instead of crashing
- multi-turn context → state/checkpointer

For Python developers, a LangGraph node is still just a function:
`state -> dictionary of changes`.

`SupportGraph` is a class that owns related dependencies. `SupportState` is the contract between
nodes. Dependency injection lets tests and production supply different retrievers/checkpointers.
Guardrails are ordinary Python rules with explicit authority.

## Reproduce the work

```text
python scripts/check_env.py
pip install -r requirements.txt
python scripts/build_index.py --force --strategy structured
RETRIEVAL_MODE=dense python scripts/evaluate_dev.py --retrieval-only
RETRIEVAL_MODE=hybrid python scripts/evaluate_dev.py --retrieval-only
python scripts/evaluate_dev.py --show 24
python -m pytest tests -q
python -c "from support_agent.graph import draw; draw()"
```

Change one conceptual thing at a time, run focused tests, measure, run the full dev set, inspect the
worst rows, record score + latency + tokens, then keep or revert the change.

Never add `if query_id == ...`, hard-code answers, read `dev_gold.jsonl` at runtime, or otherwise
tune directly to practice questions.
