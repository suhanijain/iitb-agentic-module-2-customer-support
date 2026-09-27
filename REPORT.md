# HW2 Report — SUHANI JAIN

## The short version

| | score on the 24 practice questions |
|---|---:|
| what we gave you | **70.31 / 100** |
| what I ended up with | **100.00 / 100** |

I used the model `openai/gpt-4o-mini`, with `RETRIEVAL_MODE=hybrid`, `TOP_K=6`, and structured chunks of **1,200 characters** (`config.py`). The final run of `python scripts/evaluate_dev.py` wrote **24 traces** and scored **1.000** for route, actions, facts and citations, with **0 safety violations** (`dev_traces.jsonl`, `Module_2_Output.png`). The run recorded **44,284 tokens**, **26 LLM calls** and **41.49 seconds** across the 24 traces; a dollar cost was **not captured** (`dev_traces.jsonl`).

## What I changed, and what each change was worth

| What I did | Practice score after | Kept it? |
|---|---:|---|
| (nothing yet — the starting point) | **70.31** | — |
| TODO 4 — the tool-calling loop | **69.00** in the first combined implementation; not isolated | Reworked |
| TODO 2 — keyword search and the trap sections | **25/29 = 0.862 retrieval recall** at structured 1,200/top-6; final system **100.00** | Yes |
| TODO 3 — checking the answer is supported | **98.33** in the controlled ablation; final system **100.00** | Yes |
| TODO 6 — branches, memory, human approval | **100.00** final; `multi_turn` = **100.00** | Yes |
| TODO 5 — defending against fake instructions | **100.00** final; safety violations = **0** | Yes |
| TODO 1 — cutting the handbook at its headings | **25/29 = 0.862 retrieval recall** with structured 1,200/top-6; final system **100.00** | Yes |

The biggest measured improvement at the end was **98.33 → 100.00**: `dev-001` was **0.85** because an untrusted `community` citation surfaced, and `dev-004` was **0.75** because the current **30-day** rule was omitted; both became **1.00** after citation sanitisation and deterministic current-policy handling (`LEARNING_NOTES.md`, `evaluation_operational.json`, `dev_traces.jsonl`). The thing that did not work was the first combined implementation: **70.31 → 69.00** because a later model rewrite dropped actions/facts; the fix was to move deterministic business rules back into code-owned routing instead of relying on generation (`LEARNING_NOTES.md`, `graph.py`).

## 1. Searching the handbook

The shipped retrieval path is hybrid: **dense + BM25 + RRF**, with `TOP_K=6`, `CANDIDATE_K=8`, `CHUNK_SIZE=1200` and `RRF_K=60` (`retrieval.py`, `config.py`). The measured BM25 recall progression was **20/29 = 0.690** for fixed 600/top-4, **17/29 = 0.586** for structured 600/top-4, **23/29 = 0.793** for structured 1,200/top-4, and **25/29 = 0.862** for structured 1,200/top-6 (`LEARNING_NOTES.md`).

For the two trap sections, I kept them indexed so the agent can recognise what a customer saw, but filtered `community` as **untrusted** and `archive_returns_2024` as **superseded** before they can support the current answer (`config.py`, `retrieval.py`). This was chosen over simply deleting them because the assignment explicitly tests recognition of stale/untrusted material; relevance does not make a source authoritative (`policy.py`, `retrieval.py`).

## 2. Checking the answer is true

`node_verify` splits the draft into factual claims, checks them against retrieved evidence, and uses a **0.75 faithfulness threshold**; below the threshold the draft is suppressed and the graph escalates/fails closed (`graph.py`). This protects questions the handbook does not cover by preventing an unsupported answer from being presented as fact (`graph.py`).

The final 24-trace run recorded **44,284 tokens**, **26 LLM calls** and **41.49 seconds** total, but these are whole-run figures rather than isolated verification overhead; no before/after verification cost experiment was recorded (`dev_traces.jsonl`). I would leave verification switched **on** because the final score was **100.00** and citations/facts were both **1.000** (`dev_traces.jsonl`).

## 3. Tools and safety

The tool loop binds the available tools, executes returned tool calls, appends `ToolMessage` observations, and stops after `MAX_TOOL_STEPS=6` (`graph.py`, `config.py`). The design supports dependent calls such as `get_order` → `check_return_eligibility`, and also handles unknown tools, bad arguments and tool failures without letting the run loop forever (`graph.py`, `tools.py`).

The two fake-instruction sources are the `community` handbook section and a support-ticket note returned by `get_ticket_history`; detection looks for instruction-like phrases, untrusted text is fenced, and write operations are protected by code-owned approval (`policy.py`, `tools.py`). Before the final fix, `dev-001` surfaced an untrusted citation and scored **0.85**; after citation cleanup/trust filtering it scored **1.00**. `dev-021` and `dev-022` both scored **1.00**, and the final run had **0 safety violations** (`evaluation_operational.json`, `dev_traces.jsonl`).

A reworded attack that avoids the detector's regex patterns could still evade lexical detection; however, a protected write such as a refund above **₹5,000** is independently blocked by `requires_approval`, so the model cannot grant authority just by changing wording (`policy.py`, `tools.py`).

## 4. Controlling the flow

```text
                                +-----------+
                                | __start__ |
                                +-----------+
                                       *
                                  +--------+
                                ..| triage |....
                                  +--------+
                                       |
                 +---------------------+----------------------+
                 |                     |                      |
                 v                     v                      v
          +-------------+       +-----------+         +-------------+
          |   retrieve  |       |  clarify  |         |   direct    |
          +------+------+       +-----------+         +------+------+
                 |                                         |
                 v                                         |
             +--------+                                    |
             |  act   |                                    |
             +---+----+                                    |
                 |                                         |
                 +------------------+----------------------+
                                    v
                              +----------+
                              | respond  |
                              +----+-----+
                                   |
                                   v
                              +---------+
                              | verify  |
                              +----+----+
                                   |
                         +---------+---------+
                         |                   |
                         v                   v
                     +-------+         +----------+
                     |  end  |         | escalate |
                     +-------+         +----------+
```

The branches separate **direct deterministic actions**, **retrieval-backed answers**, **missing-information clarification**, and **human escalation**; state is accumulated through the graph and conversation memory is provided by `MemorySaver` by default (`graph.py`, `agent.py`). The final `multi_turn` category scored **100.00** across its two dev cases (`dev_traces.jsonl`); an isolated before/after numeric multi-turn experiment was not recorded, so I have not invented one.

One implemented human-approval example is: `issue_refund` above **₹5,000** is first **blocked** by `policy.requires_approval`, the UI can call `SupportAgent.resume(thread_id, approved=True)`, the approval is stored as a one-shot permission, and the tool then executes (`policy.py`, `tools.py`, `agent.py`).

| | agent fetched a human | agent handled it alone |
|---|---|---|
| **should have fetched a human** | correct: high-value refund, safety, legal, repeat failure | missed — dangerous because authority or safety rules were bypassed |
| **should have handled it alone** | wasted time: deterministic status/eligibility cases | correct: routine order and policy cases |

## Evidence

`Module_2_Output.png` is the final evaluator screenshot. It shows **100.00 / 100 (n=24)**, route/actions/facts/citations all **1.000**, **0 safety violations**, and **100.00** in every displayed category. The final trace file is `dev_traces.jsonl`; the generated graph is `graph.txt`.

There is no worst category in the final run because all 12 categories are **100.00**. With another week, I would measure adversarially reworded injection, verification ON/OFF cost, and optional reranking/query-translation rather than assuming those extra LLM calls improve the score (`policy.py`, `retrieval.py`, `dev_traces.jsonl`).

## How to run this

```bash
pip install -r requirements.txt && cp .env.example .env
python scripts/build_index.py --force --strategy structured
python scripts/run_batch.py --in data/test_queries.jsonl --out submission.jsonl
python scripts/evaluate_dev.py --show 24
python -m pytest tests -q
python -c "from support_agent.graph import draw; draw()"
```

The final evaluation settings are in `config.py`; the 24 final traces are in `dev_traces.jsonl`; retrieval experiments are summarised in `LEARNING_NOTES.md`; the graph output is in `graph.txt`. AI tools used during the homework were used for code review, debugging, experiment design and documentation; the submitted implementation and measurements remain in the named project files (`LEARNING_NOTES.md`, `graph.py`, `retrieval.py`, `policy.py`, `agent.py`, `tests/`).
