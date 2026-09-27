# HW2 Report — SUHANI JAIN


## The short version

| | score on the 24 practice questions |
|---|---|
| what we gave you | about 70 |
| what I ended up with | 95.83 |

I used the model `openai/gpt-4o-mini`, with `RETRIEVAL_MODE=hybrid` and chunks of `1200` characters (`config.py`, `retrieval.py`). The saved trace run scored 95.83 / 100 on the 24 practice questions, with route, actions, facts, and citations each at 0.958 and 0 safety violations (`dev_traces.jsonl`, `scripts/evaluate_dev.py`). I did not have a reliable uncached dollar estimate from the saved run because the trace metadata records cached calls and no uncached token count (`dev_traces.jsonl`).

## What I changed, and what each change was worth

| What I did | Practice score after | Kept it? |
|---|---|---|
| (nothing yet — the starting point) | ~70 | — |
| TODO 4 — the tool-calling loop | 95.83 | Yes |
| TODO 2 — keyword search and the trap sections | 18/29 = 0.621 recall | Yes |
| TODO 3 — checking the answer is supported | 95.83 | Yes |
| TODO 6 — branches, memory, human approval | 100.00 on multi-turn checks | Yes |
| TODO 5 — defending against fake instructions | 100.00 on injection checks | Yes |
| TODO 1 — cutting the handbook at its headings | 18/29 = 0.621 recall | Yes |

The biggest change for me was the verification gate and tool loop, because the saved run reached 95.83 / 100 while still keeping the multi-turn and injection checks at 100.00 (`dev_traces.jsonl`, `scripts/evaluate_dev.py`). I think that combination mattered most because it stopped unsupported answers from being emitted and it prevented dangerous tool calls from running without policy approval (`graph.py`, `policy.py`, `tools.py`).

The part that did not work as well as I wanted was retrieval on its own: the retrieval-only check still missed 11 of 29 gold targets, so the search setting was not strong enough by itself (`scripts/evaluate_dev.py`). I think the main reason was that the top-k was still too small for some long-tail handbook facts, and the stale or untrusted sections needed stricter trust filtering (`retrieval.py`, `policy.py`).

## 1. Searching the handbook

I used hybrid retrieval with dense search, BM25, and RRF, with `TOP_K=6`, `CANDIDATE_K=8`, `CHUNK_SIZE=1200`, and `RRF_K=60` (`retrieval.py`, `config.py`). With that setup, `python scripts/evaluate_dev.py --retrieval-only` reported 18/29 = 0.621 recall@6, so I could still see a retrieval gap even when the agent logic itself was working (`scripts/evaluate_dev.py`).

For the two trap sections, I kept them indexed so the agent could recognize what the customer had actually seen, but I marked `community` as untrusted and `archive_returns_2024` as superseded before those sections could support a final answer (`config.py`, `retrieval.py`). I chose that instead of deleting them because the assignment explicitly checks whether the system can distinguish stale or untrusted text from the current handbook policy (`policy.py`).

## 2. Checking the answer is true

I split each draft answer into factual claims and checked them against the retrieved evidence using a 0.75 faithfulness threshold (`graph.py`). If a draft fell below that cutoff, I suppressed it and made the graph fail closed instead of presenting an unsupported answer as fact (`graph.py`).

This mattered especially for questions the handbook does not cover, because the agent would not claim a fact it could not verify from the retrieved support (`graph.py`). I would leave verification switched on because it reduces unsupported claims and the saved trace run still scored 95.83 / 100 with 0 safety violations (`dev_traces.jsonl`, `scripts/evaluate_dev.py`).

## 3. Tools and safety

My tool loop binds the available tools, executes the returned tool calls, appends the observations to the state, and stops after `MAX_TOOL_STEPS=6` (`graph.py`, `config.py`). A concrete example is a dependent sequence such as `get_order` followed by `check_return_eligibility`, which only runs after the first tool returns a real order ID (`graph.py`, `tools.py`).

I treated the `community` handbook section and the support-ticket note returned by `get_ticket_history` as hostile instruction sources, fenced that text as untrusted, and blocked any write action that was not explicitly authorized by the policy path (`policy.py`, `tools.py`). In the saved run, the hidden-instruction cases scored 1.00 each, and the run had 0 safety violations overall (`dev_traces.jsonl`, `scripts/evaluate_dev.py`).

A reworded attack that avoids the detector’s regex patterns could still pass a naive lexical filter, so I kept approval checks on sensitive actions such as refunds above ₹5,000 (`policy.py`, `tools.py`). That is the key defense: even if the wording changes, the request still fails unless the policy layer approves it.

## 4. Controlling the flow

```text
                                +-----------+
                                | __start__ |
                                +-----------+
                                       *
                                       *
                                       *
                                  +--------+
                                ..| triage |....
                            ....  +--------+... .....
                       .....           .       ...   .......
                   ....               .           ..        .....
                ...                   .             ...          .......
      +----------+                    .                ..               ...
      | retrieve |                    .                 .                 .
      +----------+                    .                 .                 .
       ..       .                     .                 .                 .
      .          .                    .                 .                 .
     .            ..                  .                 .                 .
+-----+             .                 .                 .                 .
| act |           ..                  .                 .                 .
+-----+          .                    .                 .                 .
       **       .                     .                 .                 .
         *    ..                      .                 .                 .
          *  .                        .                 .                 .
      +---------+                     .                 .                 .
      | respond |                     .                 .                 .
      +---------+                     .                 .                 .
            *                         .                 .                 .
            *                         .                 .                 .
            *                         .                 .                 .
      +--------+                +---------+        +--------+       +----------+
      | verify |***             | clarify |        | direct |    ***| escalate |
      +--------+   ****         +---------+      **+--------+****   +----------+
                       *****          *        **    *******
                            ****       *    *** ******
                                ***    *  ******
                                 +---------+
                                 | __end__ |
                                 +---------+
```

I split the flow into direct action, retrieval-backed response, clarification, verification, and escalation paths, with conversation memory provided by `MemorySaver` (`graph.py`, `agent.py`). The saved trace run scored 100.00 on the multi-turn dev cases, and I did not preserve a separate before/after numeric measurement for the memory change (`dev_traces.jsonl`).

One action I implemented was an approval gate: `issue_refund` above ₹5,000 is blocked first, then a human can approve through `SupportAgent.resume(thread_id, approved=True)`, and the tool executes only after that permission is granted (`policy.py`, `tools.py`, `agent.py`).

| | agent fetched a human | agent handled it alone |
|---|---|---|
| should have fetched a human | correct | missed — the dangerous mistake |
| should have handled it alone | wasted someone's time | correct |

## Evidence

I used the saved trace file `dev_traces.jsonl` and re-scored it with `python scripts/evaluate_dev.py --traces dev_traces.jsonl --show 24`, which gave 95.83 / 100 across 24 practice questions (`scripts/evaluate_dev.py`). The route, actions, facts, and citations categories each scored 0.958, and there were 0 safety violations in the saved run (`dev_traces.jsonl`).

The weakest category in the saved run was stale_policy at 50.00, which matches the remaining case where the agent escalated instead of resolving the stale-policy issue (`dev_traces.jsonl`, `scripts/evaluate_dev.py`). I did not include a screenshot because no app screenshot is present in the repository, and I did not want to invent one.

## How to run this

```bash
pip install -r requirements.txt && cp .env.example .env
python scripts/build_index.py --force
python scripts/run_batch.py --in data/test_queries.jsonl --out submission.jsonl
```

I used the local environment and verification scripts to reproduce the reported numbers, including `python scripts/evaluate_dev.py --traces dev_traces.jsonl --show 24` and `python -m pytest tests -q` (`scripts/evaluate_dev.py`, `tests`). I also used GitHub Copilot while I was implementing the agent and checking the report text in the workspace, mainly to review logic and tighten the wording.
