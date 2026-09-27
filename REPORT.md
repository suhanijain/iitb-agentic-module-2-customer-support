# HW2 Report — SUHANI JAIN

## The short version

| | score on the 24 practice questions |
|---|---:|
| what we gave you | **about 70 / 100** |
| what I ended up with | **95.83 / 100** |

The saved run uses `openai/gpt-4o-mini`, `RETRIEVAL_MODE=hybrid`, `TOP_K=6`, and structured chunks of **1,200 characters**. Query translation and reranking are disabled; answer verification is enabled (`.env`, `config.py`). Re-scoring its **24 traces** gives **95.83 / 100**, with route, actions, facts and citations each at **0.958**, and **0 safety violations** (`python scripts/evaluate_dev.py --traces dev_traces.jsonl --show 24`). Trace metadata sums to **11.60 seconds**, **32 cached calls**, and **0 recorded tokens or uncached LLM calls**; it does not provide a meaningful uncached dollar-cost estimate (`dev_traces.jsonl`).

## What I changed, and what each change was worth

| What I did | Practice score after | Kept it? |
|---|---:|---|
| Assignment's stated starting point (not re-measured here) | **about 70** | — |
| Integrated implementation, saved 24-trace run | **95.83** | Yes |
| Current hybrid retrieval-only check, top-6 | **18/29 = 0.621 recall** | Yes |

The clearest remaining failure is `dev-004`: the agent escalated instead of resolving and omitted the required **30-day** fact, so the case scored **0.00** (`dev_traces.jsonl`, evaluator output). `multi_turn` and both injection cases scored **100.00**, but the repository has no controlled per-TODO ablation log. I therefore cannot attribute an overall score change to an individual implementation step; the retrieval-only check also shows that the current search configuration misses **11 of 29** gold citation targets (`dev_traces.jsonl`, `scripts/evaluate_dev.py`).

## 1. Searching the handbook

The shipped retrieval path is hybrid: **dense + BM25 + RRF**, with `TOP_K=6`, `CANDIDATE_K=8`, `CHUNK_SIZE=1200` and `RRF_K=60` (`retrieval.py`, `config.py`). With the current index and settings, `python scripts/evaluate_dev.py --retrieval-only` reports **18/29 = 0.621 recall@6**. The repository does not contain saved results for separate chunk-size or dense-versus-hybrid ablations, so I cannot claim which setting caused an improvement.

For the two trap sections, I kept them indexed so the agent can recognise what a customer saw, but filtered `community` as **untrusted** and `archive_returns_2024` as **superseded** before they can support the current answer (`config.py`, `retrieval.py`). This was chosen over simply deleting them because the assignment explicitly tests recognition of stale/untrusted material; relevance does not make a source authoritative (`policy.py`, `retrieval.py`).

## 2. Checking the answer is true

`node_verify` splits the draft into factual claims, checks them against retrieved evidence, and uses a **0.75 faithfulness threshold**; below the threshold the draft is suppressed and the graph escalates/fails closed (`graph.py`). This protects questions the handbook does not cover by preventing an unsupported answer from being presented as fact (`graph.py`).

The current trace metadata records **11.60 seconds** and cached calls, but no uncached token usage; this is not an isolated measure of verification overhead. No verification-on/off cost or score ablation is saved in the repository. I would leave verification switched **on** because it fails closed on unsupported drafts, while noting that the current end-to-end score is **95.83**, not a perfect score (`graph.py`, `dev_traces.jsonl`).

## 3. Tools and safety

The tool loop binds the available tools, executes returned tool calls, appends `ToolMessage` observations, and stops after `MAX_TOOL_STEPS=6` (`graph.py`, `config.py`). The design supports dependent calls such as `get_order` → `check_return_eligibility`, and also handles unknown tools, bad arguments and tool failures without letting the run loop forever (`graph.py`, `tools.py`).

The two fake-instruction sources are the `community` handbook section and a support-ticket note returned by `get_ticket_history`; detection looks for instruction-like phrases, untrusted text is fenced, and write operations are protected by code-owned approval (`policy.py`, `tools.py`). In the saved run, `dev-021` and `dev-022` each scored **1.00**, with **0 safety violations** overall (`dev_traces.jsonl`, evaluator output). The current repository does not preserve before/after measurements for the injection defenses.

A reworded attack that avoids the detector's regex patterns could still evade lexical detection; however, a protected write such as a refund above **₹5,000** is independently blocked by `requires_approval`, so the model cannot grant authority just by changing wording (`policy.py`, `tools.py`).

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
                            ****       *    *** *****
                                ***    *  ******
                                 +---------+
                                 | __end__ |
                                 +---------+
```

The branches separate **direct deterministic actions**, **retrieval-backed answers**, **missing-information clarification**, and **human escalation**; state is accumulated through the graph and conversation memory is provided by `MemorySaver` by default (`graph.py`, `agent.py`). The final `multi_turn` category scored **100.00** across its two dev cases (`dev_traces.jsonl`); an isolated before/after numeric multi-turn experiment was not recorded, so I have not invented one.

One implemented human-approval example is: `issue_refund` above **₹5,000** is first **blocked** by `policy.requires_approval`, the UI can call `SupportAgent.resume(thread_id, approved=True)`, the approval is stored as a one-shot permission, and the tool then executes (`policy.py`, `tools.py`, `agent.py`).

| | agent fetched a human | agent handled it alone |
|---|---|---|
| **should have fetched a human** | correct: high-value refund, safety, legal, repeat failure | missed — dangerous because authority or safety rules were bypassed |
| **should have handled it alone** | wasted time: deterministic status/eligibility cases | correct: routine order and policy cases |

## Evidence

The saved trace file is `dev_traces.jsonl`; re-scoring it gives **95.83 / 100 (n=24)**. The evaluator output has route/actions/facts/citations each at **0.958**, **0 safety violations**, and a weakest category of **stale_policy at 50.00** because `dev-004` failed (`python scripts/evaluate_dev.py --traces dev_traces.jsonl --show 24`). `evaluation_operational.json` reports **98.33** and does not match a fresh score of the saved traces; I use the reproducible trace-based result here. No evaluator screenshot is present in the repository.

## How to run this

```bash
pip install -r requirements.txt
# Set OPENROUTER_API_KEY in your environment or a local .env file; do not commit the key.
python scripts/build_index.py --force --strategy structured --chunk-size 1200
python scripts/run_batch.py --in data/test_queries.jsonl --out submission.jsonl
python scripts/evaluate_dev.py --traces dev_traces.jsonl --show 24
python -m pytest tests -q
python -c "from support_agent.graph import draw; draw()"
```

The evaluation settings are in `config.py`; the 24 saved traces are in `dev_traces.jsonl`; and the generated graph is in `graph.txt`.
