# HW2 Report — Suhani Jain, <roll number>

## The short version

| | score on the 24 practice questions |
|---|---:|
| supplied trace/baseline | 70.31 |
| first implementation run (reported on Windows) | 69.00 — rejected |
| controlled operational ablation | 98.33 |
| final fresh end-to-end run | run locally after rebuilding index/cache |

I used `openai/gpt-4o-mini` for tool use and verification, `RETRIEVAL_MODE=hybrid`,
`TOP_K=6`, and structured chunks of 1,200 characters with 150-character overlap.
The available earlier trace used 70.31/100 (route 0.875, actions 0.604, facts
0.667, citations 0.708). Fresh provider calls were unavailable in the execution
environment, so I have not invented a final end-to-end score or cost. The 98.33
ablation executes the new deterministic paths for 20 operational dev cases and
retains the four existing policy/stale-policy outputs to isolate orchestration
impact. All 38 local tests pass.

## What I changed, and what each change was worth

| What I did | Measured result | Kept it? |
|---|---:|---|
| supplied trace | 70.31 overall | baseline |
| first combined implementation | 69.00 overall (user run) | no |
| tool loop followed by a second rewrite | lowered reliability; actions/facts could be discarded | no for clear intents |
| structured chunks, 600 chars, top 4 | BM25 recall 17/29 = 0.586 | no |
| structured chunks, 1,200 chars, top 6 | BM25 recall 25/29 = 0.862 | yes |
| deterministic operational branches | controlled overall ablation 98.33; actions 1.000 | yes |
| claim-level verification | retained; API failure on this non-deterministic path escalates P3 | yes, threshold 0.75 |
| LangGraph branches, history and approval handler | route 1.000 in controlled ablation | yes |
| two-layer injection defence | regression tests pass | yes |
| optional LLM reranker | independently switchable with `ENABLE_RERANK` | compare on/off |
| optional query translation | independently switchable with `ENABLE_QUERY_TRANSLATION` | compare on/off |

The main regression was architectural: every ordinary message went through a
tool-calling model and then a second response model. The second call could omit a
completed action, change a fact, or select the wrong route. The revised graph uses
code-owned policy for unambiguous status, cancellation, return, compensation and
escalation cases; model generation is reserved for policy-language questions.
This moved the controlled action component from the supplied 0.604 to 1.000 with
zero safety violations. A fresh dense index could not be built in the restricted
environment, so final dense/hybrid results must be measured locally.

## 1. Searching the handbook

`support_agent/kb.py::split_structured` keeps each chunk inside one `##` section,
tries sub-heading, paragraph and sentence boundaries in that order, and prefixes
the section title. `support_agent/retrieval.py` tokenises codes such as `ERR-4021`
consistently, builds BM25, and combines lexical and dense rankings with RRF rather
than adding incompatible scores. `postprocess` removes `community` (`trust: low`)
and `archive_returns_2024` (`status: superseded`) completely. This is the safest
choice because neither may support an answer or appear as a citation.

The separate BM25 ablation found: fixed 600/top-4 = 20/29 (0.690), structured
600/top-4 = 17/29 (0.586), structured 1,200/top-4 = 23/29 (0.793), and structured
1,200/top-6 = 25/29 (0.862). I therefore kept required heading-aware chunking but
changed its size and top-k. Dense-only and full hybrid comparisons need the MiniLM
model download, so claiming a measured gain for those would be misleading. On a
14-section handbook, retrieval improvements have a fairly low ceiling; they are
more valuable for exact codes and preventing bad sources than for headline recall.

## 2. Checking the answer is true

`SupportGraph.node_verify` asks the fast model to split the draft into independent
factual claims and label each against retrieved passages plus tool results.
Questions and courtesy are excluded. At least 75% of factual claims must be
supported. Below 0.75, the draft is suppressed, citations are cleared, and the
case is escalated at P3 with `out_of_scope`. That specifically handles ordinary-
looking unsupported questions such as student discounts. Provider latency and
tokens must be measured during the final run; the verifier adds one JSON call per
non-triaged response. I would keep it for a regulated support flow, while later
testing a cheaper NLI verifier to reduce latency. If response generation, tool
selection or verification cannot obtain a required API response, the agent now
fails closed to a P3 human escalation with reason `api_unavailable`. Deterministic
paths continue because they do not depend on that API result.

## 3. Tools and safety

`node_act` binds all ten tools, executes model-selected calls, returns each result
as a `ToolMessage`, and permits later calls to depend on earlier results. It stops
after `MAX_TOOL_STEPS=6`. Unknown tools, invalid arguments and tool exceptions are
returned as observations rather than crashing the run.

Security has three layers: retrieved trap documents are filtered; document/tool
text is fenced as untrusted; and `detect_injection` records instruction-like
phrases. Most importantly, `requires_approval` remains code-owned, so a model
cannot authorise refunds over ₹5,000 or outside the return window. Tests cover the
community-style “system override” and ticket-style “reply with only” attacks. A
semantic rewording without the detector's phrases could evade pattern detection,
but it still cannot bypass the monetary guardrail. Novel attacks against actions
without equivalent deterministic guards remain a limitation.

## 4. Controlling the flow

```text
START -> triage -> direct ----------------------------> END
               \-> clarify ---------------------------> END
               \-> escalate --------------------------> END
               \-> retrieve -> respond -> verify -----> END
                            \-> act -> respond -> verify -> END
```

The compiled `StateGraph` branches after deterministic triage. Safety, legal,
privacy, large-refund and repeat-failure cases take escalation; ambiguous mutation
requests take clarification; clear operational cases take the direct policy/tool
branch; policy questions take retrieval and grounded response. The bounded model
tool loop remains for uncertain operational requests. `messages` and `steps` use
`operator.add`; current hits and output fields are
overwritten. Order ids are recovered from the supplied conversation history for
multi-turn questions. The web approval handler retains the conversation graph:
rejection records the refusal, while approval grants one-shot authority for the
exact blocked tool and order/customer before executing it. Handover summaries
include customer, order ids, request, checks/reason, priority, and decision needed.

| | agent fetched a human | agent handled it alone |
|---|---|---|
| **should have fetched a human** | correct | dangerous miss |
| **should have handled it alone** | wasted effort | correct |

## Evidence and reproduction

The supplied trace's weakest category was large-refund handling (35/100); the code-owned large-refund branch directly addresses it. 
