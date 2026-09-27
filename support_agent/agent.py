"""The facade every entry point uses: `scripts/run_batch.py`, the Gradio app,
and the tests. Keep the graph behind it so those three never drift apart.
"""

import time
import traceback

from . import config, llm, trace
from .graph import SupportGraph
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover - older LangGraph installs
    MemorySaver = None
from .retrieval import get_retriever


class SupportAgent:
    def __init__(self, checkpointer=None, interrupt_before=None):
        self.retriever = get_retriever()          # load the index once, reuse it
        # Keep conversation state by default. Callers can still inject a different
        # checkpointer for database-backed deployment or deterministic tests.
        self.checkpointer = checkpointer if checkpointer is not None else (
            MemorySaver() if MemorySaver is not None else None)
        self.interrupt_before = interrupt_before
        self._graphs = {}

    def resolve(self, query_record, thread_id=None):
        """Run one customer query end to end. Returns a trace dict."""
        started = time.time()
        before = llm.usage_snapshot()

        graph = SupportGraph(customer_id=query_record.get("customer_id"),
                             checkpointer=self.checkpointer,
                             interrupt_before=self.interrupt_before,
                             retriever=self.retriever)
        if thread_id:
            self._graphs[thread_id] = graph
        try:
            final = graph.run(query_id=query_record["query_id"],
                              query=query_record["query"],
                              customer_id=query_record.get("customer_id"),
                              history=query_record.get("history") or [],
                              thread_id=thread_id)
        except Exception as e:                        # noqa: BLE001 — never lose a row
            # Print the whole thing to the console: the one-line version in the
            # trace tells you what broke but never where, which makes debugging
            # far harder than it needs to be.
            traceback.print_exc()
            tb = traceback.extract_tb(e.__traceback__)
            where = f" at {tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}" if tb else ""
            final = {"answer": f"(agent error: {type(e).__name__}: {e}{where})",
                     "route": "escalated", "citations": [],
                     "actions": list(graph.ctx.actions), "escalation": None,
                     "steps": ["error"], "hits": []}

        after = llm.usage_snapshot()
        meta = {
            "latency_s": round(time.time() - started, 2),
            "llm_calls": after["calls"] - before["calls"],
            "cached_calls": after["cached"] - before["cached"],
            "tokens": (after["prompt_tokens"] - before["prompt_tokens"]
                       + after["completion_tokens"] - before["completion_tokens"]),
            "model": config.TOOL_MODEL,
            "retrieval_mode": config.RETRIEVAL_MODE,
            "graph_path": final.get("steps", []),
            "retrieved": [h["doc_id"] for h in final.get("hits", [])],
            "dropped_citations": final.get("dropped_citations", []),
        }
        record = trace.build(
            query_id=query_record["query_id"],
            route=final.get("route", "resolved"),
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            actions=final.get("actions", []),
            escalation=final.get("escalation"),
            meta=meta)
        record["_debug"] = {"hits": final.get("hits", [])}   # dropped before submission
        return record

    def resume(self, thread_id, approved, note=""):
        """IMPLEMENTED TODO 6c — resume a paused action after human approval/rejection.

        If you built the "stop and ask" step, the graph pauses before doing
        something risky and its half-finished state is saved. This function is
        what starts it again once a person has decided.

        Three LangGraph calls do the work:

            graph.get_state(cfg)              # what was it about to do?
            graph.update_state(cfg, {...})    # change it, or record the refusal
            graph.invoke(None, cfg)           # None means "carry on from where you stopped"

        The Approve and Reject buttons in the web app call this function.
        """
        graph = self._graphs.get(thread_id)
        if graph is None:
            raise KeyError(f"no saved conversation for thread {thread_id!r}")
        blocked = [a for a in graph.ctx.actions if a.get("status") == "blocked"]
        if not blocked:
            return {"approved": approved, "message": "No action is waiting for approval."}
        action = blocked[-1]
        if not approved:
            graph.ctx.log(action["tool"], action["args"], "rejected",
                          reason=note or "human_rejected")
            return {"approved": False, "action": action["tool"], "status": "rejected"}
        subject = action["args"].get("order_id") or action["args"].get("customer_id") or ""
        graph.ctx.approved_actions.add((action["tool"], subject))
        result = graph.tools[action["tool"]].invoke(action["args"])
        return {"approved": True, "action": action["tool"], "status": "executed",
                "result": str(result)}
