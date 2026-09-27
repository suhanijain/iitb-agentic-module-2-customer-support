"""Every setting in one place. Change things here, not scattered through the code.

You can also override any of these from the command line without editing the
file, which is handy when you want to try a few values quickly:

    RETRIEVAL_MODE=hybrid TOP_K=6 python scripts/evaluate_dev.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KB_DIR = DATA / "kb"
RECORDS_DIR = DATA / "records"
INDEX_DIR = ROOT / ".index"
CACHE_DIR = ROOT / ".cache"

# Load project settings before evaluating any _env(...) declarations below.
# Previously llm.py loaded .env after config.py had already fixed all defaults,
# so toggles written in .env appeared to have no effect.
load_dotenv(ROOT / ".env")


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


# --------------------------------------------------------------------------- #
# models  (OpenRouter ids — see the model list in README §Budget)
# --------------------------------------------------------------------------- #
# TOOL_MODEL must support native function calling. Small "free" models usually
# do not; if your tool calls silently never fire, that is the first thing to check.
TOOL_MODEL = _env("TOOL_MODEL", "openai/gpt-4o-mini")
FAST_MODEL = _env("FAST_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = _env("TEMPERATURE", 0.0, float)
MAX_TOKENS = _env("MAX_TOKENS", 900, int)

# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
# "dense" = search by meaning only. "hybrid" uses the implemented TODO 2 BM25 + RRF path;
# "hybrid_rerank" additionally re-sorts the filtered candidates with an AI (optional).
RETRIEVAL_MODE = _env("RETRIEVAL_MODE", "hybrid")
ENABLE_QUERY_TRANSLATION = _env("ENABLE_QUERY_TRANSLATION", False, bool)
ENABLE_RERANK = _env("ENABLE_RERANK", False, bool)
TOP_K = _env("TOP_K", 6, int)              # measured structured recall improves at k=6
CANDIDATE_K = _env("CANDIDATE_K", 8, int)   # how many to fetch before narrowing down
CHUNK_SIZE = _env("CHUNK_SIZE", 1200, int)  # best measured structured setting on dev
CHUNK_OVERLAP = _env("CHUNK_OVERLAP", 150, int)
RRF_K = _env("RRF_K", 60, int)

# The two handbook sections that must never be the source of an answer.
# Search may still return them — the agent should be able to recognise a claim a
# customer read on the community page — but it must not answer from them.
UNTRUSTED_DOCS = {"community"}
SUPERSEDED_DOCS = {"archive_returns_2024"}

# --------------------------------------------------------------------------- #
# the company's rules (these numbers come from the handbook — keep them matching)
# --------------------------------------------------------------------------- #
AGENT_REFUND_LIMIT_INR = _env("AGENT_REFUND_LIMIT_INR", 5000, int)
GOODWILL_DELAY_CREDIT_INR = 500
HUMAN_HOURS_IST = "09:00-21:00"

# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #
MAX_TOOL_STEPS = _env("MAX_TOOL_STEPS", 6, int)   # stops the tool loop running forever
VERIFY_ENABLED = _env("VERIFY_ENABLED", True, bool)
LLM_CACHE = _env("LLM_CACHE", True, bool)         # reuse saved answers, so reruns are free
REQUEST_TIMEOUT = _env("REQUEST_TIMEOUT", 120, int)
MAX_RETRIES = _env("MAX_RETRIES", 4, int)

ROUTES = ("resolved", "needs_info", "escalated")
PRIORITIES = ("P1", "P2", "P3")
