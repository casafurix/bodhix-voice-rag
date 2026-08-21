"""Optional config the eval loop reads defensively (see the eval loop's
own TARGET_INTERFACE.md). Nothing here is required — a missing name just
falls back to the suite's own default.
"""

# Mirrors bench/run_retrieval_latency.py's RETRIEVAL_SUB_BUDGET_MS. The
# eval loop's embed() call is real (ours), but the ANN search it times
# runs against its own throwaway FAISS index, not our production
# Qdrant+BM25 hybrid search — this is the honest number to compare that
# slice against, not our full t_core budget (200ms, includes guardrails
# and answer generation too).
LATENCY_BUDGET_MS = 50

GENERATION_MODEL = "bodhix-extractive-lexical-overlap"
