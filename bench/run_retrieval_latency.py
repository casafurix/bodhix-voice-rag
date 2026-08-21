"""Measure embed + hybrid-retrieve latency in isolation, against a
sub-budget of the 200ms t_core target. See docs/08-latency.md.

This is deliberately narrower than the full /ask benchmark
(bench/run_latency.py, added once ingest lands): it isolates just
`embed` + `retrieve_dense` (all strategies) + `retrieve_sparse` + `fuse`,
skipping guard_in/coverage_gate/assemble/answer_fast/guard_out. Per
docs/08-latency.md's own per-stage budget table, that slice's nominal
cost is ~18ms and its P95 estimate is ~33ms — RETRIEVAL_SUB_BUDGET_MS
below (50ms) leaves headroom above that estimate without being anywhere
close to the full 200ms t_core budget, so a regression here is caught
before it ever reaches the full-pipeline benchmark.

Adapted from a reference benchmark.py another participant shared (FAISS +
`app.retriever`, 50ms budget) — same shape (warmup, percentile table,
pass/fail against a budget), rewired to our actual stack: fastembed +
Qdrant (embedded mode) + bm25s + RRF, under api/retrieval/.

Requires an index to already exist (ingest/build_index.py) — will raise a
clear error, not a bare stack trace, if run before ingest.

Usage:
    uv run python -m bench.run_retrieval_latency [n_queries]
"""

from __future__ import annotations

import statistics
import sys
import time

from pydantic import BaseModel

from api.retrieval.embed import embed_query
from api.retrieval.fuse import ScoredChunk, reciprocal_rank_fusion
from api.retrieval.qdrant_store import search_dense_grouped
from api.retrieval.sparse import search_sparse

RETRIEVAL_SUB_BUDGET_MS = 50.0  # embed + retrieve + fuse only — see module docstring

QUERIES = [
    "what is the capital of India",
    "how does the human heart pump blood",
    "who wrote the Indian constitution",
    "what causes rainfall",
    "how many states are in India",
    "what is the boiling point of water",
    "when was the taj mahal built",
    "how do vaccines work",
]


class SearchTiming(BaseModel):
    total_ms: float
    embed_ms: float
    search_ms: float  # dense (all strategies) + sparse + fuse, combined


def warmup() -> None:
    """Loads the embedding model and, if the index exists, runs the ANN
    path once so first-inference cost isn't counted in the timed loop.
    """
    embed_query("warmup query")


def search(query: str, top_k: int = 50) -> SearchTiming:
    t0 = time.perf_counter()
    query_vector = embed_query(query)
    t1 = time.perf_counter()

    # Same production shape as api/harness/pipeline.py's retrieve stage:
    # ONE grouped dense search (top-50 per strategy, client-side bucketing)
    # + the BM25 arm, then RRF. The per-strategy filtered-search loop this
    # replaces measured ~600ms avg / ~611ms P95 against qdrant local mode —
    # the payload filter forces a brute-force scan per query.
    try:
        dense_results = search_dense_grouped(query_vector, per_arm_k=top_k)
        sparse_hits = search_sparse(query, top_k=top_k)
    except ValueError as exc:
        # Qdrant's embedded/local client raises a plain ValueError for a
        # missing collection (its remote/HTTP client would raise
        # UnexpectedResponse instead — we're on the local client here).
        print(f"FAIL: {exc}")
        print(
            "No index found. Run ingest/build_index.py first — this "
            "benchmark measures retrieval against a real index, not a mock."
        )
        sys.exit(1)

    ranked_lists: list[list[ScoredChunk]] = []
    for hits in dense_results:
        ranked_lists.append(
            [
                ScoredChunk(chunk_id=(h.payload or {}).get("chunk_id", str(h.id)), score=h.score)
                for h in hits
            ]
        )
    ranked_lists.append([ScoredChunk(chunk_id=cid, score=s) for cid, s in sparse_hits])

    reciprocal_rank_fusion(ranked_lists)
    t2 = time.perf_counter()

    return SearchTiming(
        total_ms=(t2 - t0) * 1000.0,
        embed_ms=(t1 - t0) * 1000.0,
        search_ms=(t2 - t1) * 1000.0,
    )


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Warming up (model load + first inference)...")
    try:
        warmup()
    except Exception as exc:
        print(f"FAIL: warmup failed — {exc}")
        sys.exit(1)

    total_ms, embed_ms, search_ms = [], [], []
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        try:
            timing = search(query, top_k=50)
        except ValueError as exc:
            # Qdrant's embedded/local client raises a plain ValueError for a
            # missing collection (its remote/HTTP client would raise
            # UnexpectedResponse instead — we're on the local client here).
            print(f"FAIL: {exc}")
            print(
                "No index found. Run ingest/build_index.py first — this "
                "benchmark measures retrieval against a real index, not a mock."
            )
            sys.exit(1)
        total_ms.append(timing.total_ms)
        embed_ms.append(timing.embed_ms)
        search_ms.append(timing.search_ms)

    print(f"\nRan {n} queries\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [("embed", embed_ms), ("search", search_ms), ("total", total_ms)]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nSub-budget: {RETRIEVAL_SUB_BUDGET_MS}ms (embed+retrieve only, not full t_core) "
          f"| p95 total: {p95_total:.2f}ms")
    if p95_total <= RETRIEVAL_SUB_BUDGET_MS:
        print("PASS: within sub-budget")
    else:
        print("FAIL: over sub-budget — see docs/08-latency.md degradation ladder")
        sys.exit(1)


if __name__ == "__main__":
    main()
