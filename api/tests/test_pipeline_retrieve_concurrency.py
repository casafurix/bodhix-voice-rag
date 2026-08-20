"""Regression test for the retrieve-stage concurrency fix documented in
docs/13-build-status.md: the 6 dense-strategy searches + 1 sparse search
must run concurrently (asyncio.to_thread + gather), not sequentially — cost
should approach max() of the 7 arms, not their sum(). Encodes the fix so a
future change can't silently revert to a sequential `for` loop without a
test failing.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from api.harness import pipeline
from api.harness.context import Context
from api.harness.deadline import Deadline
from api.retrieval.strategies import STRATEGY_IDS

SLEEP_S = 0.05

# All 7 arms (6 dense strategies + sparse) return the SAME ranked chunk set,
# so RRF fusion produces a clean, discriminative score distribution and the
# request actually reaches ANSWERED — a flat/single-candidate distribution
# would otherwise trip the coverage gate's LOW_CONFIDENCE check regardless
# of the concurrency fix under test here.
FAKE_HITS = [
    ("chunk-1", 0.95, "Paris is the capital city of France and a major European hub."),
    ("chunk-2", 0.90, "Berlin is the capital of Germany."),
    ("chunk-3", 0.85, "Madrid is the capital of Spain."),
    ("chunk-4", 0.80, "Rome is the capital of Italy."),
    ("chunk-5", 0.75, "Lisbon is the capital of Portugal."),
]


def fake_search_dense(strategy_id, query_vector, top_k=50, query_filter=None, vector_name="dense"):
    time.sleep(SLEEP_S)
    return [
        SimpleNamespace(
            id=cid,
            score=score,
            payload={
                "chunk_id": cid, "parent_id": cid, "strategy": "s1_fixed",
                "text": text, "language": "en",
            },
        )
        for cid, score, text in FAKE_HITS
    ]


def fake_search_sparse(query, top_k=50):
    time.sleep(SLEEP_S)
    return [(cid, score) for cid, score, _ in FAKE_HITS]


@pytest.mark.asyncio
async def test_retrieve_stage_runs_the_seven_arms_concurrently(monkeypatch):
    monkeypatch.setattr(pipeline, "search_dense", fake_search_dense)
    monkeypatch.setattr(pipeline, "search_sparse", fake_search_sparse)

    deadline = Deadline(budget_ms=5000)
    ctx = Context(deadline=deadline)

    response = await pipeline.run_retrieval_and_answer(
        "Where is Paris?", "en", {}, ctx, deadline,
        embedding_provider="local", answer_mode="extractive",
    )

    assert response.verdict == "ANSWERED"

    n_arms = len(STRATEGY_IDS) + 1  # 6 dense strategies + 1 sparse
    sequential_cost_ms = n_arms * SLEEP_S * 1000
    retrieve_ms = ctx.timings_ms["retrieve"]

    # Concurrent: close to one arm's cost. Sequential (the bug): close to
    # n_arms * one arm's cost. A generous threshold well below the
    # sequential cost still clearly distinguishes the two.
    assert retrieve_ms < sequential_cost_ms * 0.5, (
        f"retrieve stage took {retrieve_ms:.1f}ms, expected well under "
        f"{sequential_cost_ms:.1f}ms (sequential) if arms ran concurrently"
    )
