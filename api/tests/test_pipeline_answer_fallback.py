"""Regression tests for the abstractive -> extractive degradation ladder in
run_retrieval_and_answer. Confirmed live during this build: the NVIDIA LLM
can fail outright (network/timeout) OR succeed but produce an answer
ungrounded in the retrieved context (a reasoning-model drift observed in
practice, not a hypothetical) — both must degrade to the extractive
fallback rather than fail the whole request. See api/harness/pipeline.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from api.answer.abstractive import AbstractiveAnswer
from api.harness import pipeline
from api.harness.context import Context
from api.harness.deadline import Deadline
from api.llm.nvidia_client import NvidiaCallError

FAKE_HITS = [
    ("chunk-1", 0.95, "Paris is the capital city of France and a major European hub."),
    ("chunk-2", 0.90, "Berlin is the capital of Germany."),
    ("chunk-3", 0.85, "Madrid is the capital of Spain."),
    ("chunk-4", 0.80, "Rome is the capital of Italy."),
    ("chunk-5", 0.75, "Lisbon is the capital of Portugal."),
]


def fake_search_dense(strategy_id, query_vector, top_k=50, query_filter=None, vector_name="dense"):
    return [
        SimpleNamespace(
            id=cid, score=score,
            payload={"chunk_id": cid, "parent_id": cid, "strategy": "s1_fixed", "text": text, "language": "en"},
        )
        for cid, score, text in FAKE_HITS
    ]


def fake_search_sparse(query, top_k=50):
    return [(cid, score) for cid, score, _ in FAKE_HITS]


@pytest.fixture(autouse=True)
def _patch_retrieval(monkeypatch):
    monkeypatch.setattr(pipeline, "search_dense", fake_search_dense)
    monkeypatch.setattr(pipeline, "search_sparse", fake_search_sparse)


@pytest.mark.asyncio
async def test_nvidia_call_error_falls_back_to_extractive(monkeypatch):
    async def failing_abstractive(query, blocks, deadline):
        raise NvidiaCallError("simulated outage")

    monkeypatch.setattr(pipeline, "generate_abstractive_answer", failing_abstractive)

    deadline = Deadline(budget_ms=5000)
    ctx = Context(deadline=deadline)
    response = await pipeline.run_retrieval_and_answer(
        "Where is Paris?", "en", {}, ctx, deadline,
        embedding_provider="local", answer_mode="abstractive",
    )

    assert response.verdict == "ANSWERED"
    assert response.answer.mode == "extractive"
    assert "abstractive_failed_fallback_extractive" in response.degradations


@pytest.mark.asyncio
async def test_ungrounded_abstractive_answer_falls_back_to_extractive(monkeypatch):
    async def ungrounded_abstractive(query, blocks, deadline):
        # Real content, real citation — but shares no vocabulary with the
        # retrieved context, exactly like the live drift this test guards.
        return AbstractiveAnswer(
            text="I am unable to determine this from unrelated topics entirely.",
            cited_chunk_ids=[blocks[0].chunk_id],
        )

    monkeypatch.setattr(pipeline, "generate_abstractive_answer", ungrounded_abstractive)

    deadline = Deadline(budget_ms=5000)
    ctx = Context(deadline=deadline)
    response = await pipeline.run_retrieval_and_answer(
        "Where is Paris?", "en", {}, ctx, deadline,
        embedding_provider="local", answer_mode="abstractive",
    )

    assert response.verdict == "ANSWERED"
    assert response.answer.mode == "extractive"
    assert response.answer.language == "en"
    assert "abstractive_ungrounded_fallback_extractive" in response.degradations


@pytest.mark.asyncio
async def test_grounded_abstractive_answer_is_kept(monkeypatch):
    async def grounded_abstractive(query, blocks, deadline):
        return AbstractiveAnswer(
            text="Paris is the capital city of France, a major hub in Europe.",
            cited_chunk_ids=[blocks[0].chunk_id],
        )

    monkeypatch.setattr(pipeline, "generate_abstractive_answer", grounded_abstractive)

    deadline = Deadline(budget_ms=5000)
    ctx = Context(deadline=deadline)
    response = await pipeline.run_retrieval_and_answer(
        "Where is Paris?", "en", {}, ctx, deadline,
        embedding_provider="local", answer_mode="abstractive",
    )

    assert response.verdict == "ANSWERED"
    assert response.answer.mode == "abstractive"
    assert response.degradations == []
