"""Regression test for the memory-constrained deployment path
(settings.embedding_provider="nvidia" + coverage_local_reembed=False,
see api/config.py and docs/13-build-status.md).

The whole point of this mode is that the local MiniLM ONNX model (224MB,
confirmed live to OOM Render's free tier on first load) is NEVER loaded.
This test proves that at the code level: api.retrieval.embed.embed_query
is monkeypatched to raise if called at all, so any regression that
reintroduces a local-model call in this mode fails loudly here instead of
silently reappearing as a production OOM.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from api.harness import pipeline
from api.harness.context import Context
from api.harness.deadline import Deadline
from api.schemas import AskOptions, AskRequest

# Scores on the nvidia-embedding-space scale (see api/config.py's
# nvidia_coverage_tau_absolute/mean, calibrated coarser than the MiniLM
# path) -- comfortably above both thresholds so the request answers rather
# than refuses, which is what exercises the answer/guard_out stages too.
FAKE_HITS = [
    ("chunk-1", 0.40, "Paris is the capital city of France and a major European hub."),
    ("chunk-2", 0.38, "Berlin is the capital of Germany."),
    ("chunk-3", 0.36, "Madrid is the capital of Spain."),
    ("chunk-4", 0.34, "Rome is the capital of Italy."),
    ("chunk-5", 0.32, "Lisbon is the capital of Portugal."),
    ("chunk-6", 0.30, "Vienna is the capital of Austria."),
    ("chunk-7", 0.29, "Prague is the capital of Czechia."),
    ("chunk-8", 0.28, "Warsaw is the capital of Poland."),
    ("chunk-9", 0.27, "Budapest is the capital of Hungary."),
    ("chunk-10", 0.26, "Athens is the capital of Greece."),
]


def fake_search_dense_grouped(query_vector, per_arm_k=50, vector_name="dense_nvidia"):
    hits = [
        SimpleNamespace(
            id=cid, score=score,
            payload={"chunk_id": cid, "parent_id": cid, "strategy": "s1_fixed", "text": text, "language": "en"},
        )
        for cid, score, text in FAKE_HITS
    ]
    return [hits], [score for _, score, _ in FAKE_HITS]


def fake_search_sparse(query, top_k=50):
    return [(cid, score) for cid, score, _ in FAKE_HITS]


async def fake_aembed_query(text, deadline, input_type=None):
    return [0.1] * 2048


def never_call_local_embed(*args, **kwargs):
    raise AssertionError(
        "embed_query (the local 224MB ONNX model) was called in "
        "memory-constrained deployment mode -- this defeats the whole "
        "point of settings.coverage_local_reembed=False. See "
        "docs/13-build-status.md's Render OOM section."
    )


@pytest.fixture(autouse=True)
def _memory_constrained_mode(monkeypatch):
    monkeypatch.setattr(pipeline.settings, "embedding_provider", "nvidia")
    monkeypatch.setattr(pipeline.settings, "coverage_local_reembed", False)
    monkeypatch.setattr(pipeline, "aembed_query", fake_aembed_query)
    monkeypatch.setattr(pipeline, "search_dense_grouped", fake_search_dense_grouped)
    monkeypatch.setattr(pipeline, "search_sparse", fake_search_sparse)
    monkeypatch.setattr(pipeline, "embed_query", never_call_local_embed)


@pytest.mark.asyncio
async def test_text_ask_never_loads_local_model_in_nvidia_mode():
    request = AskRequest(query="Where is Paris?", budget_ms=5000, options=AskOptions())
    response = await pipeline.run_ask(request)

    assert response.verdict == "ANSWERED"
    assert response.answer.mode == "extractive"


@pytest.mark.asyncio
async def test_coverage_gate_uses_nvidia_thresholds_not_local_reembed():
    deadline = Deadline(budget_ms=5000)
    ctx = Context(deadline=deadline)
    response = await pipeline.run_retrieval_and_answer(
        "Where is Paris?", "en", {}, ctx, deadline,
        embedding_provider="nvidia", answer_mode="extractive",
    )
    assert response.verdict == "ANSWERED"
