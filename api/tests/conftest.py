"""Shared test fixtures. See docs/13-build-status.md.

The `tiny_index` fixture builds a small, real Qdrant + BM25 index from a
hand-written corpus, following the exact same code path as
ingest/build_index.py (real chunkers, real local embedding model), so
integration tests exercise real retrieval rather than mocks — without
depending on the full ~8000-chunk MSMARCO ingest or any network access.

Caching gotcha: `qdrant_store.get_client` (a lock-guarded lazy singleton,
not `@lru_cache` — see its docstring for why) and `sparse._index_path` /
`sparse._retriever` (`lru_cache`d) are module-level singletons bound at
import time. Mutating `settings` after import does not reach them —
instead we monkeypatch each accessor function directly. Because every
call site in qdrant_store.py/sparse.py looks up these names as module
globals at call time (not a pre-bound reference), replacing the module
attribute is sufficient; no need to touch `api.config.settings` at all.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
import regex
from qdrant_client import models

from api.config import settings
from api.retrieval import qdrant_store, sparse
from api.retrieval.chunkers.base import PassageDoc
from api.retrieval.chunkers.registry import chunk_with_all_strategies
from api.retrieval.embed import embed_passages
from api.retrieval.qdrant_store import VECTOR_NAME, VECTOR_NAME_NVIDIA

_HAS_NUMBER_RE = regex.compile(r"[0-9०-९০-৯]")

EIFFEL_EN = (
    "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
    "It was completed in 1889 and stands 330 metres tall."
)
EIFFEL_HI = (
    "एफिल टॉवर पेरिस, फ्रांस में स्थित एक लोहे की जालीदार मीनार है। "
    "यह 1889 में पूरी हुई थी और 330 मीटर ऊंची है।"
)
EVEREST_EN = "Mount Everest is the tallest mountain above sea level, with a peak elevation of 8849 metres."
EVEREST_BN = "মাউন্ট এভারেস্ট সমুদ্রপৃষ্ঠ থেকে সর্বোচ্চ পর্বত, এর চূড়ার উচ্চতা ৮৮৪৯ মিটার।"
PYTHON_EN = "Python is a high-level, general-purpose programming language known for its readable syntax."
GANGES_HI = "गंगा नदी भारत की सबसे पवित्र और सबसे लंबी नदी मानी जाती है।"
GREATWALL_EN = "The Great Wall of China is a series of fortifications built across the historical borders of China."


def _tiny_corpus() -> list[PassageDoc]:
    return [
        PassageDoc(
            doc_id="en/1/p0", text=EIFFEL_EN, language="en", script="Latin",
            query_id=1, query_type="description", is_selected=True,
            twin_doc_id="hi/1/p0", twin_text=EIFFEL_HI,
        ),
        PassageDoc(
            doc_id="hi/1/p0", text=EIFFEL_HI, language="hi", script="Devanagari",
            query_id=1, query_type="description", is_selected=True,
            twin_doc_id="en/1/p0", twin_text=EIFFEL_EN,
        ),
        PassageDoc(
            doc_id="en/2/p0", text=EVEREST_EN, language="en", script="Latin",
            query_id=2, query_type="numeric", is_selected=True,
            twin_doc_id="bn/2/p0", twin_text=EVEREST_BN,
            free_question="How tall is Mount Everest?",
        ),
        PassageDoc(
            doc_id="bn/2/p0", text=EVEREST_BN, language="bn", script="Bengali",
            query_id=2, query_type="numeric", is_selected=True,
            twin_doc_id="en/2/p0", twin_text=EVEREST_EN,
        ),
        PassageDoc(
            doc_id="en/3/p0", text=PYTHON_EN, language="en", script="Latin",
            query_id=3, query_type="description", is_selected=True,
            free_question="What is Python?",
        ),
        PassageDoc(
            doc_id="hi/3/p0", text=GANGES_HI, language="hi", script="Devanagari",
            query_id=4, query_type="description", is_selected=False,
        ),
        PassageDoc(
            doc_id="en/4/p0", text=GREATWALL_EN, language="en", script="Latin",
            query_id=5, query_type="description", is_selected=True,
        ),
    ]


def fake_nvidia_embed(text: str, dim: int | None = None) -> list[float]:
    """Deterministic toy embedding standing in for the real NVIDIA API in
    tests — hashed character-trigram bag, so semantically similar/identical
    text lands near itself under cosine similarity (real nearest-neighbor
    behavior for the dense_nvidia integration test, not random noise).
    """
    dim = dim or settings.nvidia_embed_dim
    vec = [0.0] * dim
    lowered = text.lower()
    trigrams = [lowered[i : i + 3] for i in range(max(len(lowered) - 2, 1))] or [lowered]
    for trigram in trigrams:
        idx = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def tiny_index(tmp_path, monkeypatch):
    qdrant_path = tmp_path / "qdrant"
    bm25_path = tmp_path / "bm25"
    bm25_path.mkdir()

    qdrant_store._reset_client_for_tests()
    from qdrant_client import QdrantClient

    fresh_client = QdrantClient(path=str(qdrant_path))
    monkeypatch.setattr(qdrant_store, "get_client", lambda: fresh_client)

    sparse._index_path.cache_clear()
    monkeypatch.setattr(sparse, "_index_path", lambda: bm25_path)
    sparse._retriever.cache_clear()

    docs = _tiny_corpus()
    chunks = [c for doc in docs for c in chunk_with_all_strategies(doc)]

    embed_texts = [c.embed_text for c in chunks]
    vectors_local = embed_passages(embed_texts)
    vectors_nvidia = [fake_nvidia_embed(t) for t in embed_texts]

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={VECTOR_NAME: v_local, VECTOR_NAME_NVIDIA: v_nvidia},
            payload={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "parent_id": chunk.parent_id,
                "strategy": chunk.strategy,
                "text": chunk.text,
                "language": chunk.language,
                "query_type": chunk.query_type,
                "is_selected": chunk.is_selected,
                "has_numbers": bool(_HAS_NUMBER_RE.search(chunk.text)),
            },
        )
        for chunk, v_local, v_nvidia in zip(chunks, vectors_local, vectors_nvidia)
    ]
    qdrant_store.upsert_chunks(points)

    sparse.build_index([c.chunk_id for c in chunks], embed_texts)
    sparse._retriever.cache_clear()

    yield

    # No cache_clear() here: at this point `get_client`/`_index_path` are
    # still the monkeypatched replacements (monkeypatch's own teardown, which
    # restores the originals, runs after this fixture finishes tearing
    # down) — calling .cache_clear() on a plain lambda would AttributeError.
    # The originals' caches were already cleared at fixture *start*, which is
    # what actually matters (defends against a stale singleton from a prior
    # unpatched call).
