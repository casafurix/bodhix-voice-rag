"""Round-trip test for the export -> load embeddings cache, so a fresh
clone can populate the index without any embedding API calls. See
ingest/export_embeddings.py and ingest/load_cached_embeddings.py.
"""

from __future__ import annotations

import math
import uuid

import pytest
from qdrant_client import QdrantClient, models

from api.retrieval import qdrant_store, sparse
from api.retrieval.qdrant_store import COLLECTION_NAME, VECTOR_NAME, VECTOR_NAME_NVIDIA
from ingest import export_embeddings, load_cached_embeddings

_ROW = {
    "chunk_id": "en/1/p0/s1_fixed/c0",
    "doc_id": "en/1/p0",
    "parent_id": "en/1/p0",
    "strategy": "s1_fixed",
    "text": "Paris is the capital of France.",
    "embed_text": "Paris is the capital of France.",
    "language": "en",
    "query_type": "description",
    "is_selected": True,
    "has_numbers": False,
}


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


@pytest.fixture
def seeded_source_index(tmp_path, monkeypatch):
    """A tiny, real Qdrant collection with both vector fields — the thing
    export_embeddings.py reads from.

    Both `qdrant_store.get_client` and `export_embeddings.get_client` need
    patching: export_embeddings.py did `from ...qdrant_store import
    get_client`, a direct-import binding separate from qdrant_store's own
    module attribute — patching one does not reach the other's call site.
    """
    qdrant_store._reset_client_for_tests()
    source_client = QdrantClient(path=str(tmp_path / "source_qdrant"))
    monkeypatch.setattr(qdrant_store, "get_client", lambda: source_client)
    monkeypatch.setattr(export_embeddings, "get_client", lambda: source_client)

    qdrant_store.ensure_collection()
    point_id = str(uuid.uuid4())
    source_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector={VECTOR_NAME: [0.1] * 384, VECTOR_NAME_NVIDIA: [0.2] * 2048},
                payload=dict(_ROW),
            )
        ],
    )
    yield


def test_export_then_load_reproduces_vectors_and_payload(seeded_source_index, tmp_path, monkeypatch):
    cache_dir = tmp_path / "embeddings_cache"
    monkeypatch.setattr(export_embeddings, "OUTPUT_DIR", cache_dir)
    monkeypatch.setattr(export_embeddings, "SHARD_ROWS", 1)  # force sharding logic to run

    export_embeddings.main()
    assert list(cache_dir.glob("shard_*.parquet"))

    # Point the "target" side at a brand new, empty Qdrant + BM25 location —
    # simulating a fresh clone with no prior index at all. Same direct-import
    # gotcha as above: load_cached_embeddings.get_client is its own binding.
    qdrant_store._reset_client_for_tests()
    target_client = QdrantClient(path=str(tmp_path / "target_qdrant"))
    monkeypatch.setattr(qdrant_store, "get_client", lambda: target_client)
    monkeypatch.setattr(load_cached_embeddings, "get_client", lambda: target_client)

    bm25_path = tmp_path / "target_bm25"
    bm25_path.mkdir()
    sparse._index_path.cache_clear()
    monkeypatch.setattr(sparse, "_index_path", lambda: bm25_path)
    sparse._retriever.cache_clear()
    monkeypatch.setattr(load_cached_embeddings, "OUTPUT_DIR", cache_dir)

    load_cached_embeddings.main()

    points, _ = target_client.scroll(COLLECTION_NAME, limit=10, with_payload=True, with_vectors=True)
    assert len(points) == 1
    loaded = points[0]
    assert loaded.payload["chunk_id"] == _ROW["chunk_id"]
    assert loaded.payload["embed_text"] == _ROW["embed_text"]
    # Qdrant's cosine-distance collections store L2-normalized vectors, so
    # the round-tripped vector matches direction, not raw magnitude — the
    # same reason a self-similarity search returns exactly 1.0 regardless
    # of the original vector's scale.
    assert loaded.vector[VECTOR_NAME] == pytest.approx(_normalize([0.1] * 384))
    assert loaded.vector[VECTOR_NAME_NVIDIA] == pytest.approx(_normalize([0.2] * 2048))

    hits = sparse.search_sparse("Paris capital France", top_k=5)
    assert hits and hits[0][0] == _ROW["chunk_id"]

    sparse._retriever.cache_clear()
