"""Populate Qdrant + BM25 from the parquet cache in ingest/embeddings_cache/
— no embedding calls at all (neither the local model nor the NVIDIA API).
This is the fast path: a fresh clone runs this and gets a working index in
seconds instead of running ingest/build_index.py's full ~16-minute
re-embed. Re-run ingest/build_index.py (+ ingest/export_embeddings.py to
refresh the cache) only when the corpus itself changes.

Usage:
    uv run python -m ingest.load_cached_embeddings
"""

from __future__ import annotations

import time
import uuid

import polars as pl
from qdrant_client import models

from api.retrieval.qdrant_store import (
    COLLECTION_NAME,
    VECTOR_NAME,
    VECTOR_NAME_NVIDIA,
    ensure_collection,
    get_client,
)
from api.retrieval.sparse import build_index as build_sparse_index
from ingest.export_embeddings import OUTPUT_DIR

UPSERT_BATCH = 256

# Same fixed namespace as ingest/build_index.py — point ids stay stable
# across a build -> export -> load round trip.
_POINT_ID_NAMESPACE = uuid.UUID("6f6e8f2a-4b1a-4b8a-9a6e-2a0b6e6b2c11")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def main() -> None:
    shards = sorted(OUTPUT_DIR.glob("shard_*.parquet"))
    if not shards:
        raise SystemExit(
            f"No parquet shards found in {OUTPUT_DIR}/ — run "
            "`uv run python -m ingest.export_embeddings` against a built index first, "
            "or run `uv run python -m ingest.build_index` for a full (re-embedding) build."
        )

    t0 = time.perf_counter()
    ensure_collection()
    client = get_client()

    chunk_ids: list[str] = []
    embed_texts: list[str] = []
    total = 0

    for shard_path in shards:
        df = pl.read_parquet(shard_path)
        for start in range(0, len(df), UPSERT_BATCH):
            batch = df[start : start + UPSERT_BATCH]
            points = [
                models.PointStruct(
                    id=_point_id(row["chunk_id"]),
                    vector={VECTOR_NAME: row["dense"], VECTOR_NAME_NVIDIA: row["dense_nvidia"]},
                    payload={
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "parent_id": row["parent_id"],
                        "strategy": row["strategy"],
                        "text": row["text"],
                        "embed_text": row["embed_text"],
                        "language": row["language"],
                        "query_type": row["query_type"],
                        "is_selected": row["is_selected"],
                        "has_numbers": row["has_numbers"],
                    },
                )
                for row in batch.iter_rows(named=True)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            chunk_ids.extend(p["chunk_id"] for p in batch.select("chunk_id").iter_rows(named=True))
            embed_texts.extend(p["embed_text"] for p in batch.select("embed_text").iter_rows(named=True))
            total += len(batch)
        print(f"[load] {shard_path.name}: {total} chunks upserted so far")

    build_sparse_index(chunk_ids, embed_texts)
    print(f"[load] BM25 sparse index rebuilt over {len(chunk_ids)} chunks")

    elapsed = time.perf_counter() - t0
    print(f"[load] done in {elapsed:.1f}s — {total} chunks, 0 embedding API calls")


if __name__ == "__main__":
    main()
