"""Export the built Qdrant index to a portable parquet cache.

Point of this: `ingest/build_index.py` re-embeds via the NVIDIA API every
time it runs (~16 min for the current 30-rows/language corpus, real API
cost) — fine for the person who builds the index, painful for every other
clone. This dumps chunk_id/payload/both vectors to parquet so
`ingest/load_cached_embeddings.py` can repopulate a fresh Qdrant + BM25
index from disk in seconds, with zero embedding calls (neither the local
model nor NVIDIA).

Sharded because a single file for ~8K chunks at (384+2048)-dim float32
runs close to GitHub's per-file warning threshold — output is split into
several files under SHARD_ROWS each so the repo stays comfortable to clone
and diff. Written to ingest/embeddings_cache/, which .gitignore carves an
explicit exception for despite the broader `data/`/`*.parquet` ignores.

Usage:
    uv run python -m ingest.export_embeddings
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

from api.retrieval.qdrant_store import COLLECTION_NAME, VECTOR_NAME, VECTOR_NAME_NVIDIA, get_client

OUTPUT_DIR = Path("ingest/embeddings_cache")
SHARD_ROWS = 2000  # ~(384+2048)*4 bytes/row of raw vector data -> comfortably under 50MB/shard
SCROLL_BATCH = 500

PAYLOAD_FIELDS = [
    "chunk_id", "doc_id", "parent_id", "strategy", "text", "embed_text",
    "language", "query_type", "is_selected", "has_numbers",
]


def _iter_points():
    client = get_client()
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME, limit=SCROLL_BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        yield from points
        if offset is None:
            return


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    rows: list[dict] = []
    shard_idx = 0
    total = 0

    def flush():
        nonlocal shard_idx, rows
        if not rows:
            return
        df = pl.DataFrame(rows)
        path = OUTPUT_DIR / f"shard_{shard_idx:03d}.parquet"
        df.write_parquet(path, compression="zstd")
        print(f"[export] wrote {path} ({len(rows)} rows, {path.stat().st_size / 1e6:.1f} MB)")
        shard_idx += 1
        rows = []

    for point in _iter_points():
        payload = point.payload or {}
        row = {field: payload.get(field) for field in PAYLOAD_FIELDS}
        row["dense"] = point.vector[VECTOR_NAME]
        row["dense_nvidia"] = point.vector[VECTOR_NAME_NVIDIA]
        rows.append(row)
        total += 1
        if len(rows) >= SHARD_ROWS:
            flush()

    flush()
    print(f"[export] done — {total} chunks across {shard_idx} shard(s) in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
