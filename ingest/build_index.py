"""Orchestrates the full offline ingest: stream -> explode/dedupe -> filter
-> chunk (all 6 strategies) -> embed -> upsert to Qdrant -> build the BM25
sparse index. See docs/02-dataset.md, "Ingest pipeline".

Scope vs. the full doc: no tiered T0/T1/T2/T3 snapshotting, no metadata
enrichment beyond what explode_dedupe.py already attaches, no
manifest.json artefact versioning yet. ROWS_PER_LANGUAGE below is
intentionally small for a first correctness-focused run — the goal right
now is proving the pipeline produces real, queryable answers end to end,
not hitting a target corpus size. Scaling up is a one-line constant change
once correctness is confirmed.

Usage:
    uv run python -m ingest.build_index [rows_per_language]
"""

from __future__ import annotations

import sys
import time
import uuid

import regex
from qdrant_client import models

from api.retrieval.chunkers.base import Chunk, PassageDoc
from api.retrieval.chunkers.registry import chunk_with_all_strategies
from api.retrieval.embed import embed_passages
from api.retrieval.qdrant_store import COLLECTION_NAME, VECTOR_NAME, ensure_collection, get_client
from api.retrieval.sparse import build_index as build_sparse_index
from ingest.explode_dedupe import DedupIndex, explode_rows
from ingest.filters import FilterCounts, passes_filters
from ingest.stream_corpus import LANGUAGE_FILES, load_language_rows

ROWS_PER_LANGUAGE_DEFAULT = 200
EMBED_BATCH_SIZE = 64

# Fixed namespace so re-running ingest on the same chunk_id always produces
# the same Qdrant point id — upserts are idempotent, not append-only.
_POINT_ID_NAMESPACE = uuid.UUID("6f6e8f2a-4b1a-4b8a-9a6e-2a0b6e6b2c11")

_HAS_NUMBER_RE = regex.compile(r"[0-9०-९০-৯]")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def _load_and_prepare(rows_per_language: int) -> list[PassageDoc]:
    dedup = DedupIndex()
    all_docs: list[PassageDoc] = []
    filter_counts: dict[str, FilterCounts] = {}

    for lang in LANGUAGE_FILES:
        print(f"[ingest] downloading + loading {rows_per_language} rows for '{lang}'...")
        rows = load_language_rows(lang, limit_rows=rows_per_language)
        docs = explode_rows(rows, lang, dedup)
        print(f"[ingest] '{lang}' file exploded to {len(docs)} deduped passage docs "
              f"(includes derived English docs)")

        for doc in docs:
            counts = filter_counts.setdefault(doc.language, FilterCounts())
            if passes_filters(doc.text, doc.language, counts):
                all_docs.append(doc)

    for lang, counts in filter_counts.items():
        print(f"[ingest] filters[{lang}]: {counts.summary()}")

    return all_docs


def _chunk_all(docs: list[PassageDoc]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_with_all_strategies(doc))
    return chunks


def _embed_and_upsert(chunks: list[Chunk]) -> None:
    ensure_collection()
    client = get_client()

    total = len(chunks)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        vectors = embed_passages([c.embed_text for c in batch])

        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={VECTOR_NAME: vector},
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
            for chunk, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[ingest] embedded + upserted {min(start + EMBED_BATCH_SIZE, total)}/{total} chunks")


def _build_sparse(chunks: list[Chunk]) -> None:
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.embed_text for c in chunks]
    build_sparse_index(chunk_ids, texts)
    print(f"[ingest] BM25 sparse index built over {len(chunks)} chunks")


def main() -> None:
    rows_per_language = (
        int(sys.argv[1]) if len(sys.argv) > 1 else ROWS_PER_LANGUAGE_DEFAULT
    )
    t0 = time.perf_counter()

    docs = _load_and_prepare(rows_per_language)
    print(f"[ingest] {len(docs)} passage docs survived filtering "
          f"(languages: {sorted({d.language for d in docs})})")

    chunks = _chunk_all(docs)
    print(f"[ingest] {len(chunks)} chunks produced across "
          f"{sorted({c.strategy for c in chunks})}")

    _embed_and_upsert(chunks)
    _build_sparse(chunks)

    elapsed = time.perf_counter() - t0
    print(f"[ingest] done in {elapsed:.1f}s — {len(docs)} docs, {len(chunks)} chunks")


if __name__ == "__main__":
    main()
