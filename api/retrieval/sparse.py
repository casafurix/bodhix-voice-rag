"""BM25 sparse index — bm25s, memory-mapped. See docs/04-retrieval.md.

Indic tokenisation note (docs/04-retrieval.md): whitespace tokenisation is
wrong for morphologically rich languages. MVP ships whitespace + basic
normalisation for all three languages and flags this as a known gap rather
than pretending it is solved — a real indic-nlp-library stemmer is a
documented follow-up, not silently skipped.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import bm25s

from api.config import settings


def tokenize(text: str) -> list[str]:
    # TODO: swap for per-script tokenisation (indic-nlp-library) — see
    # docs/04-retrieval.md. Whitespace-only is the known-honest gap for now.
    return bm25s.tokenize(text, stopwords=None, show_progress=False)


@lru_cache
def _index_path() -> Path:
    path = Path(settings.bm25_index_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_index(chunk_ids: list[str], texts: list[str]) -> None:
    corpus_tokens = bm25s.tokenize(texts, stopwords=None, show_progress=False)
    retriever = bm25s.BM25(corpus=chunk_ids)
    retriever.index(corpus_tokens)
    retriever.save(str(_index_path()))


@lru_cache
def _retriever() -> bm25s.BM25:
    return bm25s.BM25.load(str(_index_path()), load_corpus=True)


def _corpus_item_to_chunk_id(item) -> str:
    # bm25s round-trips a plain-string corpus through disk as JSONL records
    # shaped {"id": int, "text": str} — `.load(load_corpus=True)` returns
    # those records, not the original strings we passed to build_index().
    # Our chunk_id lives in the "text" field in that case.
    if isinstance(item, dict):
        return item.get("text", str(item))
    return str(item)


def search_sparse(query: str, top_k: int = 50) -> list[tuple[str, float]]:
    query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
    results, scores = _retriever().retrieve(query_tokens, k=top_k)
    chunk_ids = [_corpus_item_to_chunk_id(item) for item in results[0].tolist()]
    return list(zip(chunk_ids, scores[0].tolist()))
