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


def search_sparse(query: str, top_k: int = 50) -> list[tuple[str, float]]:
    query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
    results, scores = _retriever().retrieve(query_tokens, k=top_k)
    return list(zip(results[0].tolist(), scores[0].tolist()))
