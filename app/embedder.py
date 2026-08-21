"""Real embedder for rag-local-eval-loop (organizer-provided grading
harness — see wiring-in-the-eval-loop.pdf). This is NOT a separate
eval-only model: it's the exact local embedding path POST /ask uses for
every text query (api/retrieval/embed.py, paraphrase-multilingual-
MiniLM-L12-v2). The eval loop imports this module directly and calls
these three functions against its own throwaway index — see
TARGET_INTERFACE.md in the eval loop's own repo for the required shape.
"""

from __future__ import annotations

import numpy as np

from api.retrieval.embed import embed_passages, embed_query
from api.retrieval.embed import _model as _load_model  # noqa: F401
# Reaching past embed.py's own lru_cache wrapper is intentional: the only
# required effect of get_model() (per TARGET_INTERFACE.md) is "load the
# model once" -- _model() IS that loader. There's no public alias because
# production code never needed one (embed_query/embed_passages already
# call it internally).


def get_model():
    return _load_model()


def embed(texts: list[str]) -> np.ndarray:
    return np.array(embed_passages(texts))


def embed_one(text: str) -> np.ndarray:
    return np.array(embed_query(text))
