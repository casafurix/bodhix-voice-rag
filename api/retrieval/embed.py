"""Query/passage embedding. See docs/04-retrieval.md.

MODEL NOTE: the doc's first choice is `intfloat/multilingual-e5-small`
(384-dim, ONNX int8). The installed fastembed (0.8.0) only bundles
`multilingual-e5-large` (1024-dim, slower, bigger index), not `-small`. We
use the doc's own documented fallback instead —
`paraphrase-multilingual-MiniLM-L12-v2` (384-dim, still covers en/hi/ta,
directly supported by fastembed with zero extra engineering).
TODO: revisit — export multilingual-e5-small to ONNX ourselves (optimum)
if retrieval quality on the ablation demands it. See docs/04-retrieval.md,
embedding options table.

e5-family models require asymmetric prefixes ("query: " / "passage: ").
MiniLM does not, but we keep the two-function shape so swapping the model
back to e5-small later is a one-line change, not a call-site rewrite.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)


def embed_query(text: str) -> list[float]:
    (vec,) = _model().embed([text])
    return vec.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in _model().embed(texts)]
