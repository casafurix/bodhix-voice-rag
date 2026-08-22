"""Query/passage embedding. See docs/04-retrieval.md.

MODEL NOTE: the doc's first choice is `intfloat/multilingual-e5-small`
(384-dim, ONNX int8). The installed fastembed (0.8.0) only bundles
`multilingual-e5-large` (1024-dim, slower, bigger index), not `-small`. We
use the doc's own documented fallback instead —
`paraphrase-multilingual-MiniLM-L12-v2` (384-dim, still covers en/hi/bn,
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
    # threads=1 + enable_cpu_mem_arena=False: found deploying to Render's
    # free tier (512MB hard container cap) -- this model is lazy-loaded on
    # the FIRST real query (never touched during startup, which is why
    # ingest/load_cached_embeddings.py's own cold start looked fine), and
    # loading its 224MB int8-quantized ONNX file via onnxruntime's default
    # session options (memory arena pre-allocates larger reusable blocks,
    # each thread gets its own scratch buffers) pushed the container over
    # its cap mid-request -- confirmed via a live 502/exit-137 on the first
    # /ask call, not theoretical. Both options are onnxruntime's own
    # documented levers for memory-constrained deployments; see
    # docs/13-build-status.md.
    return TextEmbedding(model_name=MODEL_NAME, threads=1, enable_cpu_mem_arena=False)


def embed_query(text: str) -> list[float]:
    (vec,) = _model().embed([text])
    return vec.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in _model().embed(texts)]
