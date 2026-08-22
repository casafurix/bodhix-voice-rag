"""Qdrant in embedded/local mode — no server process, no network hop.
See docs/04-retrieval.md and the architecture note on avoiding microservices.

DESIGN NOTE — deviation from docs/04-retrieval.md's "named vectors per
point" idea, and why: that design works cleanly when the *same* logical
unit gets several vector representations (e.g. late_chunking adding a
document-conditioned vector to an existing chunk). It does not work for
comparing S1/S2/S3/S5/S9/S10 against each other, because each strategy
produces genuinely different chunk boundaries and counts over the same
source passages — a sentence-level S3 chunk and a passage-level S2 chunk
are not "the same point with two vectors", they're different points
entirely. Qdrant also requires every point to supply every declared named
vector unless configured otherwise, which a per-strategy chunk set can't
satisfy anyway.

What we do instead: **one collection, one vector field ("dense"), and a
`strategy` payload field used as a filter.** Retrieving "strategy X's
candidates" is a filtered ANN search rather than a differently-named
vector search. This is functionally equivalent for both the per-strategy
ablation (docs/03-chunking.md) and the S12-style ensemble (search each
strategy's filtered subset, then RRF-fuse the results, same as we already
fuse dense+sparse) — it is simpler to implement correctly under time
pressure, at the cost of N separate filtered searches instead of one
multi-vector search. At T0/T1 scale that cost is negligible.
"""

from __future__ import annotations

import threading

from qdrant_client import QdrantClient, models

from api.config import settings

COLLECTION_NAME = "chunks"
VECTOR_NAME = "dense"
VECTOR_DIM = 384  # matches api/retrieval/embed.py's model output dim

# Second named vector on the SAME points, populated by ingest/build_index.py by
# re-embedding the same corpus with the NVIDIA API — this is the case
# named-vectors-per-point is actually for (see module docstring above): the
# same logical chunk, a second vector representation. Text queries search
# "dense" (local, fast, free); voice queries search "dense_nvidia" (online),
# since a voice query is embedded via the NVIDIA API and a cross-space
# comparison against "dense" would be meaningless. See docs/13-build-status.md.
VECTOR_NAME_NVIDIA = "dense_nvidia"


_client: QdrantClient | None = None
_client_lock = threading.Lock()


def get_client() -> QdrantClient:
    """Lazy singleton, guarded by a real lock — not `@lru_cache`.

    The retrieve stage now dispatches 6 dense-strategy searches concurrently
    via `asyncio.to_thread` (api/harness/pipeline.py), so on the first
    request after startup, all 6 worker threads call this near-simultaneously.
    `lru_cache` does NOT serialize concurrent calls on a cache miss — it can
    let multiple threads construct independent `QdrantClient` instances in
    parallel, and Qdrant's embedded/local mode holds an exclusive file lock
    on its storage path, so the second constructor call fails with
    "already accessed by another instance". Double-checked locking here
    guarantees exactly one `QdrantClient` is ever constructed.

    Deploy note: `QDRANT_LOCAL_PATH=:memory:` runs Qdrant purely in RAM (no
    disk, no file lock) — the fit for an ephemeral/serverless single
    instance that repopulates from ingest/embeddings_cache/ on cold start
    (seconds, no embedding calls — see ingest/load_cached_embeddings.py)
    rather than mounting a persistent volume. Either mode is still a
    single-process embedded store, not a server — it does not support
    multiple concurrent instances/replicas sharing one index; that needs a
    real Qdrant server (self-hosted or Cloud), not this module.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            if settings.qdrant_local_path == ":memory:":
                _client = QdrantClient(location=":memory:")
            else:
                _client = QdrantClient(path=settings.qdrant_local_path)
        return _client


def _reset_client_for_tests() -> None:
    """Test-only: drop the singleton so a fresh get_client() (or a
    monkeypatched replacement) starts clean. Not used by production code.
    """
    global _client
    _client = None


def ensure_collection() -> None:
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            VECTOR_NAME: models.VectorParams(
                size=VECTOR_DIM,
                distance=models.Distance.COSINE,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                ),
            ),
            # Quantized for the same reason as VECTOR_NAME now, revised from
            # the original "not latency-sensitive" call: at 2048-dim this
            # field is the single largest memory consumer in the index
            # (~90MB raw float32 across ~11K chunks) and Render's free tier
            # hard-caps the whole container at 512MB RAM -- confirmed for
            # real, not theoretical (deploy exit 137 / SIGKILL from the
            # cgroup OOM killer, see docs/13-build-status.md). int8 scalar
            # quantization cuts this field's raw footprint ~4x.
            VECTOR_NAME_NVIDIA: models.VectorParams(
                size=settings.nvidia_embed_dim,
                distance=models.Distance.COSINE,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99, always_ram=False
                    )
                ),
            ),
        },
    )
    for field_name in ("language", "query_type", "has_numbers", "strategy", "parent_id"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def _strategy_filter(strategy_id: str, extra: models.Filter | None = None) -> models.Filter:
    must = [models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy_id))]
    if extra is not None:
        must.append(extra)
    return models.Filter(must=must)


# Oversample factor for search_dense_grouped: one unfiltered search fetching
# per_arm_k * this many candidates guarantees every strategy's true top-k is
# contained in the global result set (measured on T0: limit=600 -> >=52 hits
# per strategy, ~22ms; a filtered per-strategy search costs 100-150ms each in
# qdrant local mode because the payload filter forces a brute-force scan).
_GROUPED_OVERSAMPLE = 12


def search_dense_grouped(
    query_vector: list[float],
    per_arm_k: int,
    vector_name: str = VECTOR_NAME,
) -> tuple[list[list[models.ScoredPoint]], list[float]]:
    """Top-`per_arm_k` hits per chunking strategy from ONE unfiltered search.

    Equivalent output to running search_dense once per strategy (same ranked
    lists, modulo strategies whose entire candidate pool ranks below the
    global oversampled cutoff — not observed on real data), but ~30x faster
    against Qdrant's embedded/local engine. See _GROUPED_OVERSAMPLE.

    Also returns the global cosine scores of every fetched point (highest
    first) — this is the raw-similarity signal the coverage gate runs on;
    RRF fusion scores carry no absolute coverage signal (see
    api/guardrails/coverage_gate.py).
    """
    client = get_client()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=vector_name,
        limit=min(per_arm_k * _GROUPED_OVERSAMPLE, 2000),
        with_payload=True,
    )
    buckets: dict[str, list[models.ScoredPoint]] = {}
    for hit in result.points:
        strategy = (hit.payload or {}).get("strategy", "unknown")
        bucket = buckets.setdefault(strategy, [])
        if len(bucket) < per_arm_k:
            bucket.append(hit)
    # Deterministic arm order (sorted by strategy id) so fusion downstream is
    # stable regardless of dict insertion order.
    arms = [buckets[s] for s in sorted(buckets)]
    global_scores = [hit.score for hit in result.points]
    return arms, global_scores


def search_dense(
    strategy_id: str,
    query_vector: list[float],
    top_k: int = 50,
    query_filter: models.Filter | None = None,
    vector_name: str = VECTOR_NAME,
) -> list[models.ScoredPoint]:
    client = get_client()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=vector_name,
        limit=top_k,
        query_filter=_strategy_filter(strategy_id, query_filter),
        with_payload=True,
    )
    return result.points


def upsert_chunks(points: list[models.PointStruct]) -> None:
    """`points` must already carry vectors under VECTOR_NAME and a
    `strategy` payload field — see ingest/build_index.py.
    """
    ensure_collection()
    get_client().upsert(collection_name=COLLECTION_NAME, points=points)
