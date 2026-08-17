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

from functools import lru_cache

from qdrant_client import QdrantClient, models

from api.config import settings

COLLECTION_NAME = "chunks"
VECTOR_NAME = "dense"
VECTOR_DIM = 384  # matches api/retrieval/embed.py's model output dim


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(path=settings.qdrant_local_path)


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
            )
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


def search_dense(
    strategy_id: str,
    query_vector: list[float],
    top_k: int = 50,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    client = get_client()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=VECTOR_NAME,
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
