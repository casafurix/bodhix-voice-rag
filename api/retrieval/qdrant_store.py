"""Qdrant in embedded/local mode — no server process, no network hop.
See docs/04-retrieval.md and the architecture note on avoiding microservices.

Named vectors per chunking strategy let every strategy in
retrieval/chunkers/ coexist in one collection (docs/03-chunking.md, S12
rationale) — the ablation becomes a query-parameter change, not N indexes.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient, models

from api.config import settings

COLLECTION_NAME = "chunks"
VECTOR_DIM = 384  # multilingual-e5-small


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(path=settings.qdrant_local_path)


def ensure_collection(strategy_ids: list[str]) -> None:
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            strategy_id: models.VectorParams(
                size=VECTOR_DIM,
                distance=models.Distance.COSINE,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                ),
            )
            for strategy_id in strategy_ids
        },
    )
    for field_name in ("language", "query_type", "has_numbers", "strategy"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


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
        using=strategy_id,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return result.points


def upsert_chunks(strategy_id: str, points: list[models.PointStruct]) -> None:
    get_client().upsert(collection_name=COLLECTION_NAME, points=points)
