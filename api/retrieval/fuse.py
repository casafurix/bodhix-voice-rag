"""Reciprocal Rank Fusion — see docs/04-retrieval.md.

Ranks, not scores, so dense (cosine) and sparse (BM25) results are fused
without needing to calibrate incomparable score scales.
"""

from __future__ import annotations

from pydantic import BaseModel

RRF_K = 60


class ScoredChunk(BaseModel):
    chunk_id: str
    score: float  # original arm score, kept for the coverage gate / trace


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]], k: int = RRF_K
) -> list[ScoredChunk]:
    """Each input list is one arm's results, already sorted best-first.
    Returns a fused, sorted list. The returned `score` is the RRF score
    (not the original arm score) so the coverage gate sees the fused signal.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + 1.0 / (k + rank)

    ordered_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [ScoredChunk(chunk_id=cid, score=fused[cid]) for cid in ordered_ids]
