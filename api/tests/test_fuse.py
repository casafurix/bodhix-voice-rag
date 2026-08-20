from api.retrieval.fuse import RRF_K, ScoredChunk, reciprocal_rank_fusion


def test_single_arm_preserves_order():
    arm = [ScoredChunk(chunk_id="a", score=0.9), ScoredChunk(chunk_id="b", score=0.5)]
    fused = reciprocal_rank_fusion([arm])
    assert [c.chunk_id for c in fused] == ["a", "b"]
    assert fused[0].score == 1.0 / (RRF_K + 1)


def test_chunk_appearing_in_multiple_arms_scores_higher():
    arm1 = [ScoredChunk(chunk_id="a", score=0.9), ScoredChunk(chunk_id="b", score=0.8)]
    arm2 = [ScoredChunk(chunk_id="b", score=0.7), ScoredChunk(chunk_id="a", score=0.1)]
    fused = reciprocal_rank_fusion([arm1, arm2])
    # "a" is rank1 in arm1 + rank2 in arm2; "b" is rank2 in arm1 + rank1 in arm2 — tied.
    scores = {c.chunk_id: c.score for c in fused}
    assert scores["a"] == scores["b"]
    assert scores["a"] == pytest_approx_rrf_sum([1, 2])


def pytest_approx_rrf_sum(ranks: list[int]) -> float:
    return sum(1.0 / (RRF_K + r) for r in ranks)


def test_exclusive_arm_items_still_included():
    arm1 = [ScoredChunk(chunk_id="a", score=0.9)]
    arm2 = [ScoredChunk(chunk_id="b", score=0.9)]
    fused = reciprocal_rank_fusion([arm1, arm2])
    assert {c.chunk_id for c in fused} == {"a", "b"}


def test_empty_ranked_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
