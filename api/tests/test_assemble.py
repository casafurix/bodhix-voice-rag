from api.retrieval.assemble import AssembledChunk, assemble


def _chunk(chunk_id, parent_id, text, score=1.0):
    return AssembledChunk(
        chunk_id=chunk_id, parent_id=parent_id, text=text, strategy="s1_fixed",
        score=score, language="en",
    )


def test_dedup_keeps_first_occurrence_per_parent():
    candidates = [
        _chunk("c1", "p1", "first child of p1", score=0.9),
        _chunk("c2", "p1", "second child of p1", score=0.8),
        _chunk("c3", "p2", "only child of p2", score=0.7),
    ]
    result = assemble(candidates)
    assert [b.chunk_id for b in result.blocks] == ["c1", "c3"]


def test_token_budget_cutoff_stops_filling():
    long_text = "word " * 2000  # ~10000 chars => ~2500 approx-tokens, over any small budget
    candidates = [
        _chunk("c1", "p1", long_text),
        _chunk("c2", "p2", "short text"),
    ]
    result = assemble(candidates, token_budget=100)
    assert len(result.blocks) == 1
    assert result.blocks[0].chunk_id == "c1"


def test_always_includes_at_least_one_block_even_over_budget():
    long_text = "word " * 5000
    candidates = [_chunk("c1", "p1", long_text)]
    result = assemble(candidates, token_budget=10)
    assert len(result.blocks) == 1


def test_supplied_chunk_ids_and_text_property():
    candidates = [_chunk("c1", "p1", "alpha"), _chunk("c2", "p2", "beta")]
    result = assemble(candidates)
    assert result.supplied_chunk_ids == {"c1", "c2"}
    assert result.text == "alpha\n\nbeta"


def test_empty_candidates_returns_empty_context():
    result = assemble([])
    assert result.blocks == []
    assert result.supplied_chunk_ids == set()
