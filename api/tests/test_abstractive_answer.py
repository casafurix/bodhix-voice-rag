import pytest

from api.answer import abstractive
from api.harness.deadline import Deadline
from api.retrieval.assemble import AssembledChunk


def _block(chunk_id, text):
    return AssembledChunk(
        chunk_id=chunk_id, parent_id=chunk_id, text=text, strategy="s1_fixed", score=1.0, language="en"
    )


def test_parse_sources_valid_indices():
    blocks = [_block("c1", "a"), _block("c2", "b"), _block("c3", "c")]
    text, cited = abstractive._parse_sources("The answer is X.\nSOURCES: 1,3", blocks)
    assert text == "The answer is X."
    assert cited == ["c1", "c3"]


def test_parse_sources_out_of_range_indices_dropped():
    blocks = [_block("c1", "a")]
    _text, cited = abstractive._parse_sources("Answer text.\nSOURCES: 1,5,9", blocks)
    assert cited == ["c1"]


def test_parse_sources_none_falls_back_to_all_blocks():
    blocks = [_block("c1", "a"), _block("c2", "b")]
    _text, cited = abstractive._parse_sources("Answer text.\nSOURCES: none", blocks)
    assert cited == ["c1", "c2"]


def test_parse_sources_missing_line_falls_back_to_all_blocks():
    blocks = [_block("c1", "a"), _block("c2", "b")]
    text, cited = abstractive._parse_sources("Answer text with no sources line at all.", blocks)
    assert text == "Answer text with no sources line at all."
    assert cited == ["c1", "c2"]


@pytest.mark.asyncio
async def test_generate_answer_calls_llm_and_parses_result(monkeypatch):
    async def fake_agenerate_answer(messages, deadline, **kwargs):
        assert any("Question:" in m["content"] for m in messages if m["role"] == "user")
        return "Python is a programming language.\nSOURCES: 1"

    monkeypatch.setattr(abstractive, "agenerate_answer", fake_agenerate_answer)

    blocks = [_block("c1", "Python is a high-level language.")]
    result = await abstractive.generate_answer("What is Python?", blocks, Deadline(budget_ms=5000))
    assert result.text == "Python is a programming language."
    assert result.cited_chunk_ids == ["c1"]
