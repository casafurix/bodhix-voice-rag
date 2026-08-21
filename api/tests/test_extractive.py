from api.answer.extractive import select_span
from api.retrieval.assemble import AssembledChunk


def _chunk(text):
    return AssembledChunk(
        chunk_id="c1", parent_id="p1", text=text, strategy="s1_fixed", score=1.0, language="en"
    )


def test_selects_sentence_with_highest_query_overlap():
    text = (
        "Paris is the capital of France. Bananas are a good source of potassium. "
        "The Eiffel Tower is located in Paris."
    )
    chunk = _chunk(text)
    result = select_span("Where is the Eiffel Tower located?", chunk)
    assert "Eiffel Tower" in result.text
    assert result.chunk_id == "c1"
    assert result.strategy == "s1_fixed"


def test_char_span_matches_returned_text():
    text = "First sentence here. Second sentence about Everest and its height."
    chunk = _chunk(text)
    result = select_span("How tall is Everest?", chunk)
    start, end = result.char_span
    assert text[start:end] == result.text


def test_prefers_answer_sentence_over_generic_sentence():
    chunk = _chunk(
        "Mount Everest is in Asia. Mount Everest has a peak elevation of 8849 metres."
    )
    result = select_span("How tall is Mount Everest?", chunk)
    assert "8849 metres" in result.text


def test_single_sentence_chunk_returns_that_sentence():
    text = "Only one sentence in this chunk."
    chunk = _chunk(text)
    result = select_span("anything at all", chunk)
    assert result.text == text
