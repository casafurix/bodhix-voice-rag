import pytest

from api.guardrails.guard_out import (
    check_citation_integrity,
    check_extractive_span,
    check_language_match,
    check_lexical_overlap_groundedness,
    check_numeric_grounding,
    check_quality_floor,
    run_guard_out,
)
from api.harness.stage import StageShortCircuit

# ---- check_numeric_grounding ----


def test_numeric_grounding_passes_when_numbers_subset_of_context():
    assert check_numeric_grounding("Everest is 8849 metres tall.", "Everest peak is 8849 metres.") == "pass"


def test_numeric_grounding_fails_on_hallucinated_number():
    with pytest.raises(StageShortCircuit) as exc:
        check_numeric_grounding("Everest is 9999 metres tall.", "Everest peak is 8849 metres.")
    assert exc.value.refusal_code == "UNGROUNDED_ANSWER"


def test_numeric_grounding_handles_devanagari_and_bengali_numerals():
    assert check_numeric_grounding("এটি ৮৮৪৯ মিটার", "চূড়ার উচ্চতা ৮৮৪৯ মিটার") == "pass"


# ---- check_citation_integrity ----


def test_citation_integrity_passes_when_subset():
    assert check_citation_integrity(["c1"], {"c1", "c2"}) == "pass"


def test_citation_integrity_fails_on_unsupplied_chunk():
    with pytest.raises(StageShortCircuit) as exc:
        check_citation_integrity(["c3"], {"c1", "c2"})
    assert exc.value.refusal_code == "UNGROUNDED_ANSWER"


# ---- check_extractive_span ----


def test_extractive_span_passes_for_verbatim_substring():
    check_extractive_span("Paris is the capital.", "France's capital city. Paris is the capital. Done.")


def test_extractive_span_fails_when_not_a_substring():
    with pytest.raises(StageShortCircuit) as exc:
        check_extractive_span("This text was never in the chunk.", "Some unrelated chunk text.")
    assert exc.value.refusal_code == "UNGROUNDED_ANSWER"


# ---- check_lexical_overlap_groundedness ----


def test_lexical_overlap_passes_above_threshold():
    overlap = check_lexical_overlap_groundedness(
        "Everest peak elevation is 8849 metres", "Everest peak elevation reaches 8849 metres above sea level"
    )
    assert overlap >= 0.3


def test_lexical_overlap_fails_below_threshold():
    with pytest.raises(StageShortCircuit) as exc:
        check_lexical_overlap_groundedness("completely unrelated words here", "Everest peak elevation 8849 metres")
    assert exc.value.refusal_code == "UNGROUNDED_ANSWER"


# ---- check_language_match ----


def test_language_match_passes_when_equal():
    assert check_language_match("en", "en") is True


def test_language_match_fails_when_different():
    with pytest.raises(StageShortCircuit) as exc:
        check_language_match("hi", "en")
    assert exc.value.refusal_code == "UNGROUNDED_ANSWER"


# ---- check_quality_floor ----


def test_quality_floor_fails_on_empty_answer():
    with pytest.raises(StageShortCircuit):
        check_quality_floor("", "some query")


def test_quality_floor_fails_on_single_token_answer():
    with pytest.raises(StageShortCircuit):
        check_quality_floor("yes", "some query")


def test_quality_floor_fails_on_verbatim_echo():
    with pytest.raises(StageShortCircuit):
        check_quality_floor("what is python", "What is Python")


def test_quality_floor_passes_on_real_answer():
    check_quality_floor("Python is a programming language.", "What is Python?")


# ---- run_guard_out: the mode-conditional regression test ----


def test_run_guard_out_extractive_requires_verbatim_span():
    trace = run_guard_out(
        answer_text="Everest peak is 8849 metres.",
        context="Everest peak is 8849 metres. It is the tallest mountain.",
        query="How tall is Everest?",
        query_lang="en",
        answer_lang="en",
        cited_chunk_ids=["c1"],
        supplied_chunk_ids={"c1"},
        cited_chunk_text="Everest peak is 8849 metres. It is the tallest mountain.",
        answer_mode="extractive",
    )
    assert trace.numeric_check == "pass"
    assert trace.citation_check == "pass"


def test_run_guard_out_abstractive_answer_not_a_substring_still_passes():
    """The key regression: an abstractive answer paraphrases the context and is
    NOT a verbatim substring of any chunk — it must still pass guard_out when
    numeric/citation/lexical checks hold, unlike the extractive path.
    """
    trace = run_guard_out(
        answer_text="Everest reaches a peak elevation of 8849 metres, making it the tallest mountain.",
        context="Mount Everest is the tallest mountain above sea level, with a peak elevation of 8849 metres.",
        query="How tall is Everest?",
        query_lang="en",
        answer_lang="en",
        cited_chunk_ids=["c1"],
        supplied_chunk_ids={"c1"},
        answer_mode="abstractive",
    )
    assert trace.numeric_check == "pass"
    assert trace.citation_check == "pass"
    assert trace.groundedness_method == "lexical_overlap"


def test_run_guard_out_abstractive_still_rejects_hallucinated_numbers():
    with pytest.raises(StageShortCircuit) as exc:
        run_guard_out(
            answer_text="Everest reaches a peak elevation of 12345 metres.",
            context="Mount Everest is the tallest mountain, with a peak elevation of 8849 metres.",
            query="How tall is Everest?",
            query_lang="en",
            answer_lang="en",
            cited_chunk_ids=["c1"],
            supplied_chunk_ids={"c1"},
            answer_mode="abstractive",
        )
    assert exc.value.refusal_code == "UNGROUNDED_ANSWER"
