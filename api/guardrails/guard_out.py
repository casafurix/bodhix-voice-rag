"""Output guardrails — see docs/07-guardrails.md.

MVP scope: numeric grounding, citation integrity, language consistency,
extractive span verification, answer-quality floor, and lexical-overlap
groundedness as the documented fallback for the NLI entailment model (which
is cut from the MVP — see docs/07-guardrails.md, "Groundedness via NLI
entailment", fallback paragraph).
"""

from __future__ import annotations

from typing import Literal

import regex

from api.harness.stage import StageShortCircuit
from api.schemas import OutputGuardrailTrace

# Covers ASCII digits plus Devanagari (०-९) and Bengali (০-৯) numerals.
_NUMBER_RE = regex.compile(r"[0-9०-९০-৯]+(?:[.,][0-9०-९০-৯]+)*")

_LEXICAL_OVERLAP_MIN = 0.3  # fraction of answer content-words that must appear in context


def _extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def check_numeric_grounding(answer: str, context: str) -> Literal["pass", "fail"]:
    answer_nums = _extract_numbers(answer)
    context_nums = _extract_numbers(context)
    if not answer_nums.issubset(context_nums):
        raise StageShortCircuit(
            "UNGROUNDED_ANSWER",
            f"numeric token(s) not in context: {answer_nums - context_nums}",
        )
    return "pass"


def check_citation_integrity(cited_chunk_ids: list[str], supplied_chunk_ids: set[str]) -> str:
    if not set(cited_chunk_ids).issubset(supplied_chunk_ids):
        raise StageShortCircuit("UNGROUNDED_ANSWER", "citation references a chunk not supplied")
    return "pass"


def check_extractive_span(answer_text: str, chunk_text: str) -> None:
    """Fast-path only: the answer must be byte-for-byte a substring of the
    cited chunk. If not, we have a bug and fail closed.
    """
    if answer_text not in chunk_text:
        raise StageShortCircuit("UNGROUNDED_ANSWER", "extractive answer is not a substring of chunk")


def check_lexical_overlap_groundedness(answer: str, context: str) -> float:
    """Fallback groundedness check (documented in docs/07-guardrails.md as
    the accepted degradation when no NLI model is available/affordable).
    Cheap content-word overlap between answer and context.
    """
    def content_words(text: str) -> set[str]:
        return {w.lower() for w in regex.findall(r"\w+", text, flags=regex.UNICODE) if len(w) > 2}

    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    context_words = content_words(context)
    overlap = len(answer_words & context_words) / len(answer_words)
    if overlap < _LEXICAL_OVERLAP_MIN:
        raise StageShortCircuit(
            "UNGROUNDED_ANSWER", f"lexical overlap {overlap:.2f} < {_LEXICAL_OVERLAP_MIN}"
        )
    return overlap


def check_language_match(answer_lang: str, query_lang: str) -> bool:
    match = answer_lang == query_lang
    if not match:
        raise StageShortCircuit(
            "UNGROUNDED_ANSWER", f"answer language {answer_lang} != query language {query_lang}"
        )
    return True


def check_quality_floor(answer: str, query: str) -> None:
    stripped = answer.strip()
    if not stripped or len(stripped.split()) <= 1:
        raise StageShortCircuit("UNGROUNDED_ANSWER", "degenerate answer (empty or single token)")
    if stripped.lower() == query.strip().lower():
        raise StageShortCircuit("UNGROUNDED_ANSWER", "answer is a verbatim echo of the question")


def run_guard_out(
    *,
    answer_text: str,
    context: str,
    query: str,
    query_lang: str,
    answer_lang: str,
    cited_chunk_ids: list[str],
    supplied_chunk_ids: set[str],
    cited_chunk_text: str,
) -> OutputGuardrailTrace:
    check_quality_floor(answer_text, query)
    check_extractive_span(answer_text, cited_chunk_text)
    numeric_result = check_numeric_grounding(answer_text, context)
    citation_result = check_citation_integrity(cited_chunk_ids, supplied_chunk_ids)
    language_match = check_language_match(answer_lang, query_lang)
    overlap = check_lexical_overlap_groundedness(answer_text, context)

    return OutputGuardrailTrace(
        numeric_check=numeric_result,
        citation_check=citation_result,
        language_match=language_match,
        groundedness_score=overlap,
        groundedness_method="lexical_overlap",
    )
