"""Extractive fast-path answer — see docs/01-architecture.md.

MVP note: the doc's design scores spans using the reranker's token-level
signal (stage 6). Since the reranker is cut from the MVP, span scoring here
falls back to a lexical-overlap heuristic against the query — still
grounded by construction (the answer is a verbatim substring of a retrieved
chunk), just a cheaper selection method. Upgrading to cross-encoder span
scoring is additive once reranker/ ships.
"""

from __future__ import annotations

import regex
from pydantic import BaseModel

from api.retrieval.assemble import AssembledChunk
from api.retrieval.text_utils import split_sentences


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in regex.findall(r"\w+", text, flags=regex.UNICODE) if len(w) > 1}


class ExtractiveAnswer(BaseModel):
    text: str
    chunk_id: str
    strategy: str
    char_span: tuple[int, int]


def select_span(query: str, top_chunk: AssembledChunk) -> ExtractiveAnswer:
    query_words = _content_words(query)
    sentences = split_sentences(top_chunk.text)

    best_sentence = sentences[0]
    best_overlap = -1.0
    cursor = 0
    best_span = (0, len(best_sentence))

    for sentence in sentences:
        start = top_chunk.text.find(sentence, cursor)
        end = start + len(sentence)
        cursor = end
        overlap = len(query_words & _content_words(sentence))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
            best_span = (start, end)

    return ExtractiveAnswer(
        text=best_sentence,
        chunk_id=top_chunk.chunk_id,
        strategy=top_chunk.strategy,
        char_span=best_span,
    )
