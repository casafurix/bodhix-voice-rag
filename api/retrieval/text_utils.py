"""Shared sentence-splitting used by S3/S5 chunkers and the extractive
answer path. See docs/03-chunking.md, S3 note on Indic sentence boundaries.

Covers the Devanagari danda (।), Urdu's ۔, and standard Latin terminators.
Not a full indic-nlp-library-grade splitter — flagged as a known,
acceptable gap for the MVP rather than pretending "." is universal.
"""

from __future__ import annotations

import regex

_SENTENCE_SPLIT_RE = regex.compile(r"(?<=[.!?।۔॥])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def sentence_spans(text: str) -> list[tuple[str, tuple[int, int]]]:
    """Same split, but also returns each sentence's (start, end) char offset
    in the original text — needed by chunkers that must record char_span.
    """
    spans: list[tuple[str, tuple[int, int]]] = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        if start == -1:  # defensive: shouldn't happen, but never crash a chunker over it
            start = cursor
        end = start + len(sentence)
        spans.append((sentence, (start, end)))
        cursor = end
    return spans
