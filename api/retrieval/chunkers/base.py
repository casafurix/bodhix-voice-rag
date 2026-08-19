"""Chunker protocol — see docs/03-chunking.md.

The key property: the text embedded and the text returned need not be the
same string (S9 doc2query, S10 crosslingual_twin both exploit this).
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class PassageDoc(BaseModel):
    """One deduped passage instance, post explode/dedupe (ingest/explode_dedupe.py)."""

    doc_id: str  # e.g. "hi/1185869/p0"
    text: str
    language: str
    script: str
    query_id: int
    query_type: str
    is_selected: bool
    twin_doc_id: str | None = None  # aligned passage in the other language, if any
    twin_text: str | None = None
    # The dataset's own `query`/`Eng_Query` field for this row — a real question
    # this passage answers, at zero LLM cost. See docs/03-chunking.md, S9's
    # "free gold question" note. LLM-generated questions (3-5 per passage) are
    # cut from the MVP; only this free one is indexed.
    free_question: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    parent_id: str
    strategy: str
    text: str  # the text returned to the reader
    embed_text: str  # the text actually embedded — may differ from `text`
    char_span: tuple[int, int]
    language: str
    query_type: str
    is_selected: bool
    extra: dict = {}


class Chunker(Protocol):
    id: str

    def chunk(self, doc: PassageDoc) -> list[Chunk]: ...
