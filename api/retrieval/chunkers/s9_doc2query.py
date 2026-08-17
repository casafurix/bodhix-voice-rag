"""S9 · doc2query_multivector — generated-question indexing.
See docs/03-chunking.md.

Hypothesis: matching a user's question against an indexed *question* is
easier than matching it against a declarative passage — closes the
query/document asymmetry gap directly.

MVP scope: MSMARCO-XI already carries one real, gold question per passage
set (`query`/`Eng_Query`) at zero LLM cost. We index only that free
question — the doc's own suggested cost-gated experiment ("measure how
much of the gain comes from the free one versus the generated ones") is
exactly why we start here rather than paying for an LLM pass. Adding 3-5
LLM-generated questions per passage is a documented, additive upgrade if
the ablation shows the free question alone isn't enough.

If a passage has no associated free_question (shouldn't happen post
explode/dedupe, but chunkers must never assume clean input), this
strategy produces zero chunks for that passage rather than guessing.
"""

from __future__ import annotations

from api.retrieval.chunkers.base import Chunk, PassageDoc

STRATEGY_ID = "s9_doc2query"


def chunk(doc: PassageDoc) -> list[Chunk]:
    if not doc.free_question:
        return []

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}/{STRATEGY_ID}/c0",
            doc_id=doc.doc_id,
            parent_id=doc.doc_id,
            strategy=STRATEGY_ID,
            text=doc.text,  # returned to the reader: the actual passage
            embed_text=doc.free_question,  # embedded: the question it answers
            char_span=(0, len(doc.text)),
            language=doc.language,
            query_type=doc.query_type,
            is_selected=doc.is_selected,
            extra={"source": "free_query_field"},
        )
    ]
