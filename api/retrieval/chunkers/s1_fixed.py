"""S1 · fixed_512_64 — the fixed-size control. See docs/03-chunking.md.

Hypothesis: will underperform, because it violates passage boundaries.
Purpose: the control the brief warns against — required for the ablation
to mean anything. We implement it anyway, precisely so it can lose.
"""

from __future__ import annotations

from api.retrieval.chunkers.base import Chunk, PassageDoc

STRATEGY_ID = "s1_fixed"
WINDOW_CHARS = 512 * 4  # ~4 chars/token approximation, good enough for a control
STRIDE_CHARS = 64 * 4


def chunk(doc: PassageDoc) -> list[Chunk]:
    text = doc.text
    if len(text) <= WINDOW_CHARS:
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}/{STRATEGY_ID}/c0",
                doc_id=doc.doc_id,
                parent_id=doc.doc_id,
                strategy=STRATEGY_ID,
                text=text,
                embed_text=text,
                char_span=(0, len(text)),
                language=doc.language,
                query_type=doc.query_type,
                is_selected=doc.is_selected,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + WINDOW_CHARS, len(text))
        window = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}/{STRATEGY_ID}/c{idx}",
                doc_id=doc.doc_id,
                parent_id=doc.doc_id,
                strategy=STRATEGY_ID,
                text=window,
                embed_text=window,
                char_span=(start, end),
                language=doc.language,
                query_type=doc.query_type,
                is_selected=doc.is_selected,
            )
        )
        if end == len(text):
            break
        start += WINDOW_CHARS - STRIDE_CHARS
        idx += 1

    return chunks
