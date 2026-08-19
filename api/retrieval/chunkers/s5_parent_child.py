"""S5 · parent_child — small-to-big. See docs/03-chunking.md.

Index small children (1-2 sentences). Return the full parent passage.
Hypothesis: the cleanest decoupling of retrieval precision (small,
undiluted child vectors) from generation context (the full parent).
Multiple children of one parent collapsing to a single parent at assembly
is handled by retrieval/assemble.py's parent_id dedup — that's what makes
this strategy safe to combine with others in the fused top-k.
"""

from __future__ import annotations

from api.retrieval.chunkers.base import Chunk, PassageDoc
from api.retrieval.text_utils import sentence_spans

STRATEGY_ID = "s5_parent_child"
CHILD_SENTENCES = 2  # sentences per child unit


def chunk(doc: PassageDoc) -> list[Chunk]:
    spans = sentence_spans(doc.text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    for i in range(0, len(spans), CHILD_SENTENCES):
        group = spans[i : i + CHILD_SENTENCES]
        child_text = " ".join(s for s, _ in group)
        child_start = group[0][1][0]
        child_end = group[-1][1][1]

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}/{STRATEGY_ID}/c{i // CHILD_SENTENCES}",
                doc_id=doc.doc_id,
                parent_id=doc.doc_id,  # the full passage — assemble.py resolves & dedups on this
                strategy=STRATEGY_ID,
                text=doc.text,  # returned to the reader: the full parent
                embed_text=child_text,  # embedded: the small child only
                char_span=(child_start, child_end),
                language=doc.language,
                query_type=doc.query_type,
                is_selected=doc.is_selected,
            )
        )

    return chunks
