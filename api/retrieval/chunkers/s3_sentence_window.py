"""S3 · sentence_window — sentence index, window return. See docs/03-chunking.md.

Hypothesis: small units maximise embedding precision (one sentence = one
idea, undiluted vector), while the returned window preserves the context
needed to actually answer. This is the clearest example of embed_text
differing from the returned text: we embed one sentence but return it
plus ±WINDOW neighbours.
"""

from __future__ import annotations

from api.retrieval.chunkers.base import Chunk, PassageDoc
from api.retrieval.text_utils import sentence_spans

STRATEGY_ID = "s3_sentence_window"
WINDOW = 1  # neighbours on each side returned alongside the hit sentence


def chunk(doc: PassageDoc) -> list[Chunk]:
    spans = sentence_spans(doc.text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    for i, (sentence, _span) in enumerate(spans):
        lo = max(0, i - WINDOW)
        hi = min(len(spans), i + WINDOW + 1)
        window_sentences = [spans[j][0] for j in range(lo, hi)]
        window_text = " ".join(window_sentences)
        window_start = spans[lo][1][0]
        window_end = spans[hi - 1][1][1]

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}/{STRATEGY_ID}/c{i}",
                doc_id=doc.doc_id,
                parent_id=doc.doc_id,
                strategy=STRATEGY_ID,
                text=window_text,  # returned to the reader: sentence + neighbours
                embed_text=sentence,  # embedded: the sentence alone, undiluted
                char_span=(window_start, window_end),
                language=doc.language,
                query_type=doc.query_type,
                is_selected=doc.is_selected,
            )
        )

    return chunks
