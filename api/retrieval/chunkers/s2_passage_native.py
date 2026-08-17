"""S2 · passage_native — atomic passage, no splitting. See docs/03-chunking.md.

Hypothesis: a strong baseline that many fancier strategies will fail to
beat, because MS MARCO passages are already well-formed retrieval units.
This is the honest baseline the ablation is measured against.
"""

from __future__ import annotations

from api.retrieval.chunkers.base import Chunk, PassageDoc

STRATEGY_ID = "s2_passage_native"


def chunk(doc: PassageDoc) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}/{STRATEGY_ID}/c0",
            doc_id=doc.doc_id,
            parent_id=doc.doc_id,
            strategy=STRATEGY_ID,
            text=doc.text,
            embed_text=doc.text,
            char_span=(0, len(doc.text)),
            language=doc.language,
            query_type=doc.query_type,
            is_selected=doc.is_selected,
        )
    ]
