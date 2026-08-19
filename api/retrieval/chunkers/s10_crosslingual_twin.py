"""S10 · crosslingual_twin — parallel twin indexing. See docs/03-chunking.md.

Hypothesis: recovers recall lost to bad machine translation. If a Hindi
passage was mistranslated, a Hindi query embeds semantically far from the
(drifted) Hindi passage vector — but the English original is intact, and a
*multilingual* embedder still places a semantically-similar Hindi query
reasonably close to it. So: embed the twin's text, but return this
passage's own (home-language) text. A Hindi query can therefore match via
the English embedding and still get a Hindi answer back — language
consistency is satisfied by construction, not by translating the answer
after the fact.

This only exists because MSMARCO-XI is a parallel corpus (docs/02-dataset.md
§2) — the strategy most specific to the provided dataset. Symmetric
coverage (English query -> Hindi twin -> English answer) falls out for
free as long as ingest/explode_dedupe.py creates English as its own
first-class PassageDoc set with twin pointers back to each Indic
counterpart, not just an attribute of the Indic rows.
"""

from __future__ import annotations

from api.retrieval.chunkers.base import Chunk, PassageDoc

STRATEGY_ID = "s10_crosslingual_twin"


def chunk(doc: PassageDoc) -> list[Chunk]:
    if not doc.twin_text or not doc.twin_doc_id:
        return []

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}/{STRATEGY_ID}/c0",
            doc_id=doc.doc_id,
            parent_id=doc.doc_id,
            strategy=STRATEGY_ID,
            text=doc.text,  # returned to the reader: this doc's own language
            embed_text=doc.twin_text,  # embedded: the twin's language
            char_span=(0, len(doc.text)),
            language=doc.language,
            query_type=doc.query_type,
            is_selected=doc.is_selected,
            extra={"twin_doc_id": doc.twin_doc_id},
        )
    ]
