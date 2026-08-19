"""Maps each strategy id (api/retrieval/strategies.py) to its chunk()
function, so ingest/build_index.py can iterate without a chain of ifs.
"""

from __future__ import annotations

from typing import Callable

from api.retrieval.chunkers import (
    s1_fixed,
    s2_passage_native,
    s3_sentence_window,
    s5_parent_child,
    s9_doc2query,
    s10_crosslingual_twin,
)
from api.retrieval.chunkers.base import Chunk, PassageDoc

CHUNKERS: dict[str, Callable[[PassageDoc], list[Chunk]]] = {
    s1_fixed.STRATEGY_ID: s1_fixed.chunk,
    s2_passage_native.STRATEGY_ID: s2_passage_native.chunk,
    s3_sentence_window.STRATEGY_ID: s3_sentence_window.chunk,
    s5_parent_child.STRATEGY_ID: s5_parent_child.chunk,
    s9_doc2query.STRATEGY_ID: s9_doc2query.chunk,
    s10_crosslingual_twin.STRATEGY_ID: s10_crosslingual_twin.chunk,
}


def chunk_with_all_strategies(doc: PassageDoc) -> list[Chunk]:
    """Runs every registered chunker over one PassageDoc, concatenating
    results. A strategy that can't apply (e.g. S9 with no free_question,
    S10 with no twin) simply returns an empty list — never an error.
    """
    chunks: list[Chunk] = []
    for chunk_fn in CHUNKERS.values():
        chunks.extend(chunk_fn(doc))
    return chunks
