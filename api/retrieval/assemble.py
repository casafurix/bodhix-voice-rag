"""Context assembly — between fusion/coverage-gate and answer_fast.
See docs/04-retrieval.md, "Context assembly".

Order matters: parent resolution and dedup happen before token budgeting, or
the budget fills with repeated text from multiple children of one parent
(the S5 parent_child failure mode the docs call out explicitly).
"""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_TOKEN_BUDGET = 400  # ~ chars/4, generous for 5 short chunks


class AssembledChunk(BaseModel):
    chunk_id: str
    parent_id: str
    text: str
    strategy: str
    score: float
    language: str


class AssembledContext(BaseModel):
    blocks: list[AssembledChunk]

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def supplied_chunk_ids(self) -> set[str]:
        return {b.chunk_id for b in self.blocks}


def assemble(
    candidates: list[AssembledChunk],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> AssembledContext:
    """Dedup by parent_id (keep the highest-scoring child per parent, in
    fused-rank order), then fill to the token budget.
    """
    seen_parents: set[str] = set()
    deduped: list[AssembledChunk] = []
    for c in candidates:  # already sorted best-first by the caller
        if c.parent_id in seen_parents:
            continue
        seen_parents.add(c.parent_id)
        deduped.append(c)

    budgeted: list[AssembledChunk] = []
    running_tokens = 0
    for c in deduped:
        approx_tokens = max(len(c.text) // 4, 1)
        if running_tokens + approx_tokens > token_budget and budgeted:
            break
        budgeted.append(c)
        running_tokens += approx_tokens

    return AssembledContext(blocks=budgeted)
