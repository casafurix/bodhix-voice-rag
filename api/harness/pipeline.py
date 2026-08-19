"""The retrieval-and-answer DAG for POST /ask. See docs/06-harness.md.

MVP scope vs. the full doc: no `rerank` stage (cut), no `answer_rich` stage
(cut). Every stage that remains still reports its timing into ctx.timings_ms
and the degradation ladder still has real rungs to pull (skip rerank was
rung #2 in the doc; since rerank doesn't exist yet in code, the ladder here
starts at rung #3 — reduce candidates — see docs/06-harness.md).
"""

from __future__ import annotations

from api.answer.extractive import select_span
from api.guardrails.coverage_gate import coverage_verdict
from api.guardrails.guard_in import run_guard_in
from api.guardrails.guard_out import run_guard_out
from api.harness.context import Context
from api.harness.deadline import Deadline
from api.harness.stage import StageShortCircuit, timed
from api.normalise import detect_language, normalise_text
from api.retrieval.assemble import AssembledChunk, assemble
from api.retrieval.embed import embed_query
from api.retrieval.fuse import ScoredChunk, reciprocal_rank_fusion
from api.retrieval.qdrant_store import search_dense
from api.retrieval.sparse import search_sparse
from api.retrieval.strategies import STRATEGY_IDS
from api.schemas import (
    Answer,
    AskRequest,
    AskResponse,
    Citation,
    GuardrailTrace,
    InputGuardrailTrace,
)

CANDIDATES_PER_ARM = 50
REDUCED_CANDIDATES_PER_ARM = 20  # degradation ladder rung 3, docs/06-harness.md


async def run_ask(request: AskRequest) -> AskResponse:
    deadline = Deadline(budget_ms=request.budget_ms)
    ctx = Context(deadline=deadline)

    try:
        return await _run_pipeline(request, ctx, deadline)
    except StageShortCircuit as short_circuit:
        return AskResponse(
            trace_id=ctx.trace_id,
            verdict="REFUSED",
            refusal_code=short_circuit.refusal_code,  # type: ignore[arg-type]
            timings_ms=ctx.timings_ms,
            degradations=ctx.degradations,
            guardrails=GuardrailTrace(
                input=InputGuardrailTrace(detail=short_circuit.detail),
            ),
        )


async def _run_pipeline(request: AskRequest, ctx: Context, deadline: Deadline) -> AskResponse:
    # 1. normalise
    async def _normalise():
        text = normalise_text(request.query)
        lang, lang_conf = detect_language(text, request.lang_hint)
        return text, lang, lang_conf

    text, lang, lang_conf = await timed(ctx, "normalise", _normalise())

    # 2. guard_in — may short-circuit
    async def _guard_in():
        return run_guard_in(text, lang)

    checks_passed = await timed(ctx, "guard_in", _guard_in())

    # 3. embed
    async def _embed():
        return embed_query(text)

    query_vector = await timed(ctx, "embed", _embed())

    # 4. hybrid retrieve — reduce candidate count if budget is already thin
    top_k = CANDIDATES_PER_ARM if deadline.affords(80) else REDUCED_CANDIDATES_PER_ARM
    if top_k < CANDIDATES_PER_ARM:
        ctx.degrade("reduced_candidates")

    async def _retrieve():
        # NOTE: Qdrant's point `.id` is an internal UUID (see
        # ingest/build_index.py) and must never leak into fusion/citations —
        # it isn't shared with the BM25 arm's identifier space. The payload's
        # `chunk_id` field (our own human-readable id) is what both arms and
        # the assembled citations use, so RRF can actually fuse the same
        # logical chunk across dense and sparse hits.
        ranked_lists: list[list[ScoredChunk]] = []
        chunk_payloads: dict[str, dict] = {}

        for strategy_id in STRATEGY_IDS:
            hits = search_dense(strategy_id, query_vector, top_k=top_k)
            for h in hits:
                payload = h.payload or {}
                cid = payload.get("chunk_id", str(h.id))
                chunk_payloads[cid] = payload
            ranked_lists.append(
                [
                    ScoredChunk(chunk_id=(h.payload or {}).get("chunk_id", str(h.id)), score=h.score)
                    for h in hits
                ]
            )

        sparse_hits = search_sparse(text, top_k=top_k)
        ranked_lists.append(
            [ScoredChunk(chunk_id=cid, score=score) for cid, score in sparse_hits]
        )

        return ranked_lists, chunk_payloads

    ranked_lists, chunk_payloads = await timed(ctx, "retrieve", _retrieve())

    # 5. fuse
    async def _fuse():
        return reciprocal_rank_fusion(ranked_lists)

    fused = await timed(ctx, "fuse", _fuse())

    # 6. coverage gate — may short-circuit
    async def _coverage():
        return coverage_verdict([c.score for c in fused])

    coverage_stats = await timed(ctx, "coverage_gate", _coverage())

    # 7. assemble (rerank stage cut from MVP — see module docstring)
    async def _assemble():
        candidates = [
            AssembledChunk(
                chunk_id=c.chunk_id,
                parent_id=chunk_payloads.get(c.chunk_id, {}).get("parent_id", c.chunk_id),
                text=chunk_payloads.get(c.chunk_id, {}).get("text", ""),
                strategy=chunk_payloads.get(c.chunk_id, {}).get("strategy", "unknown"),
                score=c.score,
                language=chunk_payloads.get(c.chunk_id, {}).get("language", lang),
            )
            for c in fused
        ]
        return assemble(candidates)

    assembled = await timed(ctx, "assemble", _assemble())

    if not assembled.blocks:
        raise StageShortCircuit("OUT_OF_SCOPE", "no chunks survived assembly")

    # 8. fast answer (extractive)
    async def _answer_fast():
        return select_span(text, assembled.blocks[0])

    extractive = await timed(ctx, "answer_fast", _answer_fast())

    # 9. guard_out — may veto
    async def _guard_out():
        return run_guard_out(
            answer_text=extractive.text,
            context=assembled.text,
            query=text,
            query_lang=lang,
            answer_lang=lang,  # MVP: extractive answers inherit the chunk's language
            cited_chunk_ids=[extractive.chunk_id],
            supplied_chunk_ids=assembled.supplied_chunk_ids,
            cited_chunk_text=assembled.blocks[0].text,
        )

    output_trace = await timed(ctx, "guard_out", _guard_out())

    return AskResponse(
        trace_id=ctx.trace_id,
        verdict="ANSWERED",
        answer=Answer(text=extractive.text, language=lang),
        citations=[
            Citation(
                chunk_id=extractive.chunk_id,
                score=fused[0].score if fused else 0.0,
                strategy=extractive.strategy,
                span=extractive.char_span,
            )
        ],
        guardrails=GuardrailTrace(
            input=InputGuardrailTrace(checks=checks_passed),
            output=output_trace,
            coverage=coverage_stats.model_dump(),
        ),
        timings_ms=ctx.timings_ms,
        degradations=ctx.degradations,
    )
