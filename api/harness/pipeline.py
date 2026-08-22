"""The retrieval-and-answer DAG. See docs/06-harness.md.

Two entrypoints share this DAG: `run_ask` (text query, local embedding,
extractive-by-default) and `pipeline_voice.run_ask_voice` (audio query,
NVIDIA online embedding, abstractive-by-default). `normalise_and_guard`,
`run_retrieval_and_answer` and `build_refusal_response` are the shared
pieces both call into — see docs/13-build-status.md.

MVP scope vs. the full doc: no `rerank` stage (cut). Every stage that
remains still reports its timing into ctx.timings_ms and the degradation
ladder still has real rungs to pull: reduced candidate count under budget
pressure (rung 3, rerank being rung 2 doesn't exist yet), and a fallback
from NVIDIA embedding/LLM to local embedding/extractive answer if the
NVIDIA API fails or times out — a live provider outage degrades the
answer, it doesn't take the request down.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from api.answer.abstractive import generate_answer as generate_abstractive_answer
from api.answer.extractive import select_span
from api.config import settings
from api.guardrails.coverage_gate import coverage_verdict
from api.guardrails.guard_in import run_guard_in
from api.guardrails.guard_out import run_guard_out
from api.harness.context import Context
from api.harness.deadline import Deadline
from api.harness.stage import StageShortCircuit, timed
from api.llm.nvidia_client import NvidiaCallError, aembed_query
from api.normalise import detect_language, normalise_text
from api.retrieval.assemble import AssembledChunk, assemble
from api.retrieval.embed import embed_query
from api.retrieval.fuse import ScoredChunk, reciprocal_rank_fusion
from api.retrieval.qdrant_store import VECTOR_NAME, VECTOR_NAME_NVIDIA, search_dense_grouped
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
        text, lang, _lang_conf, checks_passed = await normalise_and_guard(
            request.query, request.lang_hint, ctx
        )
        return await run_retrieval_and_answer(
            text,
            lang,
            checks_passed,
            ctx,
            deadline,
            embedding_provider=settings.embedding_provider,
            answer_mode=request.options.answer_mode,
        )
    except StageShortCircuit as short_circuit:
        return build_refusal_response(ctx, short_circuit)


async def normalise_and_guard(
    raw_text: str, lang_hint: str | None, ctx: Context
) -> tuple[str, str, float, dict[str, bool]]:
    """Stages 1-2: normalise + guard_in. Shared by the text and voice paths —
    a voice request runs this over the STT transcript, not the raw audio.
    """

    async def _normalise():
        text = normalise_text(raw_text)
        lang, lang_conf = detect_language(text, lang_hint)
        return text, lang, lang_conf

    text, lang, lang_conf = await timed(ctx, "normalise", _normalise())

    async def _guard_in():
        return run_guard_in(text, lang)

    checks_passed = await timed(ctx, "guard_in", _guard_in())
    return text, lang, lang_conf, checks_passed


async def run_retrieval_and_answer(
    text: str,
    lang: str,
    checks_passed: dict[str, bool],
    ctx: Context,
    deadline: Deadline,
    *,
    embedding_provider: Literal["local", "nvidia"],
    answer_mode: Literal["extractive", "abstractive"],
) -> AskResponse:
    """Stages 3-9: embed, retrieve (concurrent), fuse, coverage_gate,
    assemble, answer, guard_out. `embedding_provider` and `answer_mode` are
    the effective values requested; both can degrade at runtime (NVIDIA
    unavailable -> local embedding / extractive answer) and the *actual*
    values used are what end up in the response.
    """
    effective_provider = embedding_provider

    # 3. embed
    async def _embed():
        nonlocal effective_provider
        if effective_provider == "nvidia":
            try:
                return await aembed_query(text, deadline.child(min(deadline.remaining_ms, 3000)))
            except NvidiaCallError as exc:
                if not settings.coverage_local_reembed:
                    # Memory-constrained deployment mode: falling back to
                    # embed_query() here would load the 224MB local model
                    # this mode exists specifically to avoid -- confirmed
                    # live that doing so OOMs the whole container (exit
                    # 137/502), not a clean per-request error. Surface a
                    # real, diagnosable refusal instead of crashing.
                    raise StageShortCircuit(
                        "INTERNAL_ERROR", f"NVIDIA embedding failed, local fallback disabled: {exc}"
                    ) from exc
                ctx.degrade("nvidia_embed_failed_fallback_local")
                effective_provider = "local"
        return embed_query(text)

    query_vector = await timed(ctx, "embed", _embed())
    vector_name = VECTOR_NAME if effective_provider == "local" else VECTOR_NAME_NVIDIA

    # 4. hybrid retrieve — all dense arms come from ONE unfiltered grouped
    # search (search_dense_grouped buckets top-k per strategy client-side;
    # a per-strategy filtered search costs 100-150ms each in qdrant local
    # mode — brute-force under a payload filter — which put the whole stage
    # at ~600ms against the 200ms t_core budget). The single dense call and
    # the BM25 arm run concurrently via worker threads; both release the GIL
    # during their numeric work, so cost approaches max() of the two arms.
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
        dense_call = asyncio.to_thread(
            search_dense_grouped, query_vector, top_k, vector_name
        )
        sparse_call = asyncio.to_thread(search_sparse, text, top_k)
        (dense_results, dense_cosine_scores), sparse_hits = await asyncio.gather(
            dense_call, sparse_call
        )

        ranked_lists: list[list[ScoredChunk]] = []
        chunk_payloads: dict[str, dict] = {}

        for hits in dense_results:
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

        ranked_lists.append([ScoredChunk(chunk_id=cid, score=score) for cid, score in sparse_hits])
        return ranked_lists, chunk_payloads, dense_cosine_scores

    ranked_lists, chunk_payloads, dense_cosine_scores = await timed(ctx, "retrieve", _retrieve())

    # 5. fuse
    async def _fuse():
        return reciprocal_rank_fusion(ranked_lists)

    fused = await timed(ctx, "fuse", _fuse())

    # 6. coverage gate — may short-circuit. Gates on LOCAL-model cosine
    # scores by default: the calibration in
    # bench/run_guardrails_calibration.py is per-embedding-space, and the
    # NVIDIA nemotron space shows poor in/out-of-domain separation (in-min
    # 0.248 vs out-max 0.270, measured) while MiniLM separates cleanly
    # (0.817 / 0.600). On the local path the retrieve-stage scores already
    # are local cosines; on the voice/NVIDIA path we normally re-score the
    # transcript with the local model (~25ms extra) UNLESS
    # settings.coverage_local_reembed is off — the memory-constrained
    # deployment path (settings.embedding_provider == "nvidia") sets this
    # so the local model is never loaded at all, accepting the coarser
    # nvidia-space calibration as a documented tradeoff. See api/config.py.
    async def _coverage():
        if vector_name == VECTOR_NAME:
            return coverage_verdict(dense_cosine_scores)
        if settings.coverage_local_reembed:
            local_vector = await asyncio.to_thread(embed_query, text)
            _, local_scores = await asyncio.to_thread(
                search_dense_grouped, local_vector, top_k, VECTOR_NAME
            )
            return coverage_verdict(local_scores)
        return coverage_verdict(
            dense_cosine_scores,
            tau_absolute=settings.nvidia_coverage_tau_absolute,
            tau_mean=settings.nvidia_coverage_tau_mean,
        )

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

    # 8. answer — extractive (fast, free, always available) or abstractive
    # (NVIDIA LLM, falls back to extractive on failure/timeout)
    effective_mode = answer_mode
    answer_lang = lang

    async def _answer():
        nonlocal effective_mode, answer_lang
        if effective_mode == "abstractive":
            try:
                # nvidia_llm_model is a reasoning model — observed live
                # latency for a short RAG answer is 8-25s wall-clock (see
                # api/llm/nvidia_client.py's agenerate_answer docstring), so
                # this slice is generous, not a typo. Capped by whatever the
                # caller's overall deadline actually affords.
                result = await generate_abstractive_answer(
                    text, assembled.blocks, deadline.child(min(deadline.remaining_ms, 30000))
                )
                answer_lang, _ = detect_language(result.text)
                return result
            except NvidiaCallError:
                ctx.degrade("abstractive_failed_fallback_extractive")
                effective_mode = "extractive"
        return select_span(text, assembled.blocks[0])

    answer_result = await timed(ctx, "answer", _answer())

    # 9. guard_out — may veto
    async def _guard_out():
        if effective_mode == "extractive":
            return run_guard_out(
                answer_text=answer_result.text,
                context=assembled.text,
                query=text,
                query_lang=lang,
                answer_lang=lang,  # extractive answers inherit the chunk's language
                cited_chunk_ids=[answer_result.chunk_id],
                supplied_chunk_ids=assembled.supplied_chunk_ids,
                cited_chunk_text=assembled.blocks[0].text,
                answer_mode="extractive",
            )
        return run_guard_out(
            answer_text=answer_result.text,
            context=assembled.text,
            query=text,
            query_lang=lang,
            answer_lang=answer_lang,
            cited_chunk_ids=answer_result.cited_chunk_ids,
            supplied_chunk_ids=assembled.supplied_chunk_ids,
            answer_mode="abstractive",
        )

    try:
        output_trace = await timed(ctx, "guard_out", _guard_out())
    except StageShortCircuit:
        if effective_mode != "abstractive":
            raise
        # The LLM occasionally produces a response ungrounded in the
        # supplied context even after a successful call — confirmed live:
        # a reasoning model, told "detailed thinking off", can still
        # occasionally drift into a generic/refusal-style answer that
        # shares no vocabulary with the real retrieved passages, and
        # guard_out correctly vetoes it. Rather than fail the whole
        # request over one bad generation, fall back to the extractive
        # answer — the same real retrieved context, already known-grounded
        # by construction — same as the NvidiaCallError fallback above.
        ctx.degrade("abstractive_ungrounded_fallback_extractive")
        effective_mode = "extractive"
        answer_lang = lang  # discard the failed abstractive attempt's detected language
        answer_result = select_span(text, assembled.blocks[0])
        output_trace = await timed(ctx, "guard_out", _guard_out())

    if effective_mode == "extractive":
        citations = [
            Citation(
                chunk_id=answer_result.chunk_id,
                score=fused[0].score if fused else 0.0,
                strategy=answer_result.strategy,
                span=answer_result.char_span,
            )
        ]
    else:
        blocks_by_id = {b.chunk_id: b for b in assembled.blocks}
        citations = [
            Citation(
                chunk_id=cid,
                score=blocks_by_id[cid].score if cid in blocks_by_id else 0.0,
                strategy=blocks_by_id[cid].strategy if cid in blocks_by_id else "unknown",
                span=None,
            )
            for cid in answer_result.cited_chunk_ids
        ]

    return AskResponse(
        trace_id=ctx.trace_id,
        verdict="ANSWERED",
        answer=Answer(text=answer_result.text, mode=effective_mode, language=answer_lang),
        citations=citations,
        guardrails=GuardrailTrace(
            input=InputGuardrailTrace(checks=checks_passed),
            output=output_trace,
            coverage=coverage_stats.model_dump(),
        ),
        timings_ms=ctx.timings_ms,
        degradations=ctx.degradations,
    )


def build_refusal_response(ctx: Context, short_circuit: StageShortCircuit) -> AskResponse:
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
