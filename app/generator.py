"""Real generator for rag-local-eval-loop (organizer-provided grading
harness — see wiring-in-the-eval-loop.pdf).

Wraps this project's ACTUAL production guardrail + answer logic, not a
reimplementation:

- api/guardrails/coverage_gate.py's calibrated coverage_verdict() decides
  whether the supplied context actually covers the query — the same
  TAU_ABSOLUTE/TAU_MEAN thresholds POST /ask uses. This is what drives
  `.grounded`, which the eval loop's reliability ("lying factor") check
  grades directly.
- api/answer/extractive.py's select_span() picks the answer sentence when
  grounded — the same lexical-overlap span picker /ask's extractive path
  uses, verbatim substring of real context, no LLM in the loop.

Why extractive, not abstractive: eval/pipeline.py calls generate_answer()
once per sampled example, concurrently across --workers threads. The
abstractive path (api/answer/abstractive.py) is a NVIDIA NIM call that
takes 8-25s per answer by its own docstring's measurement — multiplying
that by --num-answerable/--num-unanswerable would make a real run
impractically slow. Extractive is BodhiX's default answer_mode in
production (api/schemas.py); this exercises the exact code path a plain
POST /ask hits.

Coverage-gate note: the eval loop's own throwaway FAISS index scores hits
with raw (unnormalised) inner product (see its eval/index_build.py) —
confirmed by measurement that this project's embed_query() output has
norm ~5.47, not 1.0, so that raw score is NOT on the same scale
coverage_gate.py was calibrated against (Qdrant's COSINE distance, which
normalises internally). Re-embedding candidates and computing true cosine
here — rather than trusting the eval loop's own .score field — keeps the
grounded/ungrounded decision comparable to what POST /ask actually decides.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from api.answer.extractive import select_span
from api.guardrails.coverage_gate import coverage_verdict
from api.harness.stage import StageShortCircuit
from api.retrieval.assemble import AssembledChunk
from api.retrieval.embed import embed_passages, embed_query

MODEL_LABEL = "bodhix-extractive (MiniLM cosine coverage-gate + lexical-overlap span)"
_DECLINE_TEXT = "I couldn't find enough relevant information in my sources to answer that."


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) or 1e-9
    return float(np.dot(a, b) / denom)


def generate_answer(query: str, results: list) -> GeneratedAnswer:
    t0 = time.perf_counter()

    if not results:
        return GeneratedAnswer(
            text=_DECLINE_TEXT, grounded=False,
            generation_ms=(time.perf_counter() - t0) * 1000.0, model=MODEL_LABEL,
        )

    query_vec = np.array(embed_query(query))
    result_vecs = [np.array(v) for v in embed_passages([r.text for r in results])]
    ranked = sorted(
        zip(results, result_vecs), key=lambda pair: _cosine(query_vec, pair[1]), reverse=True
    )
    scores = [_cosine(query_vec, vec) for _, vec in ranked]

    try:
        coverage_verdict(scores)  # raises StageShortCircuit(OUT_OF_SCOPE|LOW_CONFIDENCE)
    except StageShortCircuit:
        return GeneratedAnswer(
            text=_DECLINE_TEXT, grounded=False,
            generation_ms=(time.perf_counter() - t0) * 1000.0, model=MODEL_LABEL,
        )

    top_result, top_vec = ranked[0]
    chunk = AssembledChunk(
        chunk_id=top_result.source,
        parent_id=top_result.source,
        text=top_result.text,
        strategy="eval_external",
        score=_cosine(query_vec, top_vec),
        language="unknown",
    )
    answer = select_span(query, chunk)

    return GeneratedAnswer(
        text=answer.text, grounded=True,
        generation_ms=(time.perf_counter() - t0) * 1000.0, model=MODEL_LABEL,
    )
