"""The frozen `/ask` contract — see docs/01-architecture.md.

This is deliberately the first file in the codebase: every stage in
harness/pipeline.py reads and writes these models, so freezing the shape
here is what lets ingestion, retrieval and guardrail work happen without
waiting on each other.

Scope note (MVP): no `rich_path` / abstractive answer, no reranker fields.
Both are cut per the descoping plan in docs/11-roadmap.md; the schema still
reserves space for them (commented) so re-adding is additive, not a rewrite.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RefusalCode = Literal[
    "NO_SPEECH",
    "UNINTELLIGIBLE_AUDIO",
    "MALFORMED_QUERY",
    "UNSUPPORTED_LANGUAGE",
    "UNSAFE_INPUT",
    "INJECTION_DETECTED",
    "OUT_OF_SCOPE",
    "LOW_CONFIDENCE",
    "UNGROUNDED_ANSWER",
    "BUDGET_EXCEEDED",
    "INTERNAL_ERROR",
]


class AskOptions(BaseModel):
    rerank: Literal["auto", "off"] = "off"  # MVP default: no reranker built yet


class AskRequest(BaseModel):
    query: str
    budget_ms: float = 200.0
    lang_hint: str | None = None
    options: AskOptions = Field(default_factory=AskOptions)


class Citation(BaseModel):
    chunk_id: str
    score: float
    strategy: str
    span: tuple[int, int]


class Answer(BaseModel):
    text: str
    mode: Literal["extractive"] = "extractive"  # only mode in MVP
    language: str


class InputGuardrailTrace(BaseModel):
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: str | None = None


class OutputGuardrailTrace(BaseModel):
    numeric_check: Literal["pass", "fail", "skipped"] = "skipped"
    citation_check: Literal["pass", "fail", "skipped"] = "skipped"
    language_match: bool | None = None
    groundedness_score: float | None = None
    groundedness_method: Literal["lexical_overlap", "nli", "skipped"] = "skipped"


class GuardrailTrace(BaseModel):
    input: InputGuardrailTrace = Field(default_factory=InputGuardrailTrace)
    output: OutputGuardrailTrace = Field(default_factory=OutputGuardrailTrace)
    coverage: dict[str, float] = Field(default_factory=dict)  # top1, mean5, margin, spread


class AskResponse(BaseModel):
    trace_id: str
    verdict: Literal["ANSWERED", "REFUSED"]
    refusal_code: RefusalCode | None = None
    answer: Answer | None = None
    citations: list[Citation] = Field(default_factory=list)
    guardrails: GuardrailTrace = Field(default_factory=GuardrailTrace)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    degradations: list[str] = Field(default_factory=list)
