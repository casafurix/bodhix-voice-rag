"""Voice entrypoint — batch STT then the shared retrieval/answer DAG.
See docs/13-build-status.md.

Batch, not streaming: the caller uploads a complete audio clip, we
transcribe the whole thing, then proceed exactly like a text query. No
WebSocket, no VAD, no partial transcripts — `api/stt/sarvam.py`'s
`stream()` stays unimplemented; that's a deliberate scope decision, not a
gap. Voice queries default to NVIDIA online embedding (searching the
`dense_nvidia` field, see api/retrieval/qdrant_store.py) and abstractive
answers, both with degradation to the local/extractive path built into
`run_retrieval_and_answer` if the NVIDIA API fails.
"""

from __future__ import annotations

from typing import Literal

from api.harness.context import Context
from api.harness.deadline import Deadline
from api.harness.pipeline import (
    build_refusal_response,
    normalise_and_guard,
    run_retrieval_and_answer,
)
from api.harness.stage import StageShortCircuit, timed
from api.schemas import AskResponse
from api.stt.sarvam import SarvamProvider


async def run_ask_voice(
    *,
    audio: bytes,
    lang_hint: str | None,
    budget_ms: float,
    answer_mode: Literal["extractive", "abstractive"] = "abstractive",
) -> AskResponse:
    deadline = Deadline(budget_ms=budget_ms)
    ctx = Context(deadline=deadline)

    try:
        async def _stt():
            provider = SarvamProvider()
            return await provider.transcribe(
                audio, lang_hint, deadline.child(min(deadline.remaining_ms, 4000))
            )

        transcript = await timed(ctx, "stt", _stt())

        if not transcript.text.strip():
            raise StageShortCircuit("NO_SPEECH", "empty transcript from STT")

        text, lang, _lang_conf, checks_passed = await normalise_and_guard(
            transcript.text, lang_hint or transcript.language, ctx
        )
        response = await run_retrieval_and_answer(
            text,
            lang,
            checks_passed,
            ctx,
            deadline,
            embedding_provider="nvidia",
            answer_mode=answer_mode,
        )
        response.transcript = transcript
        return response
    except StageShortCircuit as short_circuit:
        return build_refusal_response(ctx, short_circuit)
    except Exception as exc:
        return build_refusal_response(
            ctx, StageShortCircuit("INTERNAL_ERROR", f"Voice transcription failed: {exc}")
        )
