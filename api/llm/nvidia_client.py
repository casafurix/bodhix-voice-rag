"""NVIDIA NIM client — OpenAI-compatible API. See docs/13-build-status.md.

Two roles:
  - the async, deadline-aware request-path functions (`aembed_query`,
    `agenerate_answer`) used live by harness/pipeline.py for voice queries
    and abstractive answers;
  - the sync, batch functions (`embed_passages_online`) used only by
    ingest/build_index.py, an offline CLI job with no per-request deadline.

Every call is wrapped so a raw `openai` exception never escapes this module
— callers only ever see `NvidiaCallError`, which the harness's degradation
ladder (ctx.degrade(...)) can catch and fall back on (local embedding,
extractive answer) without knowing anything about the NVIDIA SDK.

MODEL NOTE: verified live against the real API — `nvidia/nemotron-3-embed-1b`
returns 2048-dim vectors and does not require `extra_body={"input_type":...}`
(confirmed with a plain `embeddings.create(model=..., input=[...])` call, no
extra_body). The `input_type` param is still threaded through and left as an
opt-in no-op (`None` by default) in case a future model swap needs it.
"""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Literal

import openai
from openai import AsyncOpenAI, OpenAI

from api.config import settings
from api.harness.deadline import Deadline


class NvidiaCallError(Exception):
    """Raised for any NVIDIA API failure — network, timeout, HTTP error.
    Callers should treat this as "provider unavailable", not a bug.
    """


@lru_cache
def _async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)


@lru_cache
def _sync_client() -> OpenAI:
    return OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)


async def aembed_query(
    text: str,
    deadline: Deadline,
    input_type: Literal["query", "passage"] | None = None,
) -> list[float]:
    timeout_s = max(deadline.remaining_ms, 500) / 1000.0
    extra_body = {"input_type": input_type} if input_type else None
    try:
        response = await asyncio.wait_for(
            _async_client().embeddings.create(
                model=settings.nvidia_embed_model,
                input=[text],
                extra_body=extra_body,
            ),
            timeout=timeout_s,
        )
    except Exception as exc:
        raise NvidiaCallError(f"NVIDIA embedding call failed: {exc}") from exc
    return response.data[0].embedding


async def agenerate_answer(
    messages: list[dict],
    deadline: Deadline,
    max_tokens: int = 900,
    temperature: float = 0.2,
) -> str:
    """900 tokens, not 300: `nvidia_llm_model` is a reasoning model that
    spends real completion tokens on a `reasoning_content` chain-of-thought
    before the visible `content` — confirmed live, 300 tokens was too
    tight and produced `finish_reason="length"` with empty `content`. The
    system prompt should prepend "detailed thinking off" (NVIDIA's
    documented convention for this model family) to shorten that phase —
    see api/answer/abstractive.py. Even so, expect several seconds of
    real latency; the timeout here is deadline-driven, not fixed, so a
    generous caller-supplied deadline (see pipeline.py's abstractive-answer
    deadline.child(...) cap) matters more than this default.
    """
    timeout_s = max(deadline.remaining_ms, 500) / 1000.0
    try:
        response = await asyncio.wait_for(
            _async_client().chat.completions.create(
                model=settings.nvidia_llm_model,
                messages=messages,
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
                stream=False,
            ),
            timeout=timeout_s,
        )
    except Exception as exc:
        raise NvidiaCallError(f"NVIDIA chat completion call failed: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        finish_reason = response.choices[0].finish_reason
        raise NvidiaCallError(
            f"NVIDIA chat completion returned no content (finish_reason={finish_reason!r}) "
            "— likely spent its whole token budget on reasoning"
        )
    return content


_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def embed_passages_online(
    texts: list[str],
    batch_size: int = 16,
    input_type: Literal["query", "passage"] | None = "passage",
) -> list[list[float]]:
    """Sync, offline batch embedding for ingest. Retries transient failures
    with jittered backoff; not deadline-bound (this is a CLI job, not a
    live request).
    """
    client = _sync_client()
    extra_body = {"input_type": input_type} if input_type else None
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        last_exc: Exception | None = None
        for attempt, backoff_s in enumerate((0, 1, 2, 4)):
            if backoff_s:
                time.sleep(backoff_s)
            try:
                response = client.embeddings.create(
                    model=settings.nvidia_embed_model,
                    input=batch,
                    extra_body=extra_body,
                )
                vectors.extend(item.embedding for item in response.data)
                last_exc = None
                break
            except _RETRYABLE as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise NvidiaCallError(
                f"NVIDIA embedding batch failed after retries: {last_exc}"
            ) from last_exc

    return vectors
