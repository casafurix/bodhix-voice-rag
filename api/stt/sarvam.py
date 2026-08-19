"""Sarvam STT adapter — the only live provider in the MVP.
See docs/05-speech-to-text.md.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

import httpx

from api.config import settings
from api.harness.deadline import Deadline
from api.stt.base import PartialTranscript, Transcript

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamProvider:
    name = "sarvam"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.sarvam_api_key

    async def transcribe(
        self, audio: bytes, lang_hint: str | None, deadline: Deadline
    ) -> Transcript:
        start = time.perf_counter()
        timeout_s = max(deadline.remaining_ms, 500) / 1000.0
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": self.api_key},
                files={"file": ("audio.wav", audio, "audio/wav")},
                data={
                    "model": "saaras:v3",
                    "mode": "transcribe",
                    "language_code": lang_hint or "unknown",
                },
            )
            resp.raise_for_status()
            body = resp.json()

        latency_ms = (time.perf_counter() - start) * 1000.0
        return Transcript(
            text=body.get("transcript", ""),
            language=body.get("language_code", lang_hint or "unknown"),
            language_confidence=body.get("language_probability"),
            translated_text=None,
            provider=self.name,
            duration_s=body.get("duration_s", 0.0),
            latency_ms=latency_ms,
        )

    async def stream(
        self, frames: AsyncIterator[bytes], lang_hint: str | None
    ) -> AsyncIterator[PartialTranscript]:
        # TODO: Sarvam realtime WebSocket streaming — deferred, see
        # docs/11-roadmap.md descoping order (speculative retrieval cut).
        # MVP uses `transcribe()` on the full utterance only.
        raise NotImplementedError("Streaming STT is not implemented in the MVP")
