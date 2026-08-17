"""SttProvider protocol — see docs/05-speech-to-text.md.

One interface, adapters plug in behind it. Only `sarvam.py` is wired into
the pipeline in the MVP; `elevenlabs.py` implements this protocol but is not
imported anywhere yet (see docs/11-roadmap.md descoping).
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from pydantic import BaseModel

from api.harness.deadline import Deadline


class Transcript(BaseModel):
    text: str
    language: str
    language_confidence: float | None = None
    translated_text: str | None = None  # Sarvam translate mode
    provider: str
    duration_s: float
    latency_ms: float


class PartialTranscript(BaseModel):
    text: str
    is_stable: bool


class SttProvider(Protocol):
    name: str

    async def transcribe(
        self, audio: bytes, lang_hint: str | None, deadline: Deadline
    ) -> Transcript: ...

    async def stream(
        self, frames: AsyncIterator[bytes], lang_hint: str | None
    ) -> AsyncIterator[PartialTranscript]: ...
