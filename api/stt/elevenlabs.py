"""ElevenLabs STT adapter — STUB ONLY, not wired into the pipeline.

Implements SttProvider so it can be dropped in as a failover later with no
interface change, per docs/05-speech-to-text.md. Deliberately not imported
by harness/pipeline.py — see the ElevenLabs-scope decision in the chat log
(stub the interface, don't wire it, for the solo-build MVP).
"""

from __future__ import annotations

from typing import AsyncIterator

from api.config import settings
from api.harness.deadline import Deadline
from api.stt.base import PartialTranscript, Transcript

ELEVENLABS_STT_URL = "https://api.in.residency.elevenlabs.io/v1/speech-to-text"


class ElevenLabsProvider:
    name = "elevenlabs"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.elevenlabs_api_key

    async def transcribe(
        self, audio: bytes, lang_hint: str | None, deadline: Deadline
    ) -> Transcript:
        raise NotImplementedError(
            "ElevenLabs adapter is stubbed, not wired. Activate only if a "
            "demo-resilience failover is needed later; see docs/05-speech-to-text.md."
        )

    async def stream(
        self, frames: AsyncIterator[bytes], lang_hint: str | None
    ) -> AsyncIterator[PartialTranscript]:
        raise NotImplementedError("ElevenLabs adapter is stubbed, not wired.")
