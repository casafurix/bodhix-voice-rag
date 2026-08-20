"""No SARVAM_API_KEY or network access is required to run this file — the
HTTP call is mocked via respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from api.harness.deadline import Deadline
from api.stt.sarvam import SARVAM_STT_URL, SarvamProvider


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_sends_expected_request_and_parses_response():
    route = respx.post(SARVAM_STT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "transcript": "what is the capital of india",
                "language_code": "en-IN",
                "language_probability": 0.97,
                "duration_s": 2.3,
            },
        )
    )

    provider = SarvamProvider(api_key="test-key")
    transcript = await provider.transcribe(b"fake-audio-bytes", "en-IN", Deadline(budget_ms=5000))

    assert route.called
    request = route.calls.last.request
    assert request.headers["api-subscription-key"] == "test-key"
    assert b'name="model"' in request.content
    assert b"saaras:v3" in request.content
    assert b'name="mode"' in request.content
    assert b"transcribe" in request.content

    assert transcript.text == "what is the capital of india"
    assert transcript.language == "en-IN"
    assert transcript.language_confidence == 0.97
    assert transcript.provider == "sarvam"
    assert transcript.duration_s == 2.3


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_defaults_language_code_to_unknown_without_hint():
    respx.post(SARVAM_STT_URL).mock(
        return_value=httpx.Response(200, json={"transcript": "hello"})
    )
    provider = SarvamProvider(api_key="test-key")
    await provider.transcribe(b"audio", None, Deadline(budget_ms=5000))

    request = respx.calls.last.request
    assert b"unknown" in request.content


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_raises_on_http_error():
    respx.post(SARVAM_STT_URL).mock(return_value=httpx.Response(500))
    provider = SarvamProvider(api_key="test-key")
    with pytest.raises(httpx.HTTPStatusError):
        await provider.transcribe(b"audio", "en-IN", Deadline(budget_ms=5000))


@pytest.mark.asyncio
async def test_stream_not_implemented():
    provider = SarvamProvider(api_key="test-key")

    async def _frames():
        yield b"chunk"

    # stream() has no `yield` in its body (see api/stt/sarvam.py) so it's a
    # plain coroutine, not an async generator — awaiting it (not iterating
    # it) is what raises.
    with pytest.raises(NotImplementedError):
        await provider.stream(_frames(), "en-IN")
