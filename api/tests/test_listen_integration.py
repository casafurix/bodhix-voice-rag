"""End-to-end /listen test — real (tiny) Qdrant + BM25 index (with the
deterministic fake-NVIDIA vectors from conftest.py), Sarvam and the NVIDIA
API both mocked so no network access or real credentials are needed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.answer.abstractive import AbstractiveAnswer
from api.guardrails import coverage_gate
from api.harness import pipeline
from api.main import app
from api.stt.base import Transcript
from api.stt.sarvam import SarvamProvider
from api.tests.conftest import fake_nvidia_embed

EVEREST_QUESTION = "How tall is Mount Everest?"
FAKE_ANSWER_TEXT = (
    "Mount Everest, located in the Himalayas, is the tallest mountain in the "
    "world with a peak elevation of 8849 metres above sea level."
)


async def _fake_transcribe(self, audio, lang_hint, deadline):
    return Transcript(
        text=EVEREST_QUESTION, language="en", provider="sarvam", duration_s=1.5, latency_ms=10.0
    )


async def _fake_aembed_query(text, deadline, input_type=None):
    return fake_nvidia_embed(text)


async def _fake_generate_abstractive_answer(query, blocks, deadline):
    return AbstractiveAnswer(text=FAKE_ANSWER_TEXT, cited_chunk_ids=[blocks[0].chunk_id])


def test_listen_returns_abstractive_grounded_answer(tiny_index, monkeypatch):
    # See the matching comment in test_ask_integration.py — relaxed only for
    # this test, to isolate voice-path wiring from coverage-gate calibration
    # (a documented, out-of-scope gap at this fixture's tiny scale).
    monkeypatch.setattr(coverage_gate, "TAU_MARGIN", 0.0)
    monkeypatch.setattr(coverage_gate, "TAU_SPREAD", 0.0)
    monkeypatch.setattr(SarvamProvider, "transcribe", _fake_transcribe)
    monkeypatch.setattr(pipeline, "aembed_query", _fake_aembed_query)
    monkeypatch.setattr(pipeline, "generate_abstractive_answer", _fake_generate_abstractive_answer)

    client = TestClient(app)
    resp = client.post(
        "/listen",
        files={"audio": ("clip.wav", b"fake-audio-bytes", "audio/wav")},
        data={"lang_hint": "en", "budget_ms": "8000", "answer_mode": "abstractive"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "ANSWERED"
    assert body["answer"]["mode"] == "abstractive"
    assert body["answer"]["text"] == FAKE_ANSWER_TEXT
    assert body["transcript"]["text"] == EVEREST_QUESTION
    assert body["citations"]
    assert body["citations"][0]["span"] is None  # abstractive citations are whole-chunk


def test_listen_empty_transcript_refuses_no_speech(tiny_index, monkeypatch):
    async def _empty_transcribe(self, audio, lang_hint, deadline):
        return Transcript(text="", language="en", provider="sarvam", duration_s=0.1, latency_ms=5.0)

    monkeypatch.setattr(SarvamProvider, "transcribe", _empty_transcribe)

    client = TestClient(app)
    resp = client.post(
        "/listen",
        files={"audio": ("silence.wav", b"fake-audio-bytes", "audio/wav")},
        data={"lang_hint": "en"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "REFUSED"
    assert body["refusal_code"] == "NO_SPEECH"
