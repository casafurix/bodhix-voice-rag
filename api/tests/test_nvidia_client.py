"""No NVIDIA_API_KEY is required to run this file — every client call is
mocked, matching the constraint that `pytest` must never need real
credentials or network access.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from api.harness.deadline import Deadline
from api.llm import nvidia_client


def _fake_embedding_response(vec):
    return SimpleNamespace(data=[SimpleNamespace(embedding=vec)])


def _fake_chat_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.asyncio
async def test_aembed_query_shapes_request_and_parses_response(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=_fake_embedding_response([0.1, 0.2, 0.3]))
    monkeypatch.setattr(nvidia_client, "_async_client", lambda: mock_client)

    result = await nvidia_client.aembed_query("hello", Deadline(budget_ms=5000))

    assert result == [0.1, 0.2, 0.3]
    _, kwargs = mock_client.embeddings.create.call_args
    assert kwargs["model"] == nvidia_client.settings.nvidia_embed_model
    assert kwargs["input"] == ["hello"]


@pytest.mark.asyncio
async def test_aembed_query_wraps_openai_error_as_nvidia_call_error(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=openai.APIConnectionError(request=MagicMock())
    )
    monkeypatch.setattr(nvidia_client, "_async_client", lambda: mock_client)

    with pytest.raises(nvidia_client.NvidiaCallError):
        await nvidia_client.aembed_query("hello", Deadline(budget_ms=5000))


@pytest.mark.asyncio
async def test_agenerate_answer_shapes_request_and_parses_response(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_fake_chat_response("The answer.")
    )
    monkeypatch.setattr(nvidia_client, "_async_client", lambda: mock_client)

    result = await nvidia_client.agenerate_answer(
        [{"role": "user", "content": "hi"}], Deadline(budget_ms=5000)
    )

    assert result == "The answer."
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == nvidia_client.settings.nvidia_llm_model
    assert kwargs["stream"] is False


@pytest.mark.asyncio
async def test_agenerate_answer_wraps_errors(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(nvidia_client, "_async_client", lambda: mock_client)

    with pytest.raises(nvidia_client.NvidiaCallError):
        await nvidia_client.agenerate_answer([{"role": "user", "content": "hi"}], Deadline(budget_ms=5000))


def test_embed_passages_online_batches_and_parses(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _fake_embedding_response([[0.1], [0.2]])
    # Return per-batch responses shaped like the real SDK: one `.data` item per text.
    mock_client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[float(i)]) for i in range(len(kwargs["input"]))]
    )
    monkeypatch.setattr(nvidia_client, "_sync_client", lambda: mock_client)

    result = nvidia_client.embed_passages_online(["a", "b", "c"], batch_size=2)

    assert result == [[0.0], [1.0], [0.0]]  # two batches: ["a","b"] then ["c"]
    assert mock_client.embeddings.create.call_count == 2


def test_embed_passages_online_retries_then_succeeds(monkeypatch):
    mock_client = MagicMock()
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise openai.APIConnectionError(request=MagicMock())
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0]) for _ in kwargs["input"]])

    mock_client.embeddings.create.side_effect = flaky
    monkeypatch.setattr(nvidia_client, "_sync_client", lambda: mock_client)
    monkeypatch.setattr(nvidia_client.time, "sleep", lambda _s: None)

    result = nvidia_client.embed_passages_online(["a"], batch_size=16)

    assert result == [[1.0]]
    assert calls["n"] == 2


def test_embed_passages_online_raises_after_exhausting_retries(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = openai.RateLimitError(
        message="rate limited", response=MagicMock(status_code=429, headers={}), body=None
    )
    monkeypatch.setattr(nvidia_client, "_sync_client", lambda: mock_client)
    monkeypatch.setattr(nvidia_client.time, "sleep", lambda _s: None)

    with pytest.raises(nvidia_client.NvidiaCallError):
        nvidia_client.embed_passages_online(["a"], batch_size=16)
