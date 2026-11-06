
import asyncio

import pytest

from app import llm
from app.config import ModelConfig
from app.llm import (
    complete,
    stream,
    ProviderError,
    ProviderRateLimitError,
)


MODEL = ModelConfig(
    id="test-model",
    provider="openai",
    litellm_model="openai/gpt-4o-mini",
)

REQUEST = {
    "messages": [
        {"role": "user", "content": "Test"}
    ]
}


def test_provider_error_is_sanitized(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise RuntimeError(
            "Sensitive provider response sk-secret-123"
        )

    monkeypatch.setattr(
        llm.litellm,
        "acompletion",
        fake_acompletion,
    )

    with pytest.raises(ProviderError) as exc:
        asyncio.run(
            complete(MODEL, "dummy-key", REQUEST)
        )

    assert "sk-secret-123" not in str(exc.value)


def test_streaming_error_is_sanitized(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise RuntimeError(
            "Sensitive provider response sk-secret-123"
        )

    monkeypatch.setattr(
        llm.litellm,
        "acompletion",
        fake_acompletion,
    )

    async def collect_stream():
        chunks = []

        async for chunk in stream(
            MODEL,
            "dummy-key",
            {**REQUEST, "stream": True},
        ):
            chunks.append(chunk)

        return "".join(chunks)

    output = asyncio.run(collect_stream())

    assert "sk-secret-123" not in output
    assert "provider_error" in output
    assert "[DONE]" in output


def test_provider_rate_limit_is_classified(monkeypatch):
    class RateLimitError(Exception):
        status_code = 429
        retry_after = 12

    async def fake_acompletion(**kwargs):
        raise RateLimitError("Provider rate limited")

    monkeypatch.setattr(
        llm.litellm,
        "acompletion",
        fake_acompletion,
    )

    with pytest.raises(ProviderRateLimitError) as exc:
        asyncio.run(
            complete(MODEL, "dummy-key", REQUEST)
        )

    assert exc.value.retry_after == "12"
