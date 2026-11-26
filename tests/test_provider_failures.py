import asyncio
from types import SimpleNamespace

import pytest

from app import llm
from app.config import ModelConfig
from app.llm import (
    ProviderAuthenticationError,
    ProviderBillingError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    complete,
    prepare_stream,
)


OPENROUTER_MODEL = ModelConfig(
    id="openrouter-test",
    provider="openrouter",
    model_id="openrouter/openai/gpt-oss-20b:free",
)

REQUEST = {
    "messages": [{"role": "user", "content": "Test"}],
}


class FakeStatusError(Exception):
    def __init__(self, status_code, message="provider response sk-secret-123", headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = type("Response", (), {"headers": headers or {}})()


class FakeClient:
    def __init__(self, create):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
        self.closed = False

    async def close(self):
        self.closed = True


def test_provider_error_classification_is_sanitized():
    error = llm._to_provider_error(
        FakeStatusError(401, "invalid key sk-secret-123")
    )
    assert isinstance(error, ProviderAuthenticationError)
    assert error.status_code == 401
    assert "sk-secret-123" not in str(error)


def test_rate_limit_preserves_retry_after_without_leaking_body():
    error = llm._to_provider_error(
        FakeStatusError(
            429,
            "rate limited sk-secret-123",
            {"retry-after": "12"},
        )
    )
    assert isinstance(error, ProviderRateLimitError)
    assert error.status_code == 429
    assert error.retry_after == "12"
    assert "sk-secret-123" not in str(error)


def test_5xx_maps_to_provider_unavailable():
    error = llm._to_provider_error(FakeStatusError(503))
    assert isinstance(error, ProviderUnavailableError)
    assert error.status_code == 503


def test_connection_error_maps_to_safe_connection_error():
    error = llm._to_provider_error(ConnectionError("secret connection details"))
    assert isinstance(error, ProviderConnectionError)
    assert "secret connection details" not in str(error)


def test_provider_model_prefix_is_removed_once():
    assert llm._provider_model_id(OPENROUTER_MODEL) == "openai/gpt-oss-20b:free"
    google = ModelConfig("g", "google", "gemini/gemini-3.5-flash")
    assert llm._provider_model_id(google) == "gemini-3.5-flash"


def test_provider_endpoints_are_explicit(monkeypatch):
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://portfolio.example")
    monkeypatch.setenv("OPENROUTER_X_TITLE", "Portfolio demo")
    assert llm._openai_endpoint("openai") == ("https://api.openai.com/v1", {})
    assert llm._openai_endpoint("openrouter") == (
        "https://openrouter.ai/api/v1",
        {"HTTP-Referer": "https://portfolio.example", "X-Title": "Portfolio demo"},
    )
    assert llm._openai_endpoint("anthropic")[0] == "https://api.anthropic.com/v1/"


def test_complete_retries_connection_failure_before_returning(monkeypatch):
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("network details must not leak")
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": kwargs["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "recovered"},
                "finish_reason": "stop",
            }],
        }

    fake_client = FakeClient(create)
    monkeypatch.setattr(llm, "_openai_client", lambda *_args: fake_client)
    monkeypatch.setattr(llm, "MAX_PROVIDER_RETRIES", 1)
    monkeypatch.setattr(llm.asyncio, "sleep", lambda _seconds: _no_sleep())

    result = asyncio.run(complete(OPENROUTER_MODEL, "dummy-key", REQUEST))
    assert result["choices"][0]["message"]["content"] == "recovered"
    assert result["model"] == "openai/gpt-oss-20b:free"
    assert calls == 2
    assert fake_client.closed is True


def test_stream_preflight_auth_failure_returns_before_sse(monkeypatch):
    async def create(**_kwargs):
        raise FakeStatusError(401)

    fake_client = FakeClient(create)
    monkeypatch.setattr(llm, "_openai_client", lambda *_args: fake_client)
    monkeypatch.setattr(llm, "MAX_PROVIDER_RETRIES", 0)

    with pytest.raises(ProviderAuthenticationError):
        asyncio.run(prepare_stream(OPENROUTER_MODEL, "dummy-key", REQUEST))

    assert fake_client.closed is True


def test_stream_is_not_replayed_after_first_chunk(monkeypatch):
    calls = 0

    class FakeStream:
        def __init__(self):
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            if self.index == 1:
                return {"id": "first", "choices": [{"delta": {"content": "hello"}}]}
            raise FakeStatusError(503)

        async def close(self):
            return None

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        return FakeStream()

    fake_client = FakeClient(create)
    monkeypatch.setattr(llm, "_openai_client", lambda *_args: fake_client)
    monkeypatch.setattr(llm, "MAX_PROVIDER_RETRIES", 2)

    async def collect():
        events = await prepare_stream(
            OPENROUTER_MODEL,
            "dummy-key",
            {**REQUEST, "stream": True},
        )
        return "".join([event async for event in events])

    output = asyncio.run(collect())
    assert calls == 1
    assert '"content":"hello"' in output
    assert "provider_unavailable_error" in output
    assert "[DONE]" in output
    assert "sk-secret-123" not in output


async def _no_sleep():
    return None


def test_anthropic_overloaded_529_is_transient():
    error = llm._to_provider_error(FakeStatusError(529))
    assert isinstance(error, ProviderUnavailableError)
    assert error.status_code == 503
    assert 529 in llm.RETRYABLE_STATUS_CODES


def test_openrouter_insufficient_credits_is_billing_error():
    error = llm._to_provider_error(FakeStatusError(402))
    assert isinstance(error, ProviderBillingError)
    assert error.status_code == 402


def test_stream_closes_failed_first_chunk_before_retry(monkeypatch):
    calls = 0
    first_stream = None

    class FirstBrokenStream:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionError("temporary first-chunk failure")

        async def aclose(self):
            self.closed = True

    class GoodStream:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return {"id": "recovered", "choices": [{"delta": {"content": "ok"}}]}

        async def aclose(self):
            return None

    async def create(**_kwargs):
        nonlocal calls, first_stream
        calls += 1
        if calls == 1:
            first_stream = FirstBrokenStream()
            return first_stream
        return GoodStream()

    fake_client = FakeClient(create)
    monkeypatch.setattr(llm, "_openai_client", lambda *_args: fake_client)
    monkeypatch.setattr(llm, "MAX_PROVIDER_RETRIES", 1)
    monkeypatch.setattr(llm.asyncio, "sleep", lambda _seconds: _no_sleep())

    async def collect():
        events = await prepare_stream(
            OPENROUTER_MODEL,
            "dummy-key",
            {**REQUEST, "stream": True},
        )
        return "".join([event async for event in events])

    output = asyncio.run(collect())
    assert calls == 2
    assert first_stream is not None and first_stream.closed
    assert '"content":"ok"' in output
    assert "[DONE]" in output
