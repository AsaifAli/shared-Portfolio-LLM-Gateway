import asyncio

import litellm
import pytest
from fastapi.testclient import TestClient

from app.config import ModelConfig, settings
from app.llm import (
    ProviderError,
    _build_kwargs,
    complete,
)
from app.main import (
    RequestSizeLimitMiddleware,
    app,
)


client = TestClient(app)


def test_default_output_token_budget_is_enforced(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "max_output_tokens",
        512,
    )

    model = ModelConfig(
        id="test-model",
        provider="openai",
        litellm_model="openai/gpt-4o-mini",
    )

    kwargs = _build_kwargs(
        model,
        "test-api-key",
        {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        },
    )

    assert kwargs["max_tokens"] == 512


def test_max_completion_tokens_are_capped(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "max_output_tokens",
        256,
    )

    model = ModelConfig(
        id="test-model",
        provider="openai",
        litellm_model="openai/gpt-4o-mini",
    )

    kwargs = _build_kwargs(
        model,
        "test-api-key",
        {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100,
            "max_completion_tokens": 1000,
        },
    )

    assert kwargs["max_completion_tokens"] == 256
    assert "max_tokens" not in kwargs


def test_provider_exception_is_sanitized(
    monkeypatch,
):
    async def fake_acompletion(**kwargs):
        raise RuntimeError(
            "Sensitive upstream detail: sk-secret-example"
        )

    monkeypatch.setattr(
        litellm,
        "acompletion",
        fake_acompletion,
    )

    model = ModelConfig(
        id="test-model",
        provider="openai",
        litellm_model="openai/gpt-4o-mini",
    )

    with pytest.raises(ProviderError) as error:
        asyncio.run(
            complete(
                model,
                "test-api-key",
                {
                    "messages": [
                        {"role": "user", "content": "Hello"}
                    ]
                },
            )
        )

    assert "sk-secret-example" not in str(error.value)


def test_request_size_middleware_rejects_content_length():
    app_called = False
    sent_messages = []

    async def downstream(scope, receive, send):
        nonlocal app_called
        app_called = True

    async def receive():
        raise AssertionError(
            "Oversized Content-Length should be rejected before reading"
        )

    async def send(message):
        sent_messages.append(message)

    middleware = RequestSizeLimitMiddleware(
        downstream,
        max_bytes=5,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [
            (b"content-length", b"6"),
        ],
    }

    asyncio.run(
        middleware(scope, receive, send)
    )

    assert app_called is False
    assert sent_messages[0]["status"] == 413


def test_request_size_middleware_rejects_chunked_body():
    app_called = False
    sent_messages = []

    async def downstream(scope, receive, send):
        nonlocal app_called
        app_called = True

    incoming = [
        {
            "type": "http.request",
            "body": b"1234",
            "more_body": True,
        },
        {
            "type": "http.request",
            "body": b"5678",
            "more_body": False,
        },
    ]

    async def receive():
        return incoming.pop(0)

    async def send(message):
        sent_messages.append(message)

    middleware = RequestSizeLimitMiddleware(
        downstream,
        max_bytes=5,
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
    }

    asyncio.run(
        middleware(scope, receive, send)
    )

    assert app_called is False
    assert sent_messages[0]["status"] == 413


def test_session_ttl_is_capped_by_server_setting(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "session_ttl_seconds",
        600,
    )

    response = client.post(
        "/v1/demo/session",
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "gateway-test",
            "ttl_seconds": 3600,
        },
    )

    assert response.status_code == 200
    assert response.json()["expires_in"] == 600


def test_readiness_endpoint():
    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_session_status_requires_bearer_token():
    response = client.get(
        "/v1/demo/session/status"
    )

    assert response.status_code == 401