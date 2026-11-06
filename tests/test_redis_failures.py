
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.security import (
    SessionStore,
    SessionStoreUnavailable,
    create_jwt,
    store,
)


client = TestClient(app)


class FailingRedis:
    async def ping(self):
        raise ConnectionError("Simulated Redis outage")

    async def set(self, *args, **kwargs):
        raise ConnectionError("Simulated Redis outage")

    async def get(self, *args, **kwargs):
        raise ConnectionError("Simulated Redis outage")

    async def delete(self, *args, **kwargs):
        raise ConnectionError("Simulated Redis outage")

    def multi(self):
        raise ConnectionError("Simulated Redis outage")


class FakeTransaction:
    def __init__(self):
        self.commands = []

    def incr(self, key):
        self.commands.append(("incr", key))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    async def exec(self):
        return [7, True]


class FakeRedis:
    def __init__(self):
        self.transaction = FakeTransaction()

    def multi(self):
        return self.transaction


@pytest.mark.parametrize(
    "operation",
    ["set", "get", "delete", "incr"],
)
def test_redis_failure_does_not_fallback_to_memory(operation):
    redis = FailingRedis()
    session_store = SessionStore(redis_client=redis)

    async def run_operation():
        if operation == "set":
            await session_store.set("key", "value", 60)

        elif operation == "get":
            await session_store.get("key")

        elif operation == "delete":
            await session_store.delete("key")

        elif operation == "incr":
            await session_store.incr_with_ttl("counter", 60)

    with pytest.raises(SessionStoreUnavailable):
        asyncio.run(run_operation())

    assert session_store.backend == "redis"
    assert session_store._memory == {}


def test_redis_ping_reports_unavailable():
    session_store = SessionStore(
        redis_client=FailingRedis()
    )

    assert asyncio.run(session_store.ping()) is False


def test_readiness_returns_503_when_redis_is_down(monkeypatch):
    monkeypatch.setattr(
        store,
        "_redis",
        FailingRedis(),
    )

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Session store is unavailable"
    )


def test_session_endpoint_returns_503_when_redis_is_down(
    monkeypatch,
):
    monkeypatch.setattr(
        store,
        "_redis",
        FailingRedis(),
    )

    token = create_jwt(
        session_id="redis-outage-test",
        provider="openai",
        model_id="gpt-4o-mini",
        project="redis-outage-test",
        ttl=60,
    )

    response = client.get(
        "/v1/demo/session/status",
        headers={
            "Authorization": f"Bearer {token}"
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Session store is temporarily unavailable."
    )
    assert response.headers["retry-after"] == "5"


def test_rate_limit_increment_uses_atomic_transaction():
    redis = FakeRedis()
    session_store = SessionStore(
        redis_client=redis
    )

    result = asyncio.run(
        session_store.incr_with_ttl(
            "test-rate-limit",
            65,
        )
    )

    assert result == 7
    assert redis.transaction.commands == [
        ("incr", "test-rate-limit"),
        ("expire", "test-rate-limit", 65),
    ]


def test_memory_store_works_when_redis_is_not_configured(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "redis_url",
        "",
    )
    monkeypatch.setattr(
        settings,
        "redis_token",
        "",
    )

    session_store = SessionStore()

    assert session_store.backend == "memory"

    async def verify():
        await session_store.set(
            "development-key",
            "development-value",
            60,
        )

        return await session_store.get(
            "development-key"
        )

    assert asyncio.run(verify()) == "development-value"
