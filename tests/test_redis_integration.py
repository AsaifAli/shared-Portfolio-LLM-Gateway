
import asyncio
import os
import uuid

import pytest
from upstash_redis.asyncio import Redis

from app.security import SessionStore


@pytest.mark.skipif(
    not os.getenv("UPSTASH_REDIS_REST_URL")
    or not os.getenv("UPSTASH_REDIS_REST_TOKEN"),
    reason="Upstash credentials not configured",
)
def test_real_upstash_session_store():

    async def run():
        redis = Redis(
            url=os.environ["UPSTASH_REDIS_REST_URL"].strip(),
            token=os.environ["UPSTASH_REDIS_REST_TOKEN"].strip(),
            rest_retries=0,
            allow_telemetry=False,
        )

        session_store = SessionStore(redis_client=redis)

        prefix = f"gateway:integration:{uuid.uuid4().hex}"
        session_key = f"{prefix}:session"
        counter_key = f"{prefix}:counter"

        connected = False

        try:
            if not await session_store.ping():
                raise RuntimeError(
                    "Cannot reach Upstash Redis. "
                    "Check the REST endpoint and network DNS."
                )

            connected = True

            assert session_store.backend == "redis"

            await session_store.set(
                session_key,
                "integration-test-value",
                120,
            )

            assert await session_store.get(session_key) == (
                "integration-test-value"
            )

            first = await session_store.incr_with_ttl(
                counter_key, 120
            )
            second = await session_store.incr_with_ttl(
                counter_key, 120
            )

            assert first == 1
            assert second == 2

            await session_store.delete(session_key)

            assert await session_store.get(session_key) is None

        finally:
            if connected:
                try:
                    await redis.delete(session_key, counter_key)
                except Exception as exc:
                    print(
                        "Warning: Redis test-key cleanup failed:",
                        type(exc).__name__,
                    )

            await redis.close()

    asyncio.run(run())
