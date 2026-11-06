
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from typing import Any

import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, Request

from .config import settings


logger = logging.getLogger(__name__)


class SessionStoreUnavailable(RuntimeError):
    """Raised when the configured session backend cannot serve a request."""


class SessionStore:
    """
    Redis-backed session and rate-limit store.

    In-memory storage is used only when Redis is not configured.
    If Redis is configured but unavailable, operations fail closed.
    """

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client
        self._memory: dict[str, tuple[float, str]] = {}

        if self._redis is not None:
            return

        if settings.redis_url and settings.redis_token:
            try:
                from upstash_redis.asyncio import Redis

                self._redis = Redis(
                    url=settings.redis_url,
                    token=settings.redis_token,
                    rest_retries=1,
                    rest_retry_interval=1,
                    allow_telemetry=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize Upstash Redis client"
                ) from exc

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def _redis_failure(
        self,
        operation: str,
        exc: Exception,
    ) -> SessionStoreUnavailable:
        logger.warning(
            "Session store operation failed: operation=%s exception_type=%s",
            operation,
            type(exc).__name__,
        )
        return SessionStoreUnavailable(
            "Session store is temporarily unavailable."
        )

    async def ping(self) -> bool:
        """
        Check the configured Redis backend.

        Memory mode is considered ready for local development only.
        Production startup separately requires Redis configuration.
        """
        if self._redis is None:
            return True

        try:
            result = await self._redis.ping()
            return bool(result)
        except Exception as exc:
            logger.warning(
                "Session store readiness check failed: exception_type=%s",
                type(exc).__name__,
            )
            return False

    async def close(self) -> None:
        """Close the configured Redis client's HTTP resources."""
        if self._redis is None:
            return

        close = getattr(self._redis, "close", None)

        if close is None:
            return

        try:
            result = close()

            if hasattr(result, "__await__"):
                await result

        except Exception as exc:
            logger.warning(
                "Session store close failed: exception_type=%s",
                type(exc).__name__,
            )

    def _purge_expired_memory_entries(self) -> None:
        now = time.monotonic()

        expired = [
            key
            for key, (expiry, _) in self._memory.items()
            if now >= expiry
        ]

        for key in expired:
            self._memory.pop(key, None)

    async def set(
        self,
        key: str,
        value: str,
        ttl: int,
    ) -> None:
        if ttl <= 0:
            raise ValueError("TTL must be positive")

        if self._redis is not None:
            try:
                await self._redis.set(
                    key,
                    value,
                    ex=ttl,
                )
                return
            except Exception as exc:
                raise self._redis_failure("set", exc) from exc

        # Development-only in-memory storage.
        if len(self._memory) > 10000:
            self._purge_expired_memory_entries()

        self._memory[key] = (
            time.monotonic() + ttl,
            value,
        )

    async def get(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                value = await self._redis.get(key)
                return str(value) if value is not None else None
            except Exception as exc:
                raise self._redis_failure("get", exc) from exc

        item = self._memory.get(key)

        if not item:
            return None

        expiry, value = item

        if time.monotonic() >= expiry:
            self._memory.pop(key, None)
            return None

        return value

    async def delete(self, key: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.delete(key)
                return
            except Exception as exc:
                raise self._redis_failure("delete", exc) from exc

        self._memory.pop(key, None)

    async def incr_with_ttl(
        self,
        key: str,
        ttl: int,
    ) -> int:
        if ttl <= 0:
            raise ValueError("TTL must be positive")

        if self._redis is not None:
            try:
                # MULTI/EXEC ensures INCR and EXPIRE execute atomically.
                # This prevents a counter from being created without
                # its expiry if requests arrive concurrently.
                transaction = self._redis.multi()

                transaction.incr(key)
                transaction.expire(key, ttl)

                results = await transaction.exec()

                if not results or results[0] is None:
                    raise RuntimeError(
                        "Redis rate-limit transaction returned no counter"
                    )

                return int(results[0])

            except Exception as exc:
                raise self._redis_failure(
                    "incr_with_ttl",
                    exc,
                ) from exc

        # Development-only in-memory counter.
        now = time.monotonic()
        item = self._memory.get(key)

        if not item or now >= item[0]:
            self._memory[key] = (
                now + ttl,
                "1",
            )
            return 1

        count = int(item[1]) + 1

        self._memory[key] = (
            item[0],
            str(count),
        )

        return count


store = SessionStore()


def get_fernet() -> Fernet:
    if settings.encryption_key:
        try:
            return Fernet(settings.encryption_key.encode())
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "GATEWAY_ENCRYPTION_KEY is invalid"
            ) from exc

    # Development-only fallback. Production requires an explicit key.
    return Fernet(Fernet.generate_key())


FERNET = get_fernet()


def new_session_id() -> str:
    return secrets.token_urlsafe(24)


def encrypt_secret(value: str) -> str:
    return FERNET.encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    try:
        return FERNET.decrypt(value.encode()).decode()
    except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Session secret could not be decrypted",
        ) from exc


def create_jwt(
    session_id: str,
    provider: str,
    model_id: str,
    project: str,
    ttl: int,
) -> str:
    if ttl <= 0:
        raise ValueError("JWT TTL must be positive")

    now = int(time.time())

    payload = {
        "iss": "portfolio-llm-gateway",
        "sub": session_id,
        "sid": session_id,
        "provider": provider,
        "model": model_id,
        "project": project,
        "iat": now,
        "exp": now + ttl,
        "jti": secrets.token_urlsafe(12),
    }

    return jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm="HS256",
    )


def decode_jwt(token: str) -> dict[str, Any]:
    if not settings.jwt_secret or len(settings.jwt_secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="Gateway JWT signing is not configured",
        )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer="portfolio-llm-gateway",
            options={
                "require": [
                    "iss",
                    "sub",
                    "sid",
                    "provider",
                    "model",
                    "project",
                    "iat",
                    "exp",
                    "jti",
                ],
            },
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired demo session",
        ) from exc

    if payload.get("sub") != payload.get("sid"):
        raise HTTPException(
            status_code=401,
            detail="Invalid demo session",
        )

    return payload


async def load_session_context(
    request: Request,
) -> tuple[dict[str, Any], dict[str, Any]]:
    auth = request.headers.get("authorization", "")
    parts = auth.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Bearer demo session token required",
        )

    payload = decode_jwt(parts[1])

    raw = await store.get(
        f"llm:session:{payload['sid']}"
    )

    if raw is None:
        raise HTTPException(
            status_code=401,
            detail="Demo session expired or revoked",
        )

    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Corrupt demo session",
        ) from exc

    if not isinstance(record, dict):
        raise HTTPException(
            status_code=500,
            detail="Invalid demo session record",
        )

    # The signed JWT and server-side session must agree.
    for field in ("sid", "provider", "model", "project"):
        if record.get(field) != payload.get(field):
            raise HTTPException(
                status_code=401,
                detail="Invalid demo session",
            )

    return payload, record


async def load_session_from_request(
    request: Request,
) -> dict[str, Any]:
    _, record = await load_session_context(request)
    return record


async def enforce_session_create_rate_limit(
    client_ip: str,
) -> None:
    minute_bucket = int(time.time() // 60)

    ip_digest = hashlib.sha256(
        client_ip.encode()
    ).hexdigest()[:16]

    count = await store.incr_with_ttl(
        f"llm:rl:session:{ip_digest}:{minute_bucket}",
        65,
    )

    if count > settings.max_session_creates_per_minute:
        raise HTTPException(
            status_code=429,
            detail="Too many demo sessions. Please wait and retry.",
            headers={"Retry-After": "60"},
        )


async def enforce_rate_limit(
    session_id: str,
    client_ip: str,
) -> None:
    minute_bucket = int(time.time() // 60)

    session_count = await store.incr_with_ttl(
        f"llm:rl:s:{session_id}:{minute_bucket}",
        65,
    )

    ip_digest = hashlib.sha256(
        client_ip.encode()
    ).hexdigest()[:16]

    ip_count = await store.incr_with_ttl(
        f"llm:rl:i:{ip_digest}:{minute_bucket}",
        65,
    )

    if (
        session_count > settings.max_requests_per_minute
        or ip_count > settings.max_requests_per_minute * 2
    ):
        raise HTTPException(
            status_code=429,
            detail="Demo rate limit reached. Please wait and retry.",
            headers={"Retry-After": "60"},
        )
