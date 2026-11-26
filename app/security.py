
from __future__ import annotations

import hashlib
import hmac
import json
import re
import logging
import secrets
import time
from typing import Any

import httpx
import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, Request

from .config import ALLOWED_HANDOFF_PROJECTS, settings


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

    async def consume(self, key: str) -> str | None:
        """Atomically read and delete a single-use value."""
        if self._redis is not None:
            try:
                value = await self._redis.getdel(key)
                return str(value) if value is not None else None
            except Exception as exc:
                raise self._redis_failure("getdel", exc) from exc

        item = self._memory.pop(key, None)
        if item is None:
            return None

        expiry, value = item
        if time.monotonic() >= expiry:
            return None

        return value

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


def owner_scope_for_user(user_id: str) -> str:
    """Return a stable opaque scope for a Supabase-verified user identity."""
    normalized = str(user_id or "").strip().lower()
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        normalized,
    ):
        raise ValueError("Supabase user ID is invalid")
    return hmac.new(
        settings.jwt_secret.encode("utf-8"),
        f"user:{normalized}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def owner_scope_for_session(session_id: str) -> str:
    """Give anonymous BYOK sessions their own non-shareable document scope."""
    return hmac.new(
        settings.jwt_secret.encode("utf-8"),
        f"session:{session_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


_WORKSPACE_CLIENT_ID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_WORKSPACE_TIMESTAMP = re.compile(r"^[0-9]{10}$")
_WORKSPACE_SIGNATURE = re.compile(r"^[0-9a-f]{64}$")
_WORKSPACE_IDENTITY_VERSION = "studio-workspace-v1"
_WORKSPACE_IDENTITY_MAX_SKEW_SECONDS = 60


def workspace_identity_signature_payload(client_id: str, timestamp: str) -> bytes:
    """Canonical payload shared by the Studio Worker and this Gateway."""
    return f"{_WORKSPACE_IDENTITY_VERSION}\n{client_id}\n{timestamp}".encode("utf-8")


def owner_scope_for_workspace_client(
    client_id: str,
    timestamp: str,
    signature: str,
) -> str:
    """Validate a Studio-signed client identity and return its stable owner scope.

    The Workspace access-key ID is never trusted by itself. A short-lived
    HMAC signed by the Studio Worker proves the identity was resolved from a
    valid, server-verified workspace session. The provider key and raw access
    key are not part of this identity assertion.
    """
    secret = str(getattr(settings, "workspace_identity_secret", "") or "").strip()
    if len(secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="Workspace identity binding is not configured on the Gateway.",
        )

    if (
        not isinstance(client_id, str)
        or not _WORKSPACE_CLIENT_ID.fullmatch(client_id)
        or not isinstance(timestamp, str)
        or not _WORKSPACE_TIMESTAMP.fullmatch(timestamp)
        or not isinstance(signature, str)
        or not _WORKSPACE_SIGNATURE.fullmatch(signature)
    ):
        raise HTTPException(status_code=401, detail="Invalid workspace identity assertion.")

    normalized_client_id = client_id.lower()
    issued_at = int(timestamp)
    if abs(int(time.time()) - issued_at) > _WORKSPACE_IDENTITY_MAX_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="Workspace identity assertion expired.")

    expected_signature = hmac.new(
        secret.encode("utf-8"),
        workspace_identity_signature_payload(normalized_client_id, timestamp),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=401, detail="Invalid workspace identity signature.")

    # A separate purpose string prevents the transport signature from being
    # reused as the persistent tenant identifier.
    return hmac.new(
        secret.encode("utf-8"),
        f"workspace-owner:{normalized_client_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


async def verify_supabase_access_token(access_token: str) -> str:
    """Validate a Supabase access token with Auth's /user endpoint.

    Only the canonical user ID is returned. Mutable user_metadata is never
    used for authorization or ownership.
    """
    url = str(getattr(settings, "supabase_url", "") or "").strip().rstrip("/")
    api_key = str(getattr(settings, "supabase_publishable_key", "") or "").strip()
    if not url or not api_key:
        raise HTTPException(
            status_code=503,
            detail="Supabase identity verification is not configured.",
        )
    if (
        not isinstance(access_token, str)
        or not access_token.strip()
        or len(access_token) > 8192
        or "\n" in access_token
        or "\r" in access_token
    ):
        raise HTTPException(status_code=401, detail="Invalid Supabase access token.")

    try:
        async with httpx.AsyncClient(
            timeout=5.0,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await client.get(
                f"{url}/auth/v1/user",
                headers={
                    "apikey": api_key,
                    "Authorization": f"Bearer {access_token.strip()}",
                    "Accept": "application/json",
                    "Cache-Control": "no-store",
                },
            )
    except httpx.HTTPError:
        raise HTTPException(
            status_code=503,
            detail="Supabase identity verification is temporarily unavailable.",
        ) from None

    if 300 <= response.status_code < 400:
        raise HTTPException(
            status_code=503,
            detail="Supabase identity verification redirects are not allowed.",
        )
    if response.status_code in {401, 403}:
        raise HTTPException(status_code=401, detail="Invalid Supabase access token.")
    if response.status_code != 200 or len(response.content) > 8192:
        raise HTTPException(
            status_code=503,
            detail="Supabase identity verification returned an invalid response.",
        )

    try:
        payload = response.json()
        user_id = str(payload.get("id") or "").strip().lower()
        owner_scope_for_user(user_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=401,
            detail="Supabase identity response did not contain a valid user.",
        ) from None
    return user_id


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
    *,
    session_project: str | None = None,
    owner_scope: str | None = None,
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
        "session_project": session_project or project,
        "owner_scope": owner_scope or owner_scope_for_session(session_id),
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

    token_project = payload.get("project")
    session_project = payload.get("session_project", token_project)

    if (
        not isinstance(token_project, str)
        or not isinstance(session_project, str)
        or not token_project
        or not session_project
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid demo session",
        )

    # Delegation is only allowed from the portfolio to an approved demo.
    if token_project != session_project and (
        session_project != "portfolio"
        or token_project not in ALLOWED_HANDOFF_PROJECTS
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid demo session",
        )

    for field in ("sid", "provider", "model"):
        if record.get(field) != payload.get(field):
            raise HTTPException(
                status_code=401,
                detail="Invalid demo session",
            )

    if record.get("project") != session_project:
        raise HTTPException(
            status_code=401,
            detail="Invalid demo session",
        )

    # Pre-upgrade active sessions have no owner_scope claim. Treat those as
    # per-session anonymous scopes; never infer a Supabase identity from
    # untrusted JWT metadata.
    owner_scope = payload.get("owner_scope") or owner_scope_for_session(payload["sid"])
    stored_scope = record.get("owner_scope") or owner_scope_for_session(payload["sid"])
    if (
        not isinstance(owner_scope, str)
        or not hmac.compare_digest(owner_scope, stored_scope)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid demo session",
        )
    payload["owner_scope"] = owner_scope
    record["owner_scope"] = stored_scope

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
