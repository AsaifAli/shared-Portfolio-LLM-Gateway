from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Any

import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, Request

from .config import settings


class SessionStore:
    def __init__(self) -> None:
        self._redis = None
        if settings.redis_url and settings.redis_token:
            try:
                from upstash_redis.asyncio import Redis
                self._redis = Redis(url=settings.redis_url, token=settings.redis_token)
            except Exception as exc:  # pragma: no cover - dependency/runtime guard
                raise RuntimeError(f"Failed to initialise Upstash Redis: {exc}") from exc
        self._memory: dict[str, tuple[float, str]] = {}

    async def set(self, key: str, value: str, ttl: int) -> None:
        if self._redis is not None:
            await self._redis.set(key, value, ex=ttl)
            return
        self._memory[key] = (time.time() + ttl, value)

    async def get(self, key: str) -> str | None:
        if self._redis is not None:
            value = await self._redis.get(key)
            return str(value) if value is not None else None
        item = self._memory.get(key)
        if not item:
            return None
        expiry, value = item
        if time.time() >= expiry:
            self._memory.pop(key, None)
            return None
        return value

    async def delete(self, key: str) -> None:
        if self._redis is not None:
            await self._redis.delete(key)
            return
        self._memory.pop(key, None)

    async def incr_with_ttl(self, key: str, ttl: int) -> int:
        if self._redis is not None:
            value = int(await self._redis.incr(key))
            if value == 1:
                await self._redis.expire(key, ttl)
            return value
        now = time.time()
        item = self._memory.get(key)
        if not item or now >= item[0]:
            self._memory[key] = (now + ttl, "1")
            return 1
        count = int(item[1]) + 1
        self._memory[key] = (item[0], str(count))
        return count


store = SessionStore()


def get_fernet() -> Fernet:
    if settings.encryption_key:
        return Fernet(settings.encryption_key.encode())
    # Development-only fallback. Production enforces a configured key.
    key = Fernet.generate_key()
    return Fernet(key)


FERNET = get_fernet()


def new_session_id() -> str:
    return secrets.token_urlsafe(24)


def encrypt_secret(value: str) -> str:
    return FERNET.encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    try:
        return FERNET.decrypt(value.encode()).decode()
    except InvalidToken as exc:
        raise HTTPException(status_code=500, detail="Session secret could not be decrypted") from exc


def create_jwt(session_id: str, provider: str, model_id: str, project: str, ttl: int) -> str:
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
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_jwt(token: str) -> dict[str, Any]:
    if not settings.jwt_secret:
        raise HTTPException(status_code=503, detail="Gateway JWT signing is not configured")
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer="portfolio-llm-gateway",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired demo session") from exc
    if not payload.get("sid"):
        raise HTTPException(status_code=401, detail="Invalid demo session")
    return payload


async def load_session_from_request(request: Request) -> dict[str, Any]:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer demo session token required")
    payload = decode_jwt(auth.split(" ", 1)[1].strip())
    raw = await store.get(f"llm:session:{payload['sid']}")
    if raw is None:
        raise HTTPException(status_code=401, detail="Demo session expired or revoked")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Corrupt demo session") from exc


async def enforce_session_create_rate_limit(client_ip: str) -> None:
    minute_bucket = int(time.time() // 60)
    ip_digest = hashlib.sha256(client_ip.encode()).hexdigest()[:16]
    count = await store.incr_with_ttl(f"llm:rl:session:{ip_digest}:{minute_bucket}", 65)
    if count > settings.max_session_creates_per_minute:
        raise HTTPException(status_code=429, detail="Too many demo sessions. Please wait a minute and retry.")


async def enforce_rate_limit(session_id: str, client_ip: str) -> None:
    minute_bucket = int(time.time() // 60)
    session_count = await store.incr_with_ttl(f"llm:rl:s:{session_id}:{minute_bucket}", 65)
    ip_digest = hashlib.sha256(client_ip.encode()).hexdigest()[:16]
    ip_count = await store.incr_with_ttl(f"llm:rl:i:{ip_digest}:{minute_bucket}", 65)
    if session_count > settings.max_requests_per_minute or ip_count > settings.max_requests_per_minute * 2:
        raise HTTPException(status_code=429, detail="Demo rate limit reached. Please wait a minute and retry.", headers={"Retry-After": "60"})
