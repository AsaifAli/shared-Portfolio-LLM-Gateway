from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Any

from fastapi import HTTPException

from .config import settings
from .models import SessionExchangeResponse
from .security import create_jwt, store


HANDOFF_CODE_TTL_SECONDS = 60


def _handoff_storage_key(code: str) -> str:
    value = code.strip()
    if not value or len(value) > 128:
        raise HTTPException(
            status_code=401,
            detail="Invalid, expired, or already used handoff code",
        )

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"llm:handoff:{digest}"


async def issue_session_handoff(
    claims: dict[str, Any],
    session: dict[str, Any],
    target_project: str,
) -> str:
    """Create a short-lived code from an already authenticated session."""
    sid = claims.get("sid")

    if not isinstance(sid, str) or not sid:
        raise HTTPException(status_code=401, detail="Invalid demo session")

    for field in ("sid", "provider", "model", "project"):
        if claims.get(field) != session.get(field):
            raise HTTPException(status_code=401, detail="Invalid demo session")

    try:
        session_expires_at = int(claims["exp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid demo session") from exc

    if session_expires_at <= int(time.time()):
        raise HTTPException(status_code=401, detail="Demo session expired")

    code = secrets.token_urlsafe(32)

    handoff_record = {
        "sid": sid,
        "provider": session["provider"],
        "model": session["model"],
        "session_project": session["project"],
        "target_project": target_project,
        "session_expires_at": session_expires_at,
    }

    await store.set(
        _handoff_storage_key(code),
        json.dumps(handoff_record, separators=(",", ":")),
        HANDOFF_CODE_TTL_SECONDS,
    )

    return code


async def exchange_session_handoff(
    code: str,
    target_project: str,
) -> SessionExchangeResponse:
    """Atomically redeem a project-bound handoff code for a session JWT."""
    key = _handoff_storage_key(code)
    raw = await store.get(key)

    if raw is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid, expired, or already used handoff code",
        )

    try:
        handoff = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid handoff code") from exc

    if (
        not isinstance(handoff, dict)
        or handoff.get("target_project") != target_project
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid, expired, or already used handoff code",
        )

    # GETDEL is atomic in Redis: only one concurrent request can win.
    consumed = await store.consume(key)
    if consumed is None or consumed != raw:
        raise HTTPException(
            status_code=401,
            detail="Invalid, expired, or already used handoff code",
        )

    sid = handoff.get("sid")
    if not isinstance(sid, str) or not sid:
        raise HTTPException(status_code=401, detail="Invalid handoff code")

    session_raw = await store.get(f"llm:session:{sid}")
    if session_raw is None:
        raise HTTPException(
            status_code=401,
            detail="The original demo session has expired or been revoked",
        )

    try:
        session = json.loads(session_raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid demo session") from exc

    if not isinstance(session, dict):
        raise HTTPException(status_code=401, detail="Invalid demo session")

    expected_fields = {
        "sid": "sid",
        "provider": "provider",
        "model": "model",
        "session_project": "project",
    }

    for handoff_field, session_field in expected_fields.items():
        if handoff.get(handoff_field) != session.get(session_field):
            raise HTTPException(status_code=401, detail="Invalid demo session")

    try:
        session_expires_at = int(handoff["session_expires_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid handoff code") from exc

    remaining = session_expires_at - int(time.time())
    ttl = min(remaining, settings.session_ttl_seconds)

    if ttl <= 0:
        raise HTTPException(status_code=401, detail="Demo session expired")

    access_token = create_jwt(
        session["sid"],
        session["provider"],
        session["model"],
        session["project"],
        ttl,
    )

    return SessionExchangeResponse(
        access_token=access_token,
        expires_in=ttl,
        provider=session["provider"],
        model=session["model"],
        target_project=target_project,
    )
