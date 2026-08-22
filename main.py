from __future__ import annotations

import json
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import model_config_for_user, settings
from .llm import ProviderError, complete, stream
from .models import (
    ChatCompletionRequest,
    ProviderModel,
    ProvidersResponse,
    RevokeResponse,
    SessionCreateRequest,
    SessionCreateResponse,
)
from .security import (
    create_jwt,
    decrypt_secret,
    encrypt_secret,
    enforce_rate_limit,
    enforce_session_create_rate_limit,
    load_session_from_request,
    decode_jwt,
    new_session_id,
    store,
)

settings.validate_runtime()

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins or ["*"],
    allow_credentials=False,
    allow_origin_regex=settings.allow_origin_regex or None,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/healthz")
@app.get("/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/v1/providers", response_model=ProvidersResponse)
async def providers() -> ProvidersResponse:
    return ProvidersResponse(
        providers=[
            ProviderModel(id=model.id, provider=model.provider, label=model.label, recommended=model.recommended)
            for model in settings.models.values()
        ]
    )


@app.get("/v1/models")
async def models() -> dict[str, list[dict[str, object]]]:
    return {
        "data": [
            {
                "id": model.id,
                "object": "model",
                "owned_by": model.provider,
            }
            for model in settings.models.values()
        ]
    }


@app.post("/v1/demo/session", response_model=SessionCreateResponse)
async def create_session(payload: SessionCreateRequest, request: Request) -> SessionCreateResponse:
    await enforce_session_create_rate_limit(request.client.host if request.client else "unknown")
    try:
        model_cfg = model_config_for_user(payload.provider, payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    api_key = payload.api_key
    if not api_key:
        if not (
            settings.fallback_api_key
            and settings.fallback_provider == payload.provider
            and settings.fallback_model_id == payload.model
        ):
            raise HTTPException(status_code=400, detail="A provider API key is required for this session")
        api_key = settings.fallback_api_key

    ttl = payload.ttl_seconds or settings.session_ttl_seconds
    sid = new_session_id()
    record = {
        "sid": sid,
        "provider": payload.provider,
        "model": payload.model,
        "project": payload.project,
        "credential": encrypt_secret(api_key),
        "created_at": int(time.time()),
    }
    await store.set(f"llm:session:{sid}", json.dumps(record), ttl)
    token = create_jwt(sid, payload.provider, payload.model, payload.project, ttl)
    return SessionCreateResponse(
        access_token=token,
        expires_in=ttl,
        provider=payload.provider,
        model=payload.model,
        project=payload.project,
    )


@app.get("/v1/demo/session/status")
async def session_status(request: Request) -> dict[str, object]:
    """Validate the Bearer demo session against both JWT and server-side session state."""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer demo session token required")
    token = auth.split(" ", 1)[1].strip()
    payload = decode_jwt(token)
    raw = await store.get(f"llm:session:{payload['sid']}")
    if raw is None:
        raise HTTPException(status_code=401, detail="Demo session expired or revoked")
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Corrupt demo session") from exc
    return {
        "active": True,
        "sid": payload["sid"],
        "provider": record.get("provider") or payload.get("provider"),
        "model": record.get("model") or payload.get("model"),
        "project": record.get("project") or payload.get("project"),
        "expires_at": payload.get("exp"),
    }


@app.post("/v1/demo/session/revoke", response_model=RevokeResponse)
async def revoke_session(request: Request) -> RevokeResponse:
    session = await load_session_from_request(request)
    await store.delete(f"llm:session:{session['sid']}")
    return RevokeResponse()


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    if request.headers.get("content-length"):
        try:
            if int(request.headers["content-length"]) > settings.max_input_bytes:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError:
            pass

    session = await load_session_from_request(request)
    await enforce_rate_limit(session["sid"], request.client.host if request.client else "unknown")

    # The session is authoritative for provider/model selection. Clients may
    # send any OpenAI-compatible model label (or omit it); the gateway maps
    # the request to the allowlisted model chosen when the session was created.
    model_id = session["model"]
    try:
        model_cfg = model_config_for_user(session["provider"], model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    api_key = decrypt_secret(session["credential"])
    body = payload.model_dump(exclude_none=True)

    if payload.stream:
        return StreamingResponse(
            stream(model_cfg, api_key, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        response = await complete(model_cfg, api_key, body)
        if hasattr(response, "model_dump"):
            return JSONResponse(content=response.model_dump())
        return JSONResponse(content=dict(response))
    except ProviderError as exc:
        message = str(exc)
        lowered = message.lower()
        upstream_429 = "429" in lowered or "rate limit" in lowered or "too many requests" in lowered
        status_code = 429 if upstream_429 else 502
        headers = {"Retry-After": "10"} if upstream_429 else None
        error_type = "provider_rate_limit" if upstream_429 else "provider_error"
        return JSONResponse(
            status_code=status_code,
            headers=headers,
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": error_type,
                }
            },
        )
