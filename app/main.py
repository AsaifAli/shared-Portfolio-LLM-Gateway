from __future__ import annotations

import json
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
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
    model_cfg = settings.models.get(payload.model)
    if model_cfg is None:
        raise HTTPException(status_code=400, detail="Model is not allowed by this gateway")
    if model_cfg.provider != payload.provider:
        raise HTTPException(status_code=400, detail="Provider does not match the selected model")

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
    model_cfg = settings.models.get(model_id)
    if model_cfg is None or model_cfg.provider != session["provider"]:
        raise HTTPException(status_code=400, detail="Session model is not allowed")

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
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": str(exc),
                    "type": "provider_error",
                    "code": "provider_error",
                }
            },
        )
