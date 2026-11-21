from __future__ import annotations

import json
import time
from typing import Any
import logging

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import model_config_for_user, settings
from .llm import (
    ProviderError,
    ProviderRateLimitError,
    complete,
    prepare_stream,
)
from .models import (
    ChatCompletionRequest,
    ProviderModel,
    ProvidersResponse,
    RevokeResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionHandoffRequest,
    SessionHandoffResponse,
    SessionExchangeRequest,
    SessionExchangeResponse,
)
from .handoff import (
    HANDOFF_CODE_TTL_SECONDS,
    exchange_session_handoff,
    issue_session_handoff,
)
from .security import (
    create_jwt,
    decrypt_secret,
    encrypt_secret,
    enforce_rate_limit,
    enforce_session_create_rate_limit,
    load_session_context,
    load_session_from_request,
    new_session_id,
    store,
    SessionStoreUnavailable,
)


settings.validate_runtime()
logger = logging.getLogger(__name__)

class RequestSizeLimitMiddleware:
    """
    Enforce a request-body limit before FastAPI parses the body.

    This checks Content-Length when present and also counts streamed
    request chunks, so requests without Content-Length are bounded too.
    """

    def __init__(
        self,
        app: Any,
        max_bytes: int,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _send_error(
        self,
        send: Any,
        status_code: int,
        message: str,
    ) -> None:
        body = json.dumps(
            {"detail": message}
        ).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })

        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or not scope.get("path", "").startswith("/v1/")
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")

        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError):
                await self._send_error(
                    send,
                    400,
                    "Invalid Content-Length header",
                )
                return

            if declared_length < 0:
                await self._send_error(
                    send,
                    400,
                    "Invalid Content-Length header",
                )
                return

            if declared_length > self.max_bytes:
                await self._send_error(
                    send,
                    413,
                    "Request body too large",
                )
                return

        chunks: list[bytes] = []
        total_bytes = 0

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return

            if message["type"] != "http.request":
                continue

            chunk = message.get("body", b"")
            total_bytes += len(chunk)

            if total_bytes > self.max_bytes:
                await self._send_error(
                    send,
                    413,
                    "Request body too large",
                )
                return

            chunks.append(chunk)

            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered

            if not delivered:
                delivered = True

                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }

            return await receive()

        await self.app(
            scope,
            replay_receive,
            send,
        )


app = FastAPI(
    title=settings.app_name,
    version="1.1.0",
)

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=settings.max_input_bytes,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_origin_regex=settings.allow_origin_regex or None,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

@app.exception_handler(SessionStoreUnavailable)
async def session_store_unavailable_handler(
    request: Request,
    exc: SessionStoreUnavailable,
):
    logger.warning(
        "Returning 503 because the session store is unavailable: path=%s",
        request.url.path,
    )

    return JSONResponse(
        status_code=503,
        headers={"Retry-After": "5"},
        content={
            "detail": "Session store is temporarily unavailable."
        },
    )

@app.get("/healthz")
@app.get("/v1/health")
async def health() -> dict[str, str]:
    # Liveness only: confirms that the process is responding.
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@app.get("/readyz")
@app.get("/v1/ready")
async def readiness() -> dict[str, str]:
    if not await store.ping():
        raise HTTPException(
            status_code=503,
            detail="Session store is unavailable",
            headers={"Retry-After": "5"},
        )

    return {
        "status": "ready",
        "service": settings.app_name,
        "session_store": store.backend,
    }


@app.get("/v1/providers", response_model=ProvidersResponse)
async def providers() -> ProvidersResponse:
    return ProvidersResponse(
        providers=[
            ProviderModel(
                id=model.id,
                provider=model.provider,
                label=model.label,
                recommended=model.recommended,
            )
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


@app.post(
    "/v1/demo/session",
    response_model=SessionCreateResponse,
)
async def create_session(
    payload: SessionCreateRequest,
    request: Request,
) -> SessionCreateResponse:
    await enforce_session_create_rate_limit(
        request.client.host if request.client else "unknown"
    )

    try:
        model_config_for_user(
            payload.provider,
            payload.model,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    api_key = payload.api_key

    if not api_key:
        fallback_matches = (
            settings.fallback_api_key
            and settings.fallback_provider == payload.provider
            and settings.fallback_model_id == payload.model
        )

        if not fallback_matches:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A provider API key is required for this session"
                ),
            )

        api_key = settings.fallback_api_key

    # The configured server TTL is the maximum session lifetime.
    requested_ttl = (
        payload.ttl_seconds
        if payload.ttl_seconds is not None
        else settings.session_ttl_seconds
    )

    ttl = min(
        requested_ttl,
        settings.session_ttl_seconds,
    )

    sid = new_session_id()

    record = {
        "sid": sid,
        "provider": payload.provider,
        "model": payload.model,
        "project": payload.project,
        "credential": encrypt_secret(api_key),
        "created_at": int(time.time()),
    }

    await store.set(
        f"llm:session:{sid}",
        json.dumps(record),
        ttl,
    )

    token = create_jwt(
        sid,
        payload.provider,
        payload.model,
        payload.project,
        ttl,
    )

    return SessionCreateResponse(
        access_token=token,
        expires_in=ttl,
        provider=payload.provider,
        model=payload.model,
        project=payload.project,
    )


@app.post(
    "/v1/demo/session/handoff",
    response_model=SessionHandoffResponse,
)
async def create_handoff(
    payload: SessionHandoffRequest,
    request: Request,
    response: Response,
) -> SessionHandoffResponse:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    claims, session = await load_session_context(request)

    await enforce_rate_limit(
        session["sid"],
        request.client.host if request.client else "unknown",
    )

    code = await issue_session_handoff(
        claims,
        session,
        payload.project,
    )

    return SessionHandoffResponse(
        code=code,
        expires_in=HANDOFF_CODE_TTL_SECONDS,
        project=payload.project,
    )


@app.post(
    "/v1/demo/session/exchange",
    response_model=SessionExchangeResponse,
)
async def exchange_handoff(
    payload: SessionExchangeRequest,
    request: Request,
    response: Response,
) -> SessionExchangeResponse:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    await enforce_session_create_rate_limit(
        request.client.host if request.client else "unknown"
    )

    return await exchange_session_handoff(
        payload.code,
        payload.project,
    )


@app.get("/v1/demo/session/status")
async def session_status(
    request: Request,
) -> dict[str, object]:
    claims, record = await load_session_context(request)

    return {
        "active": True,
        "sid": claims["sid"],
        "provider": record["provider"],
        "model": record["model"],
        "project": claims["project"],
        "session_project": record["project"],
        "expires_at": claims["exp"],
    }


@app.post(
    "/v1/demo/session/revoke",
    response_model=RevokeResponse,
)
async def revoke_session(
    request: Request,
) -> RevokeResponse:
    session = await load_session_from_request(request)

    await store.delete(
        f"llm:session:{session['sid']}"
    )

    return RevokeResponse()


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
):
    session = await load_session_from_request(request)

    await enforce_rate_limit(
        session["sid"],
        request.client.host if request.client else "unknown",
    )

    try:
        model_cfg = model_config_for_user(
            session["provider"],
            session["model"],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid session model configuration",
        ) from exc

    api_key = decrypt_secret(session["credential"])

    body = payload.model_dump(exclude_none=True)

    if payload.stream:
        try:
            # Prime the upstream stream before response headers are sent.
            # Authentication, model, quota and connection failures can then
            # return a truthful HTTP status instead of a misleading HTTP 200.
            event_stream = await prepare_stream(
                model_cfg,
                api_key,
                body,
            )
        except ProviderError as exc:
            headers = {}
            if exc.retry_after:
                headers["Retry-After"] = exc.retry_after
            return JSONResponse(
                status_code=exc.status_code,
                headers=headers,
                content={
                    "error": {
                        "message": str(exc),
                        "type": exc.code,
                        "code": exc.code,
                    }
                },
            )

        return StreamingResponse(
            event_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = await complete(
            model_cfg,
            api_key,
            body,
        )

        if hasattr(response, "model_dump"):
            return JSONResponse(
                content=response.model_dump()
            )

        return JSONResponse(
            content=dict(response)
        )

    except ProviderRateLimitError as exc:
        headers = {}

        if exc.retry_after:
            headers["Retry-After"] = exc.retry_after

        return JSONResponse(
            status_code=429,
            headers=headers,
            content={
                "error": {
                    "message": (
                        "The upstream provider rate-limited this request."
                    ),
                    "type": "rate_limit_error",
                    "code": "rate_limit",
                }
            },
        )

    except ProviderError as exc:
        headers = {}
        if exc.retry_after:
            headers["Retry-After"] = exc.retry_after
        return JSONResponse(
            status_code=exc.status_code,
            headers=headers,
            content={
                "error": {
                    "message": str(exc),
                    "type": exc.code,
                    "code": exc.code,
                }
            },
        )
