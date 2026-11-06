from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import litellm

from .config import ModelConfig, settings


logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


class ProviderRateLimitError(ProviderError):
    def __init__(
        self,
        message: str,
        retry_after: str | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


def _rate_limit_metadata(
    exc: Exception,
) -> tuple[bool, str | None]:
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "http_status", None)
    )

    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}

    retry_after = (
        headers.get("retry-after")
        or headers.get("Retry-After")
        or getattr(exc, "retry_after", None)
    )

    if retry_after is not None:
        retry_after = str(retry_after).strip()

        if (
            not retry_after
            or len(retry_after) > 128
            or "\r" in retry_after
            or "\n" in retry_after
        ):
            retry_after = None

    is_rate_limited = (
        status == 429
        or exc.__class__.__name__ == "RateLimitError"
    )

    return is_rate_limited, retry_after


def _build_kwargs(
    model_cfg: ModelConfig,
    api_key: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_cfg.litellm_model,
        "api_key": api_key,
        "messages": request["messages"],
        "stream": bool(request.get("stream", False)),
        "timeout": settings.provider_timeout_seconds,
    }

    for field in (
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "response_format",
        "stop",
        "seed",
        "reasoning_effort",
        "parallel_tool_calls",
    ):
        value = request.get(field)

        if value is not None:
            kwargs[field] = value

    # Use one token-budget parameter to avoid passing conflicting values.
    requested_completion_tokens = request.get(
        "max_completion_tokens"
    )

    if requested_completion_tokens is not None:
        kwargs["max_completion_tokens"] = min(
            int(requested_completion_tokens),
            settings.max_output_tokens,
        )
    else:
        requested_tokens = request.get("max_tokens")

        if requested_tokens is None:
            requested_tokens = settings.max_output_tokens

        kwargs["max_tokens"] = min(
            int(requested_tokens),
            settings.max_output_tokens,
        )

    if model_cfg.provider == "openrouter":
        kwargs["api_base"] = "https://openrouter.ai/api/v1"

        import os

        extra_headers: dict[str, str] = {}

        if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
            extra_headers["HTTP-Referer"] = referer

        if title := os.getenv("OPENROUTER_X_TITLE"):
            extra_headers["X-Title"] = title

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

    return kwargs


async def complete(
    model_cfg: ModelConfig,
    api_key: str,
    request: dict[str, Any],
) -> Any:
    try:
        return await litellm.acompletion(
            **_build_kwargs(model_cfg, api_key, request)
        )

    except Exception as exc:
        is_rate_limited, retry_after = _rate_limit_metadata(exc)

        # Log the exception class, not the raw upstream message. Provider
        # errors can contain request details or sensitive information.
        logger.warning(
            "LLM provider request failed: provider=%s exception_type=%s",
            model_cfg.provider,
            type(exc).__name__,
        )

        if is_rate_limited:
            raise ProviderRateLimitError(
                "The upstream provider rate-limited this request.",
                retry_after=retry_after,
            ) from exc

        raise ProviderError(
            "The upstream provider request failed."
        ) from exc


async def stream(
    model_cfg: ModelConfig,
    api_key: str,
    request: dict[str, Any],
) -> AsyncIterator[str]:
    try:
        response = await litellm.acompletion(
            **_build_kwargs(model_cfg, api_key, request)
        )

        async for chunk in response:
            payload = (
                chunk.model_dump()
                if hasattr(chunk, "model_dump")
                else dict(chunk)
            )

            yield (
                f"data: {json.dumps(payload, default=str)}\n\n"
            )

        yield "data: [DONE]\n\n"

    except Exception as exc:
        is_rate_limited, retry_after = _rate_limit_metadata(exc)

        logger.warning(
            "LLM streaming request failed: provider=%s exception_type=%s",
            model_cfg.provider,
            type(exc).__name__,
        )

        error_type = (
            "rate_limit_error"
            if is_rate_limited
            else "provider_error"
        )

        message = (
            "The upstream provider rate-limited this request."
            if is_rate_limited
            else "The upstream provider request failed."
        )

        error: dict[str, Any] = {
            "message": message,
            "type": error_type,
            "code": error_type,
        }

        if retry_after:
            error["retry_after"] = retry_after

        yield (
            "data: "
            + json.dumps({"error": error})
            + "\n\n"
        )

        yield "data: [DONE]\n\n"