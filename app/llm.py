from __future__ import annotations

from typing import Any, AsyncIterator

import litellm

from .config import ModelConfig, settings


class ProviderError(RuntimeError):
    pass


def _build_kwargs(model_cfg: ModelConfig, api_key: str, request: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_cfg.litellm_model,
        "api_key": api_key,
        "messages": request["messages"],
        "stream": bool(request.get("stream", False)),
    }
    for field in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "tools", "tool_choice", "response_format", "stop", "seed", "reasoning_effort", "parallel_tool_calls"):
        value = request.get(field)
        if value is not None:
            kwargs[field] = value

    # User-facing models are allowlisted; we still clamp requested generation.
    if isinstance(kwargs.get("max_tokens"), int):
        kwargs["max_tokens"] = min(kwargs["max_tokens"], settings.max_output_tokens)

    if model_cfg.provider == "openrouter":
        kwargs["api_base"] = "https://openrouter.ai/api/v1"
        # OpenRouter attribution headers are optional; keep them configurable.
        extra_headers: dict[str, str] = {}
        if referer := __import__("os").getenv("OPENROUTER_HTTP_REFERER"):
            extra_headers["HTTP-Referer"] = referer
        if title := __import__("os").getenv("OPENROUTER_X_TITLE"):
            extra_headers["X-Title"] = title
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

    return kwargs


async def complete(model_cfg: ModelConfig, api_key: str, request: dict[str, Any]) -> Any:
    try:
        return await litellm.acompletion(**_build_kwargs(model_cfg, api_key, request))
    except Exception as exc:
        raise ProviderError(str(exc)) from exc


async def stream(model_cfg: ModelConfig, api_key: str, request: dict[str, Any]) -> AsyncIterator[str]:
    try:
        response = await litellm.acompletion(**_build_kwargs(model_cfg, api_key, request))
        async for chunk in response:
            payload = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
            import json
            yield f"data: {json.dumps(payload, default=str)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        import json
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'gateway_error'}})}\n\n"
        yield "data: [DONE]\n\n"
