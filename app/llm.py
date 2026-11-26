"""Direct provider adapters for the Portfolio LLM Gateway.

OpenAI and OpenRouter use the official OpenAI async SDK. Anthropic uses its
OpenAI-compatible Chat Completions endpoint. Gemini uses Google's native Gen
AI SDK so function-call parts and streaming tool calls are not forced through
Gemini's OpenAI-compatibility translation layer.

The public Gateway response remains OpenAI Chat Completions compatible.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .config import ModelConfig, settings

logger = logging.getLogger(__name__)

MAX_PROVIDER_RETRIES = 2
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 500, 502, 503, 504, 529})
RETRYABLE_EXCEPTION_NAMES = frozenset({
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "WriteError",
    "WriteTimeout",
    "TimeoutError",
    "ConnectionError",
    "ServiceUnavailableError",
})


class ProviderError(RuntimeError):
    """Sanitized provider failure with a safe HTTP status and error code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "provider_error",
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after


class ProviderRateLimitError(ProviderError):
    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__(
            "The selected provider rate-limited this request. Please retry later.",
            status_code=429,
            code="rate_limit_error",
            retry_after=retry_after,
        )


class ProviderAuthenticationError(ProviderError):
    def __init__(self) -> None:
        super().__init__(
            "The selected provider rejected the API credential. Check the key and its permissions.",
            status_code=401,
            code="provider_authentication_error",
        )


class ProviderRequestError(ProviderError):
    def __init__(self, status_code: int = 400) -> None:
        super().__init__(
            "The selected provider rejected the model or request parameters.",
            status_code=status_code,
            code="provider_request_error",
        )


class ProviderBillingError(ProviderError):
    def __init__(self) -> None:
        super().__init__(
            "The selected provider account has insufficient credits or billing capacity.",
            status_code=402,
            code="provider_billing_error",
        )


class ProviderConnectionError(ProviderError):
    def __init__(self) -> None:
        super().__init__(
            "The Gateway could not establish a reliable connection to the selected provider. Please retry.",
            status_code=502,
            code="provider_connection_error",
        )


class ProviderUnavailableError(ProviderError):
    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__(
            "The selected provider is temporarily unavailable. Please retry.",
            status_code=503,
            code="provider_unavailable_error",
            retry_after=retry_after,
        )


class ProviderEmptyCompletionError(ProviderError):
    """A provider returned a nominal success without user-visible output."""

    def __init__(self, finish_reason: str) -> None:
        if finish_reason == "length":
            message = (
                "The selected model exhausted its output-token budget before "
                "producing visible text. Lower its reasoning effort or increase "
                "its output-token budget."
            )
        else:
            message = (
                "The selected model returned no visible text. Retry after "
                "provider warm-up or choose a different model."
            )
        super().__init__(
            message,
            status_code=502,
            code="empty_completion",
        )


def _provider_model_id(model_cfg: ModelConfig) -> str:
    """Strip legacy provider prefixes retained for old MODEL_REGISTRY_JSON."""
    value = model_cfg.model_id.strip()
    prefixes = {
        "openai": "openai/",
        "google": "gemini/",
        "anthropic": "anthropic/",
        "openrouter": "openrouter/",
    }
    prefix = prefixes.get(model_cfg.provider)
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    return value


def _retry_after(exc: BaseException) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = (
        headers.get("retry-after")
        or headers.get("Retry-After")
        or getattr(exc, "retry_after", None)
    )
    if value is None:
        return None
    value = str(value).strip()
    if not value or len(value) > 128 or "\r" in value or "\n" in value:
        return None
    return value


def _status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    value = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(response, "status_code", None)
    )
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _exception_chain(exc: BaseException) -> list[str]:
    """Return exception class names only; never log raw provider messages."""
    chain: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return chain


def _is_retryable(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status in RETRYABLE_STATUS_CODES
    return any(
        name in RETRYABLE_EXCEPTION_NAMES
        for name in _exception_chain(exc)
    ) or isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _to_provider_error(exc: BaseException) -> ProviderError:
    status = _status_code(exc)
    retry_after = _retry_after(exc)
    if status == 429:
        return ProviderRateLimitError(retry_after)
    if status in {401, 403}:
        return ProviderAuthenticationError()
    if status == 402:
        return ProviderBillingError()
    if status in {400, 404, 413, 422}:
        return ProviderRequestError(status)
    if status in RETRYABLE_STATUS_CODES or (status is not None and status >= 500):
        return ProviderUnavailableError(retry_after)
    chain = _exception_chain(exc)
    if any(name in RETRYABLE_EXCEPTION_NAMES for name in chain):
        return ProviderConnectionError()
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return ProviderConnectionError()
    return ProviderError(
        "The selected provider request failed. Check the provider, model and API credential.",
        status_code=502,
        code="provider_error",
    )


def _log_provider_failure(
    provider: str,
    exc: BaseException,
    *,
    attempt: int,
    retrying: bool,
    stream_started: bool = False,
) -> None:
    logger.warning(
        "Provider request failed: provider=%s exception_type=%s cause_chain=%s upstream_status=%s attempt=%s retrying=%s stream_started=%s",
        provider,
        type(exc).__name__,
        ">".join(_exception_chain(exc)),
        _status_code(exc),
        attempt,
        retrying,
        stream_started,
    )


async def _call_with_retry(
    provider: str,
    operation: Callable[[], Awaitable[Any]],
) -> Any:
    for attempt in range(1, MAX_PROVIDER_RETRIES + 2):
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            retrying = attempt <= MAX_PROVIDER_RETRIES and _is_retryable(exc)
            _log_provider_failure(
                provider,
                exc,
                attempt=attempt,
                retrying=retrying,
            )
            if retrying:
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))
                continue
            raise _to_provider_error(exc) from exc
    raise ProviderError("The selected provider request failed.")


def _openai_endpoint(provider: str) -> tuple[str, dict[str, str]]:
    headers: dict[str, str] = {}
    if provider == "openai":
        return "https://api.openai.com/v1", headers
    if provider == "openrouter":
        if referer := os.getenv("OPENROUTER_HTTP_REFERER", "").strip():
            headers["HTTP-Referer"] = referer
        if title := os.getenv("OPENROUTER_X_TITLE", "").strip():
            headers["X-Title"] = title
        return "https://openrouter.ai/api/v1", headers
    if provider == "anthropic":
        workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
        if workspace_id:
            headers["anthropic-workspace-id"] = workspace_id
        return "https://api.anthropic.com/v1/", headers
    raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}")


def _build_openai_kwargs(
    model_cfg: ModelConfig,
    request: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": _provider_model_id(model_cfg),
        "messages": request["messages"],
    }
    allowed_fields = (
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "response_format",
        "stop",
        "seed",
        "parallel_tool_calls",
        "frequency_penalty",
        "presence_penalty",
        "user",
    )
    for field in allowed_fields:
        value = request.get(field)
        if value is not None:
            kwargs[field] = value

    # GPT-OSS is a reasoning model. At the Gateway's default 4k completion
    # budget, unconstrained reasoning can consume every token and leave the
    # OpenAI-compatible message.content empty. Keep this preset useful for
    # application workflows by reserving most of the budget for visible output.
    if (
        model_cfg.provider == "openrouter"
        and _provider_model_id(model_cfg) == "openai/gpt-oss-20b:free"
    ):
        kwargs["extra_body"] = {"reasoning": {"effort": "low"}}

    if model_cfg.provider == "anthropic":
        response_format = request.get("response_format")
        if response_format and (
            response_format.get("type") != "json_schema"
            or not (response_format.get("json_schema") or {}).get("strict")
        ):
            raise ProviderRequestError(400)
        # Anthropic's compatibility API rejects some sampling combinations.
        # It also ignores these OpenAI-only penalties, so do not forward them.
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        kwargs.pop("seed", None)
        kwargs.pop("user", None)
        if "temperature" in kwargs and "top_p" in kwargs:
            kwargs.pop("top_p", None)

        # Claude 4.7+ and Mythos Preview only accept default sampling values.
        model_name = model_cfg.model_id.lower()
        version_match = re.search(r"claude-[a-z0-9]+-(\d+)-(\d+)", model_name)
        is_new_sampling_family = (
            "mythos" in model_name
            or bool(
                version_match
                and (
                    int(version_match.group(1)) > 4
                    or (
                        int(version_match.group(1)) == 4
                        and int(version_match.group(2)) >= 7
                    )
                )
            )
        )
        if is_new_sampling_family:
            if kwargs.get("temperature") != 1:
                kwargs.pop("temperature", None)
            if kwargs.get("top_p") != 0.99:
                kwargs.pop("top_p", None)

    requested_tokens = request.get("max_completion_tokens")
    if requested_tokens is None:
        requested_tokens = request.get("max_tokens")
    if requested_tokens is None:
        requested_tokens = settings.max_output_tokens
    token_limit = min(int(requested_tokens), settings.max_output_tokens)
    # OpenAI's newer reasoning model families use max_completion_tokens;
    # OpenRouter and Anthropic's compatibility API accept max_tokens.
    if model_cfg.provider == "openai":
        kwargs["max_completion_tokens"] = token_limit
    else:
        kwargs["max_tokens"] = token_limit
    return kwargs


def _openai_client(model_cfg: ModelConfig, api_key: str):
    from openai import AsyncOpenAI

    base_url, headers = _openai_endpoint(model_cfg.provider)
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=headers or None,
        timeout=float(settings.provider_timeout_seconds),
        # Retries are controlled here so streaming failures are never silently
        # replayed after partial output has reached the downstream application.
        max_retries=0,
    )


def _gemini_client(api_key: str):
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=settings.provider_timeout_seconds * 1000,
        ),
    )
    return client, types


async def _close_gemini_clients(client: Any, async_client: Any) -> None:
    """Close both SDK client surfaces, including the unused sync transport."""
    try:
        await async_client.aclose()
    except Exception:
        pass
    try:
        client.close()
    except Exception:
        pass


def _content_text_and_parts(content: Any) -> list[Any]:
    """Convert OpenAI message content to Gemini text/inline-image Parts."""
    from google.genai import types

    if content is None:
        return []
    if isinstance(content, str):
        return [types.Part.from_text(text=content)] if content else []
    if not isinstance(content, list):
        return [types.Part.from_text(text=str(content))]

    parts: list[Any] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(types.Part.from_text(text=str(item)))
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if text:
                parts.append(types.Part.from_text(text=str(text)))
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str) or not url.startswith("data:"):
                raise ProviderRequestError(400)
            try:
                header, encoded = url.split(",", 1)
                mime_type = header[5:].split(";", 1)[0]
                image_data = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise ProviderRequestError(400) from exc
            if len(image_data) > settings.max_input_bytes:
                raise ProviderRequestError(413)
            parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))
        else:
            raise ProviderRequestError(400)
    return parts


def _encode_gemini_thought_signature(signature: Any) -> str | None:
    if signature is None:
        return None
    raw = signature if isinstance(signature, bytes) else str(signature).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return encoded or None


def _decode_gemini_thought_signature(tool_call_id: str) -> bytes | None:
    marker = ".gsig."
    if marker not in tool_call_id:
        return None
    encoded = tool_call_id.rsplit(marker, 1)[1]
    try:
        return base64.urlsafe_b64decode(encoded + ("=" * (-len(encoded) % 4)))
    except (ValueError, TypeError):
        return None


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            return {"result": value}
    return {"result": value}


def _gemini_contents(messages: list[dict[str, Any]], types: Any) -> tuple[list[Any], str | None]:
    contents: list[Any] = []
    system_instructions: list[str] = []
    tool_names_by_id: dict[str, str] = {}

    for message in messages:
        role = str(message.get("role") or "user").lower()
        content = message.get("content")
        parts: list[Any] = []

        if role in {"system", "developer"}:
            if content:
                if isinstance(content, list):
                    system_instructions.append("\n".join(
                        str(item.get("text", "")) if isinstance(item, dict) else str(item)
                        for item in content
                    ))
                else:
                    system_instructions.append(str(content))
            continue

        if role == "assistant":
            parts.extend(_content_text_and_parts(content))
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                if not name:
                    continue
                call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex}")
                tool_names_by_id[call_id] = name
                function_part = types.Part.from_function_call(
                    name=name,
                    args=_json_object(function.get("arguments", "{}")),
                )
                # Gemini 3+ can require the opaque thought signature on the
                # original function-call Part after the tool result returns.
                signature = _decode_gemini_thought_signature(call_id)
                if signature is not None:
                    function_part.thought_signature = signature
                parts.append(function_part)
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue

        if role in {"tool", "function"}:
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or tool_names_by_id.get(call_id) or "")
            if not name:
                raise ProviderRequestError(400)
            parts.append(types.Part.from_function_response(
                name=name,
                response=_json_object(content),
            ))
            contents.append(types.Content(role="tool", parts=parts))
            continue

        parts.extend(_content_text_and_parts(content))
        if parts:
            contents.append(types.Content(role="user", parts=parts))

    if not contents:
        raise ProviderRequestError(400)
    return contents, "\n".join(part for part in system_instructions if part) or None


def _gemini_tools(request: dict[str, Any], types: Any) -> tuple[list[Any] | None, Any | None]:
    raw_tools = request.get("tools") or []
    declarations = []
    for item in raw_tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        function = item.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        declarations.append(types.FunctionDeclaration(
            name=name,
            description=str(function.get("description") or ""),
            parameters_json_schema=function.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        ))
    if not declarations:
        return None, None

    tools = [types.Tool(function_declarations=declarations)]
    tool_config = None
    choice = request.get("tool_choice")
    mode = "AUTO"
    allowed_names = None
    if choice == "none":
        mode = "NONE"
    elif choice == "required":
        mode = "ANY"
    elif isinstance(choice, dict):
        function = choice.get("function") or {}
        name = function.get("name")
        if name:
            mode = "ANY"
            allowed_names = [str(name)]
    if mode != "AUTO" or allowed_names:
        options: dict[str, Any] = {"mode": mode}
        if allowed_names:
            options["allowed_function_names"] = allowed_names
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(**options)
        )
    return tools, tool_config


def _gemini_config(request: dict[str, Any], system_instruction: str | None, types: Any) -> Any:
    config: dict[str, Any] = {
        "max_output_tokens": min(
            int(request.get("max_completion_tokens") or request.get("max_tokens") or settings.max_output_tokens),
            settings.max_output_tokens,
        ),
    }
    if system_instruction:
        config["system_instruction"] = system_instruction
    for source, target in (("temperature", "temperature"), ("top_p", "top_p"), ("stop", "stop_sequences"), ("seed", "seed")):
        value = request.get(source)
        if value is not None:
            config[target] = value if source != "stop" else ([value] if isinstance(value, str) else value)
    tools, tool_config = _gemini_tools(request, types)
    if tools:
        config["tools"] = tools
        # We dispatch tools in the downstream LangGraph/agent application.
        config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    if tool_config:
        config["tool_config"] = tool_config
    response_format = request.get("response_format") or {}
    if response_format.get("type") == "json_object":
        config["response_mime_type"] = "application/json"
    elif response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if schema:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = schema
    return types.GenerateContentConfig(**config)


def _finish_reason(value: Any) -> str:
    if value is None:
        return "stop"
    name = str(value).split(".")[-1].upper()
    if name in {"MAX_TOKENS", "MAX_OUTPUT_TOKENS"}:
        return "length"
    if name in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
        return "content_filter"
    if name in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
        return "tool_calls"
    return "stop"


def _gemini_response(response: Any, model: str) -> dict[str, Any]:
    candidates = getattr(response, "candidates", None) or []
    candidate = candidates[0] if candidates else None
    content = getattr(candidate, "content", None) if candidate else None
    parts = getattr(content, "parts", None) or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in parts:
        text_value = getattr(part, "text", None)
        if text_value:
            text_parts.append(str(text_value))
        function_call = getattr(part, "function_call", None)
        if function_call:
            call_id = f"call_{uuid.uuid4().hex}"
            signature = _encode_gemini_thought_signature(
                getattr(part, "thought_signature", None)
            )
            if signature:
                call_id += f".gsig.{signature}"
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(getattr(function_call, "name", "")),
                    "arguments": json.dumps(
                        getattr(function_call, "args", None) or {},
                        separators=(",", ":"),
                    ),
                },
            })
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    finish = _finish_reason(getattr(candidate, "finish_reason", None) if candidate else None)
    if tool_calls:
        finish = "tool_calls"
    return {
        "id": str(getattr(response, "response_id", None) or f"chatcmpl-{uuid.uuid4().hex}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(text_parts) or None,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            },
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(getattr(usage, "total_token_count", 0) or prompt_tokens + completion_tokens),
        },
    }


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, default=str, separators=(",", ":")) + "\n\n"


def _openai_chunk_sse(chunk: Any) -> str:
    payload = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
    return _sse(payload)


async def _close_async_resource(resource: Any) -> None:
    """Best-effort close for SDK streams; close failures must not mask the cause."""
    if resource is None:
        return
    close = getattr(resource, "aclose", None)
    if not callable(close):
        close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception:
        pass


def _gemini_chunk_payload(chunk: Any, *, model: str, stream_id: str, created: int) -> dict[str, Any] | None:
    candidates = getattr(chunk, "candidates", None) or []
    candidate = candidates[0] if candidates else None
    content = getattr(candidate, "content", None) if candidate else None
    parts = getattr(content, "parts", None) or []
    delta: dict[str, Any] = {}
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in parts:
        text_value = getattr(part, "text", None)
        if text_value:
            text_parts.append(str(text_value))
        function_call = getattr(part, "function_call", None)
        if function_call:
            call_id = f"call_{uuid.uuid4().hex}"
            signature = _encode_gemini_thought_signature(
                getattr(part, "thought_signature", None)
            )
            if signature:
                call_id += f".gsig.{signature}"
            tool_calls.append({
                "index": len(tool_calls),
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(getattr(function_call, "name", "")),
                    "arguments": json.dumps(getattr(function_call, "args", None) or {}, separators=(",", ":")),
                },
            })
    if text_parts:
        delta["content"] = "".join(text_parts)
    if tool_calls:
        delta["tool_calls"] = tool_calls
    finish_reason = None
    if candidate and getattr(candidate, "finish_reason", None) is not None:
        finish_reason = "tool_calls" if tool_calls else _finish_reason(candidate.finish_reason)
    if not delta and finish_reason is None:
        return None
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _response_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _has_visible_text(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, str) and part.strip():
            return True
        text_value = _response_field(part, "text")
        if isinstance(text_value, str) and text_value.strip():
            part_type = _response_field(part, "type")
            if part_type in {None, "text", "output_text"}:
                return True
    return False


def _validate_completion_response(response: Any, model_cfg: ModelConfig) -> None:
    """Reject 200-style provider responses that contain no usable completion."""
    choices = _response_field(response, "choices") or []
    choice = choices[0] if choices else None
    message = _response_field(choice, "message") if choice is not None else None
    content = _response_field(message, "content") if message is not None else None
    tool_calls = _response_field(message, "tool_calls") or [] if message is not None else []
    refusal = _response_field(message, "refusal") if message is not None else None

    # Tool calls and refusals are valid structured completions. The downstream
    # app can decide whether it supports those response types.
    if _has_visible_text(content) or tool_calls or refusal:
        return

    finish_reason = str(_response_field(choice, "finish_reason") or "unknown").lower()
    if finish_reason not in {
        "stop", "length", "tool_calls", "content_filter", "function_call", "unknown"
    }:
        finish_reason = "other"

    usage = _response_field(response, "usage")
    completion_details = _response_field(usage, "completion_tokens_details")
    completion_tokens = _response_field(usage, "completion_tokens")
    reasoning_tokens = _response_field(completion_details, "reasoning_tokens")
    reasoning_present = bool(
        _response_field(message, "reasoning")
        or _response_field(message, "reasoning_content")
    )

    logger.warning(
        "Provider returned no visible completion: provider=%s model=%s response_model=%s "
        "finish_reason=%s completion_tokens=%s reasoning_tokens=%s reasoning_present=%s",
        model_cfg.provider,
        model_cfg.model_id,
        str(_response_field(response, "model") or "unknown")[:160],
        finish_reason,
        completion_tokens if isinstance(completion_tokens, (int, float)) else "unknown",
        reasoning_tokens if isinstance(reasoning_tokens, (int, float)) else "unknown",
        reasoning_present,
    )
    raise ProviderEmptyCompletionError(finish_reason)


async def complete(
    model_cfg: ModelConfig,
    api_key: str,
    request: dict[str, Any],
) -> Any:
    """Run one non-streaming completion through the selected direct adapter."""
    if model_cfg.provider == "google":
        client, types = _gemini_client(api_key)
        async_client = client.aio
        try:
            contents, system_instruction = _gemini_contents(request["messages"], types)
            config = _gemini_config(request, system_instruction, types)
            response = await _call_with_retry(
                model_cfg.provider,
                lambda: async_client.models.generate_content(
                    model=_provider_model_id(model_cfg),
                    contents=contents,
                    config=config,
                ),
            )
            result = _gemini_response(response, _provider_model_id(model_cfg))
            _validate_completion_response(result, model_cfg)
            return result
        finally:
            await _close_gemini_clients(client, async_client)

    client = _openai_client(model_cfg, api_key)
    try:
        kwargs = _build_openai_kwargs(model_cfg, request)
        response = await _call_with_retry(
            model_cfg.provider,
            lambda: client.chat.completions.create(**kwargs),
        )
        _validate_completion_response(response, model_cfg)
        return response
    finally:
        await client.close()


async def _openai_stream_events(
    provider: str,
    client: Any,
    response_stream: Any,
    first_chunk: Any | None,
) -> AsyncIterator[str]:
    emitted = False
    try:
        if first_chunk is not None:
            emitted = True
            yield _openai_chunk_sse(first_chunk)
        async for chunk in response_stream:
            emitted = True
            yield _openai_chunk_sse(chunk)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _log_provider_failure(
            provider,
            exc,
            attempt=1,
            retrying=False,
            stream_started=emitted,
        )
        error = _to_provider_error(exc)
        yield _sse({"error": {"message": str(error), "type": error.code, "code": error.code}})
    finally:
        await _close_async_resource(response_stream)
        await client.close()
    yield "data: [DONE]\n\n"


async def _gemini_stream_events(
    client: Any,
    async_client: Any,
    response_stream: Any,
    first_chunk: Any | None,
    *,
    model: str,
    stream_id: str,
    created: int,
) -> AsyncIterator[str]:
    emitted = False
    try:
        if first_chunk is not None:
            payload = _gemini_chunk_payload(
                first_chunk,
                model=model,
                stream_id=stream_id,
                created=created,
            )
            if payload is not None:
                emitted = True
                yield _sse(payload)
        async for chunk in response_stream:
            payload = _gemini_chunk_payload(
                chunk,
                model=model,
                stream_id=stream_id,
                created=created,
            )
            if payload is not None:
                emitted = True
                yield _sse(payload)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _log_provider_failure(
            "google",
            exc,
            attempt=1,
            retrying=False,
            stream_started=emitted,
        )
        error = _to_provider_error(exc)
        yield _sse({"error": {"message": str(error), "type": error.code, "code": error.code}})
    finally:
        await _close_gemini_clients(client, async_client)
    yield "data: [DONE]\n\n"


async def prepare_stream(
    model_cfg: ModelConfig,
    api_key: str,
    request: dict[str, Any],
) -> AsyncIterator[str]:
    """Open and prime the upstream stream before HTTP headers are sent.

    Failures before the first upstream chunk become normal HTTP 4xx/5xx
    responses in the Gateway route. Once downstream streaming has started,
    failures become a sanitized SSE error and are never replayed.
    """
    if model_cfg.provider == "google":
        client, types = _gemini_client(api_key)
        async_client = client.aio
        try:
            contents, system_instruction = _gemini_contents(request["messages"], types)
            config = _gemini_config(request, system_instruction, types)
            first_chunk = None
            response_stream = None
            for attempt in range(1, MAX_PROVIDER_RETRIES + 2):
                try:
                    response_stream = await async_client.models.generate_content_stream(
                        model=_provider_model_id(model_cfg),
                        contents=contents,
                        config=config,
                    )
                    iterator = response_stream.__aiter__()
                    first_chunk = await anext(iterator, None)
                    response_stream = iterator
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Close a partially opened stream before retrying; otherwise
                    # transient first-chunk failures can leak HTTP connections.
                    await _close_async_resource(response_stream)
                    response_stream = None
                    retrying = attempt <= MAX_PROVIDER_RETRIES and _is_retryable(exc)
                    _log_provider_failure(
                        model_cfg.provider,
                        exc,
                        attempt=attempt,
                        retrying=retrying,
                    )
                    if not retrying:
                        raise _to_provider_error(exc) from exc
                    await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))
            return _gemini_stream_events(
                client,
                async_client,
                response_stream,
                first_chunk,
                model=_provider_model_id(model_cfg),
                stream_id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
            )
        except Exception:
            await _close_gemini_clients(client, async_client)
            raise

    client = _openai_client(model_cfg, api_key)
    kwargs = _build_openai_kwargs(model_cfg, request)
    response_stream = None
    first_chunk = None
    try:
        kwargs["stream"] = True
        for attempt in range(1, MAX_PROVIDER_RETRIES + 2):
            try:
                response_stream = await client.chat.completions.create(**kwargs)
                iterator = response_stream.__aiter__()
                first_chunk = await anext(iterator, None)
                response_stream = iterator
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await _close_async_resource(response_stream)
                response_stream = None
                retrying = attempt <= MAX_PROVIDER_RETRIES and _is_retryable(exc)
                _log_provider_failure(
                    model_cfg.provider,
                    exc,
                    attempt=attempt,
                    retrying=retrying,
                )
                if not retrying:
                    raise _to_provider_error(exc) from exc
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))
        return _openai_stream_events(model_cfg.provider, client, response_stream, first_chunk)
    except Exception:
        await client.close()
        raise


async def stream(
    model_cfg: ModelConfig,
    api_key: str,
    request: dict[str, Any],
) -> AsyncIterator[str]:
    """Backward-compatible async generator wrapper around prepare_stream."""
    events = await prepare_stream(model_cfg, api_key, request)
    async for event in events:
        yield event
