from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    litellm_model: str
    recommended: bool = False
    label: str | None = None


DEFAULT_MODELS = [
    ModelConfig("openai-default", "openai", "openai/gpt-4o-mini", True, "OpenAI recommended"),
    ModelConfig("gemini-default", "google", "gemini/gemini-2.5-flash", True, "Gemini recommended"),
    ModelConfig("anthropic-default", "anthropic", "anthropic/claude-3-5-haiku-latest", False, "Anthropic"),
    ModelConfig("openrouter-default", "openrouter", "openrouter/openai/gpt-oss-20b:free", False, "OpenRouter (configure/verify)"),
]


SUPPORTED_PROVIDERS = {
    "openai": "OpenAI",
    "google": "Google Gemini",
    "anthropic": "Anthropic",
    "openrouter": "OpenRouter",
}


def normalize_user_model(provider: str, model_id: str) -> str:
    """Normalize a user-selected model into a LiteLLM provider/model identifier."""
    provider = provider.strip().lower()
    model_id = model_id.strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError("Unsupported LLM provider")
    if not model_id or len(model_id) > 200:
        raise ValueError("Invalid model identifier")
    if any(ch in model_id for ch in ("\n", "\r", "\t")):
        raise ValueError("Invalid model identifier")
    if model_id.startswith(("http://", "https://")):
        raise ValueError("Model identifier must not contain a URL")

    prefix = {
        "openai": "openai/",
        "google": "gemini/",
        "anthropic": "anthropic/",
        "openrouter": "openrouter/",
    }[provider]
    if model_id.startswith(prefix):
        return model_id
    # OpenRouter takes the provider-qualified upstream model ID, e.g.
    # openrouter + openai/gpt-oss-20b:free -> openrouter/openai/gpt-oss-20b:free
    return f"{prefix}{model_id}"


def model_config_for_user(provider: str, model_id: str) -> ModelConfig:
    provider = provider.strip().lower()
    normalized = normalize_user_model(provider, model_id)
    return ModelConfig(
        id=model_id.strip(),
        provider=provider,
        litellm_model=normalized,
        recommended=False,
        label=f"{SUPPORTED_PROVIDERS[provider]} custom model",
    )


class Settings:
    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "Portfolio LLM Gateway")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))

        self.jwt_secret = os.getenv("GATEWAY_JWT_SECRET", "")
        self.encryption_key = os.getenv("GATEWAY_ENCRYPTION_KEY", "")
        self.admin_token = os.getenv("GATEWAY_ADMIN_TOKEN", "")

        self.session_ttl_seconds = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "12"))
        self.max_session_creates_per_minute = int(os.getenv("MAX_SESSION_CREATES_PER_MINUTE", "6"))
        self.max_input_bytes = int(os.getenv("MAX_INPUT_BYTES", str(1024 * 1024)))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
        self.allowed_origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",") if x.strip()]
        self.allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "")

        self.redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "")
        self.redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

        self.fallback_provider = os.getenv("FALLBACK_PROVIDER", "").strip()
        self.fallback_model_id = os.getenv("FALLBACK_MODEL_ID", "").strip()
        self.fallback_api_key = os.getenv("FALLBACK_API_KEY", "").strip()

        self.models = self._load_models()

    def _load_models(self) -> dict[str, ModelConfig]:
        raw = os.getenv("MODEL_REGISTRY_JSON", "").strip()
        if not raw:
            return {m.id: m for m in DEFAULT_MODELS}
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MODEL_REGISTRY_JSON must be valid JSON") from exc
        if not isinstance(data, list) or not data:
            raise RuntimeError("MODEL_REGISTRY_JSON must be a non-empty JSON list")

        models: dict[str, ModelConfig] = {}
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("Each MODEL_REGISTRY_JSON item must be an object")
            model = ModelConfig(
                id=str(item["id"]),
                provider=str(item["provider"]),
                litellm_model=str(item["litellm_model"]),
                recommended=bool(item.get("recommended", False)),
                label=str(item["label"]) if item.get("label") else None,
            )
            models[model.id] = model
        return models

    def validate_runtime(self) -> None:
        if self.environment.lower() in {"production", "prod"}:
            if len(self.jwt_secret) < 32:
                raise RuntimeError("GATEWAY_JWT_SECRET must be at least 32 characters in production")
            if not self.encryption_key:
                raise RuntimeError("GATEWAY_ENCRYPTION_KEY is required in production")
            if not self.redis_url or not self.redis_token:
                raise RuntimeError("UPSTASH_REDIS_REST_URL/TOKEN are required in production")


settings = Settings()
