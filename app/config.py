from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from cryptography.fernet import Fernet


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    litellm_model: str
    recommended: bool = False
    label: str | None = None


ALLOWED_HANDOFF_PROJECTS = frozenset({"evidenceflow", "quotesense"})


DEFAULT_MODELS = [
    ModelConfig(
        "openai-default",
        "openai",
        "openai/gpt-4o-mini",
        True,
        "OpenAI recommended",
    ),
    ModelConfig(
        "gemini-default",
        "google",
        "gemini/gemini-2.5-flash",
        True,
        "Gemini recommended",
    ),
    ModelConfig(
        "anthropic-default",
        "anthropic",
        "anthropic/claude-3-5-haiku-latest",
        False,
        "Anthropic",
    ),
    ModelConfig(
        "openrouter-default",
        "openrouter",
        "openrouter/openai/gpt-oss-20b:free",
        False,
        "OpenRouter (configure/verify)",
    ),
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

    # OpenRouter supports provider-qualified upstream model IDs.
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


def _read_int(name: str, default: str) -> int:
    """Read an integer environment variable with a clear configuration error."""
    raw_value = os.getenv(name, default)

    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a valid integer") from exc


class Settings:
    def __init__(self) -> None:
        self.app_name = os.getenv(
            "APP_NAME",
            "Portfolio LLM Gateway",
        ).strip()

        # Normalize accepted environment aliases and reject unknown values.
        environment = os.getenv(
            "ENVIRONMENT",
            "development",
        ).strip().lower()

        environment_aliases = {
            "dev": "development",
            "prod": "production",
        }

        environment = environment_aliases.get(
            environment,
            environment,
        )

        allowed_environments = {
            "development",
            "test",
            "production",
        }

        if environment not in allowed_environments:
            raise RuntimeError(
                "Invalid ENVIRONMENT. Use development, test or production."
            )

        self.environment = environment

        self.host = os.getenv("HOST", "0.0.0.0").strip()
        self.port = _read_int("PORT", "8000")

        # JWT and encryption configuration.
        self.jwt_secret = os.getenv(
            "GATEWAY_JWT_SECRET",
            "",
        ).strip()

        # Development and test environments receive an ephemeral secret
        # when one is not explicitly supplied.
        if not self.jwt_secret and self.environment != "production":
            self.jwt_secret = secrets.token_urlsafe(48)

        self.encryption_key = os.getenv(
            "GATEWAY_ENCRYPTION_KEY",
            "",
        ).strip()

        self.admin_token = os.getenv(
            "GATEWAY_ADMIN_TOKEN",
            "",
        ).strip()

        # Session and rate-limit configuration.
        self.session_ttl_seconds = _read_int(
            "SESSION_TTL_SECONDS",
            "1800",
        )

        self.max_requests_per_minute = _read_int(
            "MAX_REQUESTS_PER_MINUTE",
            "12",
        )

        self.max_session_creates_per_minute = _read_int(
            "MAX_SESSION_CREATES_PER_MINUTE",
            "6",
        )

        # Request and model-output limits.
        self.max_input_bytes = _read_int(
            "MAX_INPUT_BYTES",
            str(1024 * 1024),
        )

        self.max_output_tokens = _read_int(
            "MAX_OUTPUT_TOKENS",
            "4096",
        )

        self.provider_timeout_seconds = _read_int(
            "PROVIDER_TIMEOUT_SECONDS",
            "90",
        )

        # CORS configuration.
        cors_origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:8501",
        )

        self.allowed_origins = [
            origin.strip()
            for origin in cors_origins.split(",")
            if origin.strip()
        ]

        self.allow_origin_regex = os.getenv(
            "CORS_ALLOW_ORIGIN_REGEX",
            "",
        ).strip()

        # Upstash Redis configuration.
        self.redis_url = os.getenv(
            "UPSTASH_REDIS_REST_URL",
            "",
        ).strip()

        self.redis_token = os.getenv(
            "UPSTASH_REDIS_REST_TOKEN",
            "",
        ).strip()

        # Optional server-side fallback provider configuration.
        self.fallback_provider = os.getenv(
            "FALLBACK_PROVIDER",
            "",
        ).strip().lower()

        self.fallback_model_id = os.getenv(
            "FALLBACK_MODEL_ID",
            "",
        ).strip()

        self.fallback_api_key = os.getenv(
            "FALLBACK_API_KEY",
            "",
        ).strip()

        self.models = self._load_models()

    def _load_models(self) -> dict[str, ModelConfig]:
        raw = os.getenv("MODEL_REGISTRY_JSON", "").strip()

        if not raw:
            return {model.id: model for model in DEFAULT_MODELS}

        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "MODEL_REGISTRY_JSON must be valid JSON"
            ) from exc

        if not isinstance(data, list) or not data:
            raise RuntimeError(
                "MODEL_REGISTRY_JSON must be a non-empty JSON list"
            )

        models: dict[str, ModelConfig] = {}

        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "Each MODEL_REGISTRY_JSON item must be an object"
                )

            try:
                model = ModelConfig(
                    id=str(item["id"]),
                    provider=str(item["provider"]),
                    litellm_model=str(item["litellm_model"]),
                    recommended=bool(item.get("recommended", False)),
                    label=(
                        str(item["label"])
                        if item.get("label")
                        else None
                    ),
                )
            except KeyError as exc:
                raise RuntimeError(
                    "Each model must define id, provider and litellm_model"
                ) from exc

            if not model.id.strip():
                raise RuntimeError("Model ID cannot be empty")

            if model.id in models:
                raise RuntimeError(
                    f"Duplicate model ID in MODEL_REGISTRY_JSON: {model.id}"
                )

            models[model.id] = model

        return models

    def _validate_numeric_settings(self) -> None:
        if not 1 <= self.port <= 65535:
            raise RuntimeError("PORT must be between 1 and 65535")

        if not 60 <= self.session_ttl_seconds <= 3600:
            raise RuntimeError(
                "SESSION_TTL_SECONDS must be between 60 and 3600"
            )

        if not 1 <= self.max_requests_per_minute <= 10000:
            raise RuntimeError(
                "MAX_REQUESTS_PER_MINUTE must be between 1 and 10000"
            )

        if not 1 <= self.max_session_creates_per_minute <= 10000:
            raise RuntimeError(
                "MAX_SESSION_CREATES_PER_MINUTE must be between 1 and 10000"
            )

        if not 1024 <= self.max_input_bytes <= 10 * 1024 * 1024:
            raise RuntimeError(
                "MAX_INPUT_BYTES must be between 1024 and 10485760"
            )

        if not 1 <= self.max_output_tokens <= 32768:
            raise RuntimeError(
                "MAX_OUTPUT_TOKENS must be between 1 and 32768"
            )

        if not 1 <= self.provider_timeout_seconds <= 300:
            raise RuntimeError(
                "PROVIDER_TIMEOUT_SECONDS must be between 1 and 300"
            )

    def _validate_encryption_key(self) -> None:
        if not self.encryption_key:
            return

        try:
            Fernet(self.encryption_key.encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "GATEWAY_ENCRYPTION_KEY must be a valid Fernet key"
            ) from exc

    def _validate_redis_configuration(self) -> None:
        has_url = bool(self.redis_url)
        has_token = bool(self.redis_token)

        if has_url != has_token:
            raise RuntimeError(
                "UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN "
                "must be configured together"
            )

    def _validate_fallback_configuration(self) -> None:
        fallback_values = (
            self.fallback_provider,
            self.fallback_model_id,
            self.fallback_api_key,
        )

        if any(fallback_values) and not all(fallback_values):
            raise RuntimeError(
                "FALLBACK_PROVIDER, FALLBACK_MODEL_ID and FALLBACK_API_KEY "
                "must be configured together"
            )

        if not any(fallback_values):
            return

        if self.fallback_provider not in SUPPORTED_PROVIDERS:
            raise RuntimeError(
                "FALLBACK_PROVIDER must be a supported provider"
            )

        try:
            normalize_user_model(
                self.fallback_provider,
                self.fallback_model_id,
            )
        except ValueError as exc:
            raise RuntimeError(
                "FALLBACK_MODEL_ID is invalid for the configured provider"
            ) from exc

    def _validate_production_cors(self) -> None:
        if not self.allowed_origins:
            raise RuntimeError(
                "CORS_ORIGINS must contain explicit production origins"
            )

        if self.allow_origin_regex:
            raise RuntimeError(
                "CORS origin regex (CORS_ALLOW_ORIGIN_REGEX) is not allowed in production"
            )

        for origin in self.allowed_origins:
            parsed = urlsplit(origin)

            try:
                parsed_port = parsed.port
            except ValueError as exc:
                raise RuntimeError(
                    "Production CORS_ORIGINS contains an invalid port"
                ) from exc

            if parsed_port == 0:
                raise RuntimeError(
                    "Production CORS_ORIGINS contains an invalid port"
                )

            hostname = (parsed.hostname or "").lower()

            if (
                parsed.scheme != "https"
                or not hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or "*" in parsed.netloc
                or any(character.isspace() for character in origin)
                or hostname in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
            ):
                raise RuntimeError(
                    "Production CORS_ORIGINS must contain explicit "
                    "HTTPS origins without paths, wildcards or localhost"
                )

    def validate_runtime(self) -> None:
        # Validate values shared across environments.
        if len(self.jwt_secret) < 32:
            raise RuntimeError(
                "GATEWAY_JWT_SECRET must be at least 32 characters"
            )

        self._validate_numeric_settings()
        self._validate_encryption_key()
        self._validate_redis_configuration()
        self._validate_fallback_configuration()

        # Production must use persistent, encrypted infrastructure.
        if self.environment == "production":
            if not self.encryption_key:
                raise RuntimeError(
                    "GATEWAY_ENCRYPTION_KEY is required in production"
                )

            if not self.redis_url or not self.redis_token:
                raise RuntimeError(
                    "Redis credentials (UPSTASH_REDIS_REST_URL/TOKEN) are required in production"
                )

            self._validate_production_cors()


settings = Settings()