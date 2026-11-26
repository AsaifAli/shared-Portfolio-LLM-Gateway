

import pytest
from cryptography.fernet import Fernet

from app.config import Settings


def configure_valid_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")

    monkeypatch.setenv(
        "GATEWAY_JWT_SECRET",
        "production-test-secret-" + ("x" * 40),
    )

    monkeypatch.setenv(
        "GATEWAY_ENCRYPTION_KEY",
        Fernet.generate_key().decode(),
    )

    monkeypatch.setenv(
        "UPSTASH_REDIS_REST_URL",
        "https://test-database.upstash.io",
    )

    monkeypatch.setenv(
        "UPSTASH_REDIS_REST_TOKEN",
        "dummy-test-token",
    )

    monkeypatch.setenv(
        "CORS_ORIGINS",
        "https://app.example.com",
    )

    monkeypatch.setenv("CORS_ALLOW_ORIGIN_REGEX", "")

    monkeypatch.setenv("SESSION_TTL_SECONDS", "1800")
    monkeypatch.setenv("MAX_REQUESTS_PER_MINUTE", "60")
    monkeypatch.setenv("MAX_SESSION_CREATES_PER_MINUTE", "6")
    monkeypatch.setenv("MAX_INPUT_BYTES", "1048576")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("PROVIDER_TIMEOUT_SECONDS", "90")
    monkeypatch.delenv("WORKSPACE_IDENTITY_SECRET", raising=False)

    for name in (
        "FALLBACK_PROVIDER",
        "FALLBACK_MODEL_ID",
        "FALLBACK_API_KEY",
        "MODEL_REGISTRY_JSON",
    ):
        monkeypatch.delenv(name, raising=False)


def test_valid_production_configuration(monkeypatch):
    configure_valid_production(monkeypatch)

    config = Settings()
    config.validate_runtime()


def test_unknown_environment_is_rejected(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "prodution")

    with pytest.raises(RuntimeError, match="Invalid ENVIRONMENT"):
        Settings()


def test_production_requires_redis(monkeypatch):
    configure_valid_production(monkeypatch)

    monkeypatch.delenv("UPSTASH_REDIS_REST_URL")
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN")

    with pytest.raises(RuntimeError, match="Redis"):
        Settings().validate_runtime()


def test_redis_credentials_must_be_paired(monkeypatch):
    configure_valid_production(monkeypatch)

    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN")

    with pytest.raises(RuntimeError, match="configured together"):
        Settings().validate_runtime()


def test_production_rejects_wildcard_cors(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv("CORS_ORIGINS", "*")

    with pytest.raises(RuntimeError, match="HTTPS origins"):
        Settings().validate_runtime()


def test_production_rejects_http_origin(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "http://app.example.com",
    )

    with pytest.raises(RuntimeError, match="HTTPS origins"):
        Settings().validate_runtime()


def test_production_rejects_origin_with_path(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "https://app.example.com/dashboard",
    )

    with pytest.raises(RuntimeError, match="HTTPS origins"):
        Settings().validate_runtime()


def test_production_rejects_origin_regex(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGIN_REGEX",
        ".*",
    )

    with pytest.raises(RuntimeError, match="regex"):
        Settings().validate_runtime()


def test_production_rejects_invalid_encryption_key(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv(
        "GATEWAY_ENCRYPTION_KEY",
        "invalid-fernet-key",
    )

    with pytest.raises(RuntimeError, match="Fernet"):
        Settings().validate_runtime()


def test_supabase_identity_configuration_is_optional_when_unset(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)

    Settings().validate_runtime()


def test_supabase_identity_requires_both_settings(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY"):
        Settings().validate_runtime()


def test_supabase_identity_requires_https_project_origin(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "http://project.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")

    with pytest.raises(RuntimeError, match="HTTPS project origin"):
        Settings().validate_runtime()

def test_workspace_identity_secret_is_optional_until_workspace_is_enabled(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.delenv("WORKSPACE_IDENTITY_SECRET", raising=False)

    config = Settings()
    config.validate_runtime()
    assert config.workspace_identity_secret == ""


def test_workspace_identity_secret_rejects_short_values(monkeypatch):
    configure_valid_production(monkeypatch)
    monkeypatch.setenv("WORKSPACE_IDENTITY_SECRET", "too-short")

    with pytest.raises(RuntimeError, match="WORKSPACE_IDENTITY_SECRET"):
        Settings()
