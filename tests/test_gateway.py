import json

import pytest
from fastapi.testclient import TestClient

from app.config import model_config_for_user, settings
from app.main import app
from app.models import SessionCreateRequest


client = TestClient(app)


def test_health():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert isinstance(response.json()["data"], list)


def test_providers():
    response = client.get("/v1/providers")
    assert response.status_code == 200
    assert response.json()["providers"]


def test_provider_catalog_includes_configured_model_ids():
    response = client.get("/v1/providers")
    assert response.status_code == 200
    providers = response.json()["providers"]
    openai_model = next(
        item for item in providers
        if item["provider"] == "openai"
    )
    assert openai_model["model_id"] == "gpt-4o-mini"
    assert openai_model["recommended"] is True


def test_models_endpoint_exposes_configured_provider_model_id():
    response = client.get("/v1/models")
    assert response.status_code == 200
    openai_model = next(
        item for item in response.json()["data"]
        if item["owned_by"] == "openai"
    )
    assert openai_model["model_id"] == "gpt-4o-mini"


def test_portfolio_session_defaults_to_approved_registry():
    with pytest.raises(ValueError, match="not enabled in the Gateway model registry"):
        model_config_for_user(
            "openai",
            "gpt-4.1-mini",
            allowed_models=settings.models,
        )


def test_portfolio_byoK_custom_model_requires_explicit_opt_in():
    payload = SessionCreateRequest(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-provider-key",
    )
    assert payload.allow_custom_model is False

    with pytest.raises(ValueError, match="not enabled in the Gateway model registry"):
        model_config_for_user(
            payload.provider,
            payload.model,
            allowed_models=settings.models if not payload.allow_custom_model else None,
        )

    custom_payload = SessionCreateRequest(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="test-provider-key",
        allow_custom_model=True,
    )
    assert custom_payload.allow_custom_model is True
    resolved = model_config_for_user(
        custom_payload.provider,
        custom_payload.model,
        allowed_models=settings.models if not custom_payload.allow_custom_model else None,
    )
    assert resolved.provider == "openai"
    assert resolved.model_id == "gpt-4.1-mini"


def test_custom_model_opt_in_does_not_bypass_provider_or_identifier_validation():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        model_config_for_user("unsupported", "model-name")

    with pytest.raises(ValueError, match="must not contain a URL"):
        model_config_for_user("openai", "https://example.invalid/model")
