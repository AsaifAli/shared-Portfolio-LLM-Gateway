import json

from fastapi.testclient import TestClient

from app.main import app


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
