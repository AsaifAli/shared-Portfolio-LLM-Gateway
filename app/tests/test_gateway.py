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
