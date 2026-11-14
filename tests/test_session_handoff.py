import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security import decode_jwt, store


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_memory_store(monkeypatch):
    monkeypatch.setattr(store, "_redis", None)
    store._memory.clear()
    yield
    store._memory.clear()


def create_portfolio_session() -> str:
    response = client.post(
        "/v1/demo/session",
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
            "ttl_seconds": 600,
        },
    )

    assert response.status_code == 200
    return response.json()["access_token"]


def issue_handoff(token: str, project: str = "evidenceflow") -> str:
    response = client.post(
        "/v1/demo/session/handoff",
        headers={"Authorization": f"Bearer {token}"},
        json={"project": project},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["expires_in"] == 60
    assert payload["project"] == project
    assert "access_token" not in payload
    return payload["code"]


def test_handoff_requires_a_valid_bearer_session():
    response = client.post(
        "/v1/demo/session/handoff",
        json={"project": "evidenceflow"},
    )
    assert response.status_code == 401


def test_handoff_exchange_is_single_use():
    original_token = create_portfolio_session()
    code = issue_handoff(original_token)

    assert code != original_token
    assert "." not in code

    response = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "evidenceflow"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_project"] == "evidenceflow"
    assert payload["access_token"]
    assert payload["access_token"] != original_token

    claims = decode_jwt(payload["access_token"])
    assert claims["project"] == "evidenceflow"
    assert claims["session_project"] == "portfolio"

    status = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
    )
    assert status.status_code == 200
    assert status.json()["active"] is True
    assert status.json()["project"] == "evidenceflow"
    assert status.json()["session_project"] == "portfolio"

    chained = client.post(
        "/v1/demo/session/handoff",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
        json={"project": "quotesense"},
    )
    assert chained.status_code == 401

    replay = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "evidenceflow"},
    )
    assert replay.status_code == 401


def test_handoff_rejects_unapproved_target_project():
    token = create_portfolio_session()
    response = client.post(
        "/v1/demo/session/handoff",
        headers={"Authorization": f"Bearer {token}"},
        json={"project": "admin"},
    )
    assert response.status_code == 400


def test_handoff_code_is_bound_to_target_project():
    token = create_portfolio_session()
    code = issue_handoff(token, "evidenceflow")

    wrong_project = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "quotesense"},
    )
    assert wrong_project.status_code == 401

    correct_project = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "evidenceflow"},
    )
    assert correct_project.status_code == 200


def test_revoked_session_cannot_be_exchanged():
    token = create_portfolio_session()
    code = issue_handoff(token)

    revoked = client.post(
        "/v1/demo/session/revoke",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert revoked.status_code == 200

    exchanged = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "evidenceflow"},
    )
    assert exchanged.status_code == 401


def test_memory_store_consume_is_single_use():
    async def run():
        await store.set("test:handoff:single-use", "value", 60)
        first = await store.consume("test:handoff:single-use")
        second = await store.consume("test:handoff:single-use")
        return first, second

    import asyncio
    first, second = asyncio.run(run())

    assert first == "value"
    assert second is None
