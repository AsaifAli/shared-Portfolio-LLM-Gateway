import pytest
from fastapi.testclient import TestClient

import app.main as main_module
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

def test_flowpilot_handoff_is_project_bound_and_single_use():
    token = create_portfolio_session()
    code = issue_handoff(token, "flowpilot")

    wrong_project = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "quotesense"},
    )
    assert wrong_project.status_code == 401

    exchanged = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "flowpilot"},
    )
    assert exchanged.status_code == 200
    payload = exchanged.json()
    assert payload["target_project"] == "flowpilot"

    claims = decode_jwt(payload["access_token"])
    assert claims["project"] == "flowpilot"
    assert claims["session_project"] == "portfolio"

    replay = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "flowpilot"},
    )
    assert replay.status_code == 401



def test_portfolio_session_rejects_model_outside_gateway_registry():
    response = client.post(
        "/v1/demo/session",
        json={
            "provider": "openai",
            "model": "gpt-4.1",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
        },
    )
    assert response.status_code == 400
    assert "model registry" in response.json()["detail"].lower()


def test_project_scoped_session_can_use_valid_custom_model_id():
    response = client.post(
        "/v1/demo/session",
        json={
            "provider": "openai",
            "model": "gpt-4.1",
            "api_key": "test-api-key-123456",
            "project": "flowpilot",
            "ttl_seconds": 600,
        },
    )
    assert response.status_code == 200
    assert response.json()["model"] == "gpt-4.1"


def test_unauthenticated_byoK_sessions_are_isolated_by_owner_scope():
    first = create_portfolio_session()
    second = create_portfolio_session()

    first_status = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {first}"},
    )
    second_status = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {second}"},
    )

    assert first_status.status_code == 200
    assert second_status.status_code == 200
    first_scope = first_status.json()["owner_scope"]
    second_scope = second_status.json()["owner_scope"]
    assert len(first_scope) == 64
    assert first_scope != second_scope


def test_supabase_authenticated_sessions_share_owner_scope(monkeypatch):

    user_id = "8cce3a1c-4f9f-4f42-9cbd-e77efed3a619"

    async def verify(_token):
        return user_id

    monkeypatch.setattr(main_module, "verify_supabase_access_token", verify)

    headers = {"Authorization": "Bearer supabase-access-token"}
    first = client.post(
        "/v1/demo/session",
        headers=headers,
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
            "ttl_seconds": 600,
        },
    )
    second = client.post(
        "/v1/demo/session",
        headers=headers,
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
            "ttl_seconds": 600,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200

    first_status = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {first.json()['access_token']}"},
    )
    second_status = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {second.json()['access_token']}"},
    )
    assert first_status.json()["owner_scope"] == second_status.json()["owner_scope"]


def test_invalid_supabase_bearer_does_not_downgrade_to_anonymous(monkeypatch):
    from fastapi import HTTPException

    async def reject(_token):
        raise HTTPException(status_code=401, detail="Invalid Supabase access token")

    monkeypatch.setattr(main_module, "verify_supabase_access_token", reject)
    response = client.post(
        "/v1/demo/session",
        headers={"Authorization": "Bearer invalid-supabase-token"},
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
        },
    )
    assert response.status_code == 401


def test_handoff_preserves_owner_scope():
    original = create_portfolio_session()
    original_status = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {original}"},
    )
    owner_scope = original_status.json()["owner_scope"]

    code = issue_handoff(original, "evidenceflow")
    exchange = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "evidenceflow"},
    )
    assert exchange.status_code == 200

    target_status = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {exchange.json()['access_token']}"},
    )
    assert target_status.status_code == 200
    assert target_status.json()["owner_scope"] == owner_scope



def test_supabase_access_token_is_verified_server_side_and_only_user_id_is_used(monkeypatch):
    import asyncio

    import app.security as security

    class FakeResponse:
        status_code = 200
        content = b'{"id":"8cce3a1c-4f9f-4f42-9cbd-e77efed3a619"}'

        def json(self):
            return {"id": "8cce3a1c-4f9f-4f42-9cbd-e77efed3a619"}

    class FakeAsyncClient:
        options = None
        request = None

        def __init__(self, **kwargs):
            self.options = kwargs

        async def __aenter__(self):
            FakeAsyncClient.options = self.options
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers):
            FakeAsyncClient.request = (url, headers)
            return FakeResponse()

    monkeypatch.setattr(security.settings, "supabase_url", "https://project.supabase.co")
    monkeypatch.setattr(security.settings, "supabase_publishable_key", "sb_publishable_test")
    monkeypatch.setattr(security.httpx, "AsyncClient", FakeAsyncClient)

    user_id = asyncio.run(security.verify_supabase_access_token("supabase-access-token"))

    assert user_id == "8cce3a1c-4f9f-4f42-9cbd-e77efed3a619"
    assert FakeAsyncClient.options["follow_redirects"] is False
    assert FakeAsyncClient.options["trust_env"] is False
    assert FakeAsyncClient.request[0] == "https://project.supabase.co/auth/v1/user"
    assert FakeAsyncClient.request[1]["Authorization"] == "Bearer supabase-access-token"
    assert FakeAsyncClient.request[1]["apikey"] == "sb_publishable_test"


def test_supabase_identity_redirect_is_rejected(monkeypatch):
    import asyncio

    import app.security as security

    class FakeResponse:
        status_code = 302
        content = b""

        def json(self):
            return {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(security.settings, "supabase_url", "https://project.supabase.co")
    monkeypatch.setattr(security.settings, "supabase_publishable_key", "sb_publishable_test")
    monkeypatch.setattr(security.httpx, "AsyncClient", FakeAsyncClient)

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        asyncio.run(security.verify_supabase_access_token("supabase-access-token"))
    assert error.value.status_code == 503



def test_supabase_identity_url_validation_rejects_arbitrary_hosts():
    from types import SimpleNamespace

    from app.config import Settings

    settings = SimpleNamespace(
        supabase_url="https://attacker.example",
        supabase_publishable_key="publishable-test",
    )
    with pytest.raises(RuntimeError, match="HTTPS project origin"):
        Settings._validate_supabase_configuration(settings)

def _workspace_identity_headers(client_id: str, secret: str) -> dict[str, str]:
    import hashlib
    import hmac
    import time

    timestamp = str(int(time.time()))
    payload = f"studio-workspace-v1\\n{client_id}\\n{timestamp}".encode()
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return {
        "X-Studio-Workspace-Client-Id": client_id,
        "X-Studio-Workspace-Timestamp": timestamp,
        "X-Studio-Workspace-Signature": signature,
    }


def test_workspace_client_sessions_keep_a_stable_owner_scope(monkeypatch):
    import uuid

    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(main_module.settings, "workspace_identity_secret", secret)
    client_id = str(uuid.uuid4())

    first = client.post(
        "/v1/demo/session",
        headers=_workspace_identity_headers(client_id, secret),
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
            "ttl_seconds": 600,
        },
    )
    second = client.post(
        "/v1/demo/session",
        headers=_workspace_identity_headers(client_id, secret),
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
            "ttl_seconds": 600,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200

    first_status = client.get(
        "/v1/demo/session/status",
        headers={**_workspace_identity_headers(client_id, secret), "Authorization": f"Bearer {first.json()['access_token']}"},
    )
    second_status = client.get(
        "/v1/demo/session/status",
        headers={**_workspace_identity_headers(client_id, secret), "Authorization": f"Bearer {second.json()['access_token']}"},
    )
    assert first_status.status_code == second_status.status_code == 200
    assert first_status.json()["owner_scope"] == second_status.json()["owner_scope"]


def test_workspace_client_sessions_are_isolated_from_each_other(monkeypatch):
    import uuid

    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(main_module.settings, "workspace_identity_secret", secret)

    scopes = []
    for _ in range(2):
        client_id = str(uuid.uuid4())
        identity_headers = _workspace_identity_headers(client_id, secret)
        response = client.post(
            "/v1/demo/session",
            headers=identity_headers,
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-api-key-123456",
                "project": "portfolio",
                "ttl_seconds": 600,
            },
        )
        assert response.status_code == 200
        status = client.get(
            "/v1/demo/session/status",
            headers={**_workspace_identity_headers(client_id, secret), "Authorization": f"Bearer {response.json()['access_token']}"},
        )
        assert status.status_code == 200
        scopes.append(status.json()["owner_scope"])

    assert scopes[0] != scopes[1]


def test_workspace_identity_headers_are_rejected_when_signature_is_invalid(monkeypatch):
    import uuid

    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(main_module.settings, "workspace_identity_secret", secret)
    client_id = str(uuid.uuid4())
    headers = _workspace_identity_headers(client_id, secret)
    headers["X-Studio-Workspace-Signature"] = "0" * 64

    response = client.post(
        "/v1/demo/session",
        headers=headers,
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
        },
    )
    assert response.status_code == 401


def test_workspace_identity_cannot_be_combined_with_supabase_identity(monkeypatch):
    import uuid

    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(main_module.settings, "workspace_identity_secret", secret)
    headers = _workspace_identity_headers(str(uuid.uuid4()), secret)
    headers["Authorization"] = "Bearer supabase-token"

    response = client.post(
        "/v1/demo/session",
        headers=headers,
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
        },
    )
    assert response.status_code == 400



def _workspace_identity_headers(client_id: str, secret: str) -> dict[str, str]:
    import hashlib
    import hmac
    import time

    timestamp = str(int(time.time()))
    payload = f"studio-workspace-v1\n{client_id}\n{timestamp}".encode()
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return {
        "X-Studio-Workspace-Client-Id": client_id,
        "X-Studio-Workspace-Timestamp": timestamp,
        "X-Studio-Workspace-Signature": signature,
    }


def test_workspace_handoff_is_bound_to_the_verified_workspace_identity(monkeypatch):
    import uuid
    from app.config import settings

    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(settings, "workspace_identity_secret", secret)
    client_a = str(uuid.uuid4())
    client_b = str(uuid.uuid4())
    headers_a = _workspace_identity_headers(client_a, secret)
    headers_b = _workspace_identity_headers(client_b, secret)

    created = client.post(
        "/v1/demo/session",
        headers=headers_a,
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
            "ttl_seconds": 600,
        },
    )
    assert created.status_code == 200
    token = created.json()["access_token"]

    status_without_identity = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert status_without_identity.status_code == 401

    status_for_owner = client.get(
        "/v1/demo/session/status",
        headers={**headers_a, "Authorization": f"Bearer {token}"},
    )
    assert status_for_owner.status_code == 200

    handoff = client.post(
        "/v1/demo/session/handoff",
        headers={**headers_a, "Authorization": f"Bearer {token}"},
        json={"project": "quotesense"},
    )
    assert handoff.status_code == 200
    code = handoff.json()["code"]

    exchange_without_identity = client.post(
        "/v1/demo/session/exchange",
        json={"code": code, "project": "quotesense"},
    )
    assert exchange_without_identity.status_code == 401

    exchange_for_other_owner = client.post(
        "/v1/demo/session/exchange",
        headers=headers_b,
        json={"code": code, "project": "quotesense"},
    )
    assert exchange_for_other_owner.status_code == 401

    exchange_for_owner = client.post(
        "/v1/demo/session/exchange",
        headers=headers_a,
        json={"code": code, "project": "quotesense"},
    )
    assert exchange_for_owner.status_code == 200
    claims = decode_jwt(exchange_for_owner.json()["access_token"])
    assert claims["project"] == "quotesense"
    assert claims["session_project"] == "portfolio"

    revoke_without_identity = client.post(
        "/v1/demo/session/revoke",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert revoke_without_identity.status_code == 401

    revoke_for_owner = client.post(
        "/v1/demo/session/revoke",
        headers={**headers_a, "Authorization": f"Bearer {token}"},
    )
    assert revoke_for_owner.status_code == 200



def test_workspace_delegated_project_token_is_valid_without_studio_assertion(monkeypatch):
    """Project JWTs prove a Studio-verified single-use handoff and must work in product APIs."""
    import uuid

    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(main_module.settings, "workspace_identity_secret", secret)
    client_id = str(uuid.uuid4())
    identity_headers = _workspace_identity_headers(client_id, secret)

    created = client.post(
        "/v1/demo/session",
        headers=identity_headers,
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
            "ttl_seconds": 600,
        },
    )
    assert created.status_code == 200
    parent_token = created.json()["access_token"]

    parent_without_identity = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {parent_token}"},
    )
    assert parent_without_identity.status_code == 401

    issued = client.post(
        "/v1/demo/session/handoff",
        headers={**identity_headers, "Authorization": f"Bearer {parent_token}"},
        json={"project": "quotesense"},
    )
    assert issued.status_code == 200

    exchanged = client.post(
        "/v1/demo/session/exchange",
        headers=identity_headers,
        json={"code": issued.json()["code"], "project": "quotesense"},
    )
    assert exchanged.status_code == 200
    project_token = exchanged.json()["access_token"]

    project_status = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {project_token}"},
    )
    assert project_status.status_code == 200
    payload = project_status.json()
    assert payload["project"] == "quotesense"
    assert payload["session_project"] == "portfolio"
    assert payload["owner_scope"] == exchanged.json()["owner_scope"]


def test_workspace_product_sessions_are_isolated_and_revoked_independently(monkeypatch):
    import uuid

    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(main_module.settings, "workspace_identity_secret", secret)
    client_id = str(uuid.uuid4())
    identity = _workspace_identity_headers(client_id, secret)

    created = client.post(
        "/v1/demo/session",
        headers=identity,
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-api-key-123456",
            "project": "portfolio",
            "ttl_seconds": 600,
        },
    )
    assert created.status_code == 200
    parent_token = created.json()["access_token"]

    issued_tokens = {}
    for project in ("quotesense", "flowpilot"):
        issued = client.post(
            "/v1/demo/session/handoff",
            headers={**_workspace_identity_headers(client_id, secret),
                     "Authorization": f"Bearer {parent_token}"},
            json={"project": project},
        )
        assert issued.status_code == 200
        exchanged = client.post(
            "/v1/demo/session/exchange",
            headers=_workspace_identity_headers(client_id, secret),
            json={"code": issued.json()["code"], "project": project},
        )
        assert exchanged.status_code == 200
        issued_tokens[project] = exchanged.json()["access_token"]

    quote_claims = decode_jwt(issued_tokens["quotesense"])
    flow_claims = decode_jwt(issued_tokens["flowpilot"])
    assert quote_claims["sid"] != flow_claims["sid"]
    assert quote_claims["owner_scope"] == flow_claims["owner_scope"]

    for project, token in issued_tokens.items():
        status = client.get(
            "/v1/demo/session/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert status.status_code == 200
        assert status.json()["project"] == project

    revoked = client.post(
        "/v1/demo/session/revoke",
        headers={"Authorization": f"Bearer {issued_tokens['quotesense']}"},
    )
    assert revoked.status_code == 200

    quote_after_revoke = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {issued_tokens['quotesense']}"},
    )
    flow_after_quote_revoke = client.get(
        "/v1/demo/session/status",
        headers={"Authorization": f"Bearer {issued_tokens['flowpilot']}"},
    )
    parent_after_quote_revoke = client.get(
        "/v1/demo/session/status",
        headers={**_workspace_identity_headers(client_id, secret),
                 "Authorization": f"Bearer {parent_token}"},
    )
    assert quote_after_revoke.status_code == 401
    assert flow_after_quote_revoke.status_code == 200
    assert parent_after_quote_revoke.status_code == 200
