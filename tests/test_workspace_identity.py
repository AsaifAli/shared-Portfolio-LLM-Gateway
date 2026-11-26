import hashlib
import hmac
import time
import uuid

import pytest
from fastapi import HTTPException

from app.security import owner_scope_for_workspace_client, settings


def _signed_headers(client_id: str, secret: str, timestamp: int | None = None):
    issued_at = int(time.time()) if timestamp is None else timestamp
    payload = f"studio-workspace-v1\n{client_id}\n{issued_at}".encode()
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return str(issued_at), signature


def test_workspace_client_identity_has_stable_scope_across_provider_sessions(monkeypatch):
    secret = "workspace-identity-secret-" + "x" * 48
    client_id = str(uuid.uuid4())
    monkeypatch.setattr(settings, "workspace_identity_secret", secret)
    timestamp, signature = _signed_headers(client_id, secret)

    first = owner_scope_for_workspace_client(client_id, timestamp, signature)
    second = owner_scope_for_workspace_client(client_id, timestamp, signature)

    assert len(first) == 64
    assert first == second


def test_workspace_client_identity_separates_clients(monkeypatch):
    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(settings, "workspace_identity_secret", secret)

    client_a = str(uuid.uuid4())
    client_b = str(uuid.uuid4())
    timestamp_a, signature_a = _signed_headers(client_a, secret)
    timestamp_b, signature_b = _signed_headers(client_b, secret)

    scope_a = owner_scope_for_workspace_client(client_a, timestamp_a, signature_a)
    scope_b = owner_scope_for_workspace_client(client_b, timestamp_b, signature_b)

    assert scope_a != scope_b


def test_workspace_client_identity_rejects_tampered_signature(monkeypatch):
    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(settings, "workspace_identity_secret", secret)
    client_id = str(uuid.uuid4())
    timestamp, _signature = _signed_headers(client_id, secret)

    with pytest.raises(HTTPException) as error:
        owner_scope_for_workspace_client(client_id, timestamp, "0" * 64)
    assert error.value.status_code == 401


def test_workspace_client_identity_rejects_stale_timestamp(monkeypatch):
    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(settings, "workspace_identity_secret", secret)
    client_id = str(uuid.uuid4())
    stale_time = int(time.time()) - 300
    timestamp, signature = _signed_headers(client_id, secret, stale_time)

    with pytest.raises(HTTPException) as error:
        owner_scope_for_workspace_client(client_id, timestamp, signature)
    assert error.value.status_code == 401


def test_workspace_client_identity_fails_closed_when_secret_is_missing(monkeypatch):
    monkeypatch.setattr(settings, "workspace_identity_secret", "")
    client_id = str(uuid.uuid4())
    timestamp, signature = _signed_headers(client_id, "irrelevant-secret")

    with pytest.raises(HTTPException) as error:
        owner_scope_for_workspace_client(client_id, timestamp, signature)
    assert error.value.status_code == 503


def test_workspace_client_identity_rejects_malformed_client_id(monkeypatch):
    secret = "workspace-identity-secret-" + "x" * 48
    monkeypatch.setattr(settings, "workspace_identity_secret", secret)
    timestamp, signature = _signed_headers("not-a-uuid", secret)

    with pytest.raises(HTTPException) as error:
        owner_scope_for_workspace_client("not-a-uuid", timestamp, signature)
    assert error.value.status_code == 401
