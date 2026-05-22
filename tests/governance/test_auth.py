"""Tests for API key authentication and rate limiting."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import APIKey, TokenBucket, configure, reset
from ai_governance.config import GovernanceConfig
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _cleanup_auth():
    """Ensure auth state is clean before and after each test."""
    reset()
    yield
    reset()


def _make_client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 10
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


# ── authentication ──────────────────────────────────────────────────────────


class TestAuthentication:
    def test_no_auth_when_disabled(self):
        configure(enabled=False)
        client = _make_client()
        resp = client.get("/drift")
        assert resp.status_code == 200

    def test_missing_key_returns_401(self):
        configure([APIKey(key="test-key", role="admin")])
        client = _make_client()
        resp = client.get("/drift")
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["detail"]

    def test_invalid_key_returns_403(self):
        configure([APIKey(key="test-key", role="admin")])
        client = _make_client()
        resp = client.get("/drift", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403
        assert "Invalid API key" in resp.json()["detail"]

    def test_valid_key_allows_access(self):
        configure([APIKey(key="good-key", role="admin")])
        client = _make_client()
        resp = client.get("/drift", headers={"X-API-Key": "good-key"})
        assert resp.status_code == 200

    def test_key_via_query_param(self):
        configure([APIKey(key="qp-key", role="admin")])
        client = _make_client()
        resp = client.get("/drift?api_key=qp-key")
        assert resp.status_code == 200

    def test_health_is_public(self):
        configure([APIKey(key="secret", role="admin")])
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_docs_is_public(self):
        configure([APIKey(key="secret", role="admin")])
        client = _make_client()
        resp = client.get("/docs")
        assert resp.status_code == 200


# ── authorization (roles) ──────────────────────────────────────────────────


class TestAuthorization:
    def test_read_role_can_get_drift(self):
        configure([APIKey(key="reader", role="read")])
        client = _make_client()
        resp = client.get("/drift", headers={"X-API-Key": "reader"})
        assert resp.status_code == 200

    def test_read_role_can_ingest_events(self):
        configure([APIKey(key="reader", role="read")])
        client = _make_client()
        resp = client.post("/events", headers={"X-API-Key": "reader"}, json={
            "case_id": "test-1",
            "case_category": "billing_dispute",
            "decision": "resolve",
            "resolution_time_ms": 100,
        })
        assert resp.status_code == 201

    def test_read_role_cannot_purge(self):
        configure([APIKey(key="reader", role="read")])
        client = _make_client()
        resp = client.post(
            "/admin/purge?decisions_days=90",
            headers={"X-API-Key": "reader"},
        )
        assert resp.status_code == 403
        assert "Admin role required" in resp.json()["detail"]

    def test_read_role_cannot_acknowledge_alert(self):
        configure([APIKey(key="reader", role="read")])
        client = _make_client()
        resp = client.post(
            "/alerts/fake-id/acknowledge",
            headers={"X-API-Key": "reader"},
        )
        assert resp.status_code == 403

    def test_read_role_cannot_resolve_rollback(self):
        configure([APIKey(key="reader", role="read")])
        client = _make_client()
        resp = client.post(
            "/rollbacks/fake-id/resolve",
            headers={"X-API-Key": "reader"},
            json={"resolved_by": "someone"},
        )
        assert resp.status_code == 403

    def test_admin_role_can_purge(self):
        configure([APIKey(key="admin-key", role="admin")])
        client = _make_client()
        resp = client.post(
            "/admin/purge?decisions_days=90",
            headers={"X-API-Key": "admin-key"},
        )
        assert resp.status_code == 200

    def test_admin_can_do_everything(self):
        configure([APIKey(key="boss", role="admin")])
        client = _make_client()
        h = {"X-API-Key": "boss"}
        assert client.get("/drift", headers=h).status_code == 200
        assert client.get("/config", headers=h).status_code == 200
        assert client.get("/export/decisions", headers=h).status_code == 200
        assert client.post("/admin/purge?decisions_days=90", headers=h).status_code == 200

    def test_multiple_keys(self):
        configure([
            APIKey(key="admin-key", role="admin"),
            APIKey(key="read-key", role="read"),
        ])
        client = _make_client()
        resp_admin = client.post(
            "/admin/purge?decisions_days=90",
            headers={"X-API-Key": "admin-key"},
        )
        resp_read = client.post(
            "/admin/purge?decisions_days=90",
            headers={"X-API-Key": "read-key"},
        )
        assert resp_admin.status_code == 200
        assert resp_read.status_code == 403


# ── rate limiting ───────────────────────────────────────────────────────────


class TestRateLimiting:
    def test_token_bucket_basic(self):
        bucket = TokenBucket(capacity=3, tokens=3, refill_rate=0.0)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_token_bucket_refill(self):
        bucket = TokenBucket(capacity=2, tokens=0, refill_rate=1000.0)
        bucket.last_refill -= 1.0  # simulate 1 second ago
        assert bucket.consume() is True

    def test_rate_limit_returns_429(self):
        configure(
            [APIKey(key="limited", role="admin")],
            burst=3,
            refill_rate=0.0,
        )
        client = _make_client()
        h = {"X-API-Key": "limited"}
        for _ in range(3):
            resp = client.get("/drift", headers=h)
            assert resp.status_code == 200
        resp = client.get("/drift", headers=h)
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]

    def test_rate_limit_per_key(self):
        configure(
            [
                APIKey(key="key-a", role="admin"),
                APIKey(key="key-b", role="admin"),
            ],
            burst=2,
            refill_rate=0.0,
        )
        client = _make_client()
        for _ in range(2):
            client.get("/drift", headers={"X-API-Key": "key-a"})
        resp_a = client.get("/drift", headers={"X-API-Key": "key-a"})
        resp_b = client.get("/drift", headers={"X-API-Key": "key-b"})
        assert resp_a.status_code == 429
        assert resp_b.status_code == 200


# ── env-based configuration ────────────────────────────────────────────────


class TestEnvConfig:
    def test_configure_from_env(self, monkeypatch):
        from ai_governance.auth import configure_from_env, _keys, _enabled
        monkeypatch.setenv("GOVERNANCE_API_KEYS", "key1:admin,key2:read")
        configure_from_env()
        from ai_governance.auth import _keys as keys, _enabled as enabled
        assert enabled is True
        assert "key1" in keys
        assert keys["key1"].role == "admin"
        assert "key2" in keys
        assert keys["key2"].role == "read"

    def test_empty_env_disables_auth(self, monkeypatch):
        monkeypatch.setenv("GOVERNANCE_API_KEYS", "")
        from ai_governance.auth import configure_from_env
        configure_from_env()
        from ai_governance.auth import _enabled as enabled
        assert enabled is False

    def test_invalid_role_ignored(self, monkeypatch):
        monkeypatch.setenv("GOVERNANCE_API_KEYS", "key1:superuser,key2:admin")
        from ai_governance.auth import configure_from_env
        configure_from_env()
        from ai_governance.auth import _keys as keys
        assert "key1" not in keys
        assert "key2" in keys
