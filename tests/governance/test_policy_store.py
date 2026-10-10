"""Tests for governance policy version store."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.policy_store import PolicyStore
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def setup():
    cfg = GovernanceConfig.default_customer_service()
    db = GovernanceDB(":memory:")
    return cfg, db


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestPolicyStore:
    def test_commit_creates_version(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        version = store.commit(cfg, "alice", "Initial policy")
        assert version.policy_id
        assert version.agent_id == cfg.agent_id
        assert version.config_version == cfg.version
        assert version.config_fingerprint == cfg.fingerprint
        assert version.changed_by == "alice"
        assert version.is_active is True

    def test_list_versions_empty(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        versions = store.list_versions(cfg.agent_id)
        assert versions == []

    def test_list_versions_after_commit(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        store.commit(cfg, "alice", "v1")
        store.commit(cfg, "bob", "v2")
        versions = store.list_versions(cfg.agent_id)
        assert len(versions) == 2

    def test_list_versions_newest_first(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        v1 = store.commit(cfg, "alice", "first")
        v2 = store.commit(cfg, "bob", "second")
        versions = store.list_versions(cfg.agent_id)
        assert versions[0].policy_id == v2.policy_id
        assert versions[1].policy_id == v1.policy_id

    def test_get_active_version(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        store.commit(cfg, "alice", "v1")
        v2 = store.commit(cfg, "bob", "v2")
        active = store.get_active(cfg.agent_id)
        assert active is not None
        assert active.policy_id == v2.policy_id
        assert active.is_active is True

    def test_only_one_active_at_a_time(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        store.commit(cfg, "alice", "v1")
        store.commit(cfg, "bob", "v2")
        store.commit(cfg, "carol", "v3")
        versions = store.list_versions(cfg.agent_id)
        active_count = sum(1 for v in versions if v.is_active)
        assert active_count == 1

    def test_get_version_by_id(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        committed = store.commit(cfg, "alice", "test")
        retrieved = store.get_version(committed.policy_id)
        assert retrieved is not None
        assert retrieved.policy_id == committed.policy_id

    def test_get_nonexistent_version(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        result = store.get_version("nonexistent-id")
        assert result is None

    def test_rollback_to_prior_version(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        v1 = store.commit(cfg, "alice", "initial")
        _v2 = store.commit(cfg, "bob", "update")
        result = store.rollback_to(v1.policy_id)
        assert result is not None
        assert result.policy_id == v1.policy_id
        active = store.get_active(cfg.agent_id)
        assert active.policy_id == v1.policy_id

    def test_rollback_nonexistent_returns_none(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        result = store.rollback_to("nonexistent")
        assert result is None

    def test_config_roundtrip(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        version = store.commit(cfg, "alice", "test")
        restored = version.to_config()
        assert restored.agent_id == cfg.agent_id
        assert restored.version == cfg.version

    def test_list_limit(self, setup):
        cfg, db = setup
        store = PolicyStore(db)
        for i in range(5):
            store.commit(cfg, "user", f"v{i}")
        versions = store.list_versions(cfg.agent_id, limit=3)
        assert len(versions) == 3


class TestPolicyAPI:
    def test_commit_endpoint(self, client):
        resp = client.post("/admin/policy/commit", json={
            "changed_by": "alice",
            "description": "Initial commit",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "policy_id" in data
        assert data["changed_by"] == "alice"
        assert data["is_active"] is True

    def test_list_versions_endpoint(self, client):
        client.post("/admin/policy/commit", json={"changed_by": "a", "description": "v1"})
        client.post("/admin/policy/commit", json={"changed_by": "b", "description": "v2"})
        resp = client.get("/admin/policy/versions")
        assert resp.status_code == 200
        versions = resp.json()
        assert len(versions) == 2

    def test_active_policy_endpoint(self, client):
        client.post("/admin/policy/commit", json={"changed_by": "a", "description": "v1"})
        resp = client.get("/admin/policy/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_active"] is True

    def test_active_policy_none_when_uncommitted(self, client):
        resp = client.get("/admin/policy/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is None

    def test_rollback_endpoint(self, client):
        r1 = client.post("/admin/policy/commit", json={"changed_by": "a", "description": "v1"})
        policy_id = r1.json()["policy_id"]
        client.post("/admin/policy/commit", json={"changed_by": "b", "description": "v2"})
        resp = client.post(f"/admin/policy/{policy_id}/rollback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rolled_back"
        assert data["policy_id"] == policy_id

    def test_rollback_not_found(self, client):
        resp = client.post("/admin/policy/nonexistent/rollback")
        assert resp.status_code == 404
