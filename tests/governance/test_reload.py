"""Tests for config hot-reload endpoint."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean_auth():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestConfigReload:
    def test_reload_swaps_config(self, client):
        resp = client.get("/config")
        assert resp.json()["agent_id"] == "cs-agent-v1"

        new_cfg = GovernanceConfig.default_customer_service()
        new_cfg.agent_id = "reloaded-agent"
        new_cfg.version = "2.0.0"
        resp = client.post("/admin/reload-config", json=new_cfg.model_dump(mode="json"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reloaded"
        assert data["new_version"].startswith("2.0.0:")

        resp = client.get("/config")
        assert resp.json()["agent_id"] == "reloaded-agent"
        assert resp.json()["version"] == "2.0.0"

    def test_reload_preserves_db_data(self, client):
        client.post("/events", json={
            "case_id": "keep-me",
            "case_category": "returns",
            "decision": "resolve",
            "resolution_time_ms": 100,
        })
        new_cfg = GovernanceConfig.default_customer_service()
        new_cfg.version = "2.0.0"
        client.post("/admin/reload-config", json=new_cfg.model_dump(mode="json"))

        resp = client.get("/export/decisions")
        assert resp.json()["count"] >= 1

    def test_reload_rejects_invalid_config(self, client):
        resp = client.post("/admin/reload-config", json={
            "agent_id": "bad",
            "min_baseline_events": -1,
        })
        assert resp.status_code == 422

    def test_reload_logged_to_audit(self, client):
        new_cfg = GovernanceConfig.default_customer_service()
        new_cfg.version = "3.0.0"
        client.post("/admin/reload-config", json=new_cfg.model_dump(mode="json"))

        audit = client.get("/audit?action=config.reloaded").json()
        assert len(audit) >= 1
        assert audit[0]["detail"]["new_version"] == "3.0.0"

    def test_reload_updates_fingerprint(self, client):
        resp1 = client.get("/config")
        v1 = resp1.json()["version"]

        new_cfg = GovernanceConfig.default_customer_service()
        new_cfg.min_baseline_events = 99
        resp = client.post("/admin/reload-config", json=new_cfg.model_dump(mode="json"))
        data = resp.json()
        assert data["old_version"] != data["new_version"]

    def test_ingestion_uses_new_config_after_reload(self, client):
        new_cfg = GovernanceConfig.default_customer_service()
        new_cfg.high_risk_patterns = []
        client.post("/admin/reload-config", json=new_cfg.model_dump(mode="json"))

        resp = client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": "fraud_claim",
            "decision": "escalate",
            "resolution_time_ms": 100,
        })
        assert resp.json()["is_high_risk"] is False

    def test_drift_thresholds_updated_after_reload(self, client):
        new_cfg = GovernanceConfig.default_customer_service()
        new_cfg.drift_thresholds = []
        client.post("/admin/reload-config", json=new_cfg.model_dump(mode="json"))

        resp = client.get("/config")
        assert resp.json()["drift_thresholds"] == []
