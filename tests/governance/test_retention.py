"""Tests for data retention policy engine."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.retention import RetentionManager, RetentionPolicy, RetentionResult
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
    cfg.min_baseline_events = 5
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestRetentionPolicy:
    def test_default_policy(self):
        policy = RetentionPolicy()
        assert policy.decisions_days == 90
        assert policy.alerts_days == 180
        assert policy.metric_snapshots_days == 90

    def test_custom_policy(self):
        policy = RetentionPolicy(decisions_days=30, alerts_days=60)
        assert policy.decisions_days == 30
        assert policy.alerts_days == 60


class TestRetentionManager:
    def test_apply_on_empty_db(self, setup):
        cfg, db = setup
        mgr = RetentionManager(db)
        result = mgr.apply(cfg.agent_id)
        assert result.total_deleted == 0
        assert result.executed_at is not None

    def test_apply_with_data(self, setup):
        cfg, db = setup
        from ai_governance.ingestion import IngestionLayer, IngestRequest
        layer = IngestionLayer(cfg, db)
        for _ in range(5):
            layer.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing",
                decision="resolve",
                resolution_time_ms=100,
            ))
        mgr = RetentionManager(db, RetentionPolicy(decisions_days=0))
        result = mgr.apply(cfg.agent_id)
        assert result.decisions_deleted == 5

    def test_dry_run(self, setup):
        cfg, db = setup
        mgr = RetentionManager(db)
        preview = mgr.dry_run(cfg.agent_id)
        assert "current_counts" in preview
        assert "policy" in preview
        assert preview["policy"]["decisions_days"] == 90

    def test_policy_setter(self, setup):
        cfg, db = setup
        mgr = RetentionManager(db)
        assert mgr.policy.decisions_days == 90
        mgr.policy = RetentionPolicy(decisions_days=7)
        assert mgr.policy.decisions_days == 7

    def test_result_total_deleted(self):
        result = RetentionResult(
            decisions_deleted=10,
            alerts_deleted=5,
            snapshots_deleted=3,
        )
        assert result.total_deleted == 18


class TestRetentionAPI:
    def test_apply_endpoint(self, client):
        resp = client.post("/admin/retention")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_deleted" in data
        assert data["total_deleted"] == 0

    def test_preview_endpoint(self, client):
        resp = client.get("/admin/retention/preview")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_counts" in data
        assert "policy" in data

    def test_retention_with_custom_days(self, client):
        resp = client.post("/admin/retention?decisions_days=0&alerts_days=0&snapshots_days=0")
        assert resp.status_code == 200
