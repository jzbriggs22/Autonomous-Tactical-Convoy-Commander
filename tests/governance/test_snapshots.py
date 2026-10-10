"""Tests for governance state snapshots and diffing."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.snapshots import StateSnapshot, SnapshotDiff, diff_snapshots
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 5
    cfg.recent_window_size = 20
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
        thr.recent_window = 10
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestStateSnapshot:
    def test_roundtrip_json(self):
        snap = StateSnapshot(
            agent_id="test", timestamp="2024-01-01T00:00:00",
            config_version="1.0.0", config_fingerprint="abc123",
            is_safe=True, safety_reason="all good",
            overall_drift_score=0.05, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={"billing": {"drift_score": 0.02}},
        )
        text = snap.to_json()
        loaded = StateSnapshot.from_json(text)
        assert loaded.agent_id == "test"
        assert loaded.overall_drift_score == 0.05
        assert loaded.category_metrics["billing"]["drift_score"] == 0.02


class TestDiffSnapshots:
    def test_no_changes_produces_empty_diff(self):
        snap = StateSnapshot(
            agent_id="test", timestamp="2024-01-01T00:00:00",
            config_version="1.0.0", config_fingerprint="abc123",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.0, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        result = diff_snapshots(snap, snap)
        assert len(result.items) == 0
        assert result.safety_changed is False
        assert result.has_regressions is False

    def test_safety_change_detected(self):
        old = StateSnapshot(
            agent_id="test", timestamp="t1",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.0, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        new = StateSnapshot(
            agent_id="test", timestamp="t2",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=False, safety_reason="rollback active",
            overall_drift_score=0.0, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        result = diff_snapshots(old, new)
        assert result.safety_changed is True
        safety_items = [d for d in result.items if d.field == "is_safe"]
        assert len(safety_items) == 1
        assert safety_items[0].severity == "critical"

    def test_drift_score_increase_flagged(self):
        old = StateSnapshot(
            agent_id="test", timestamp="t1",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.05, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        new = StateSnapshot(
            agent_id="test", timestamp="t2",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.45, total_decisions=120,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        result = diff_snapshots(old, new)
        drift_items = [d for d in result.items if d.field == "overall_drift_score"]
        assert len(drift_items) == 1
        assert drift_items[0].severity == "critical"
        assert drift_items[0].delta > 0

    def test_new_category_detected(self):
        old = StateSnapshot(
            agent_id="test", timestamp="t1",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.0, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        new = StateSnapshot(
            agent_id="test", timestamp="t2",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.0, total_decisions=110,
            active_violations=0, rollback_events=0,
            category_metrics={"fraud_claim": {"drift_score": 0.0, "violations": 0}},
        )
        result = diff_snapshots(old, new)
        added = [d for d in result.items if d.field == "category_added"]
        assert len(added) == 1
        assert added[0].new_value == "fraud_claim"

    def test_config_change_detected(self):
        old = StateSnapshot(
            agent_id="test", timestamp="t1",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.0, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        new = StateSnapshot(
            agent_id="test", timestamp="t2",
            config_version="2.0.0", config_fingerprint="xyz",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.0, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        result = diff_snapshots(old, new)
        assert result.config_changed is True

    def test_summary_format(self):
        old = StateSnapshot(
            agent_id="test", timestamp="t1",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.0, total_decisions=100,
            active_violations=0, rollback_events=0,
            category_metrics={},
        )
        new = StateSnapshot(
            agent_id="test", timestamp="t2",
            config_version="1.0.0", config_fingerprint="abc",
            is_safe=True, safety_reason="ok",
            overall_drift_score=0.15, total_decisions=200,
            active_violations=2, rollback_events=0,
            category_metrics={},
        )
        result = diff_snapshots(old, new)
        summary = result.summary()
        assert "Diff:" in summary
        assert "t1" in summary
        assert "t2" in summary


class TestSnapshotAPI:
    def test_take_snapshot(self, client):
        resp = client.post("/admin/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_id" in data
        assert "is_safe" in data
        assert "category_metrics" in data

    def test_diff_after_snapshot(self, client):
        client.post("/admin/snapshot")
        # Ingest some events to change state
        for _ in range(5):
            client.post("/events", json={
                "case_id": str(uuid.uuid4()),
                "case_category": "fraud_claim",
                "decision": "escalate",
                "resolution_time_ms": 100,
            })
        resp = client.post("/admin/diff")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "summary" in data
        assert "has_regressions" in data

    def test_diff_without_snapshot_returns_400(self, client):
        import ai_governance.api as api_mod
        api_mod._last_snapshot = None
        resp = client.post("/admin/diff")
        assert resp.status_code == 400

    def test_diff_with_provided_baseline(self, client):
        baseline = {
            "agent_id": "cs-agent-v1",
            "timestamp": "2024-01-01T00:00:00",
            "config_version": "1.0.0",
            "config_fingerprint": "abc123def456",
            "is_safe": True,
            "safety_reason": "ok",
            "overall_drift_score": 0.0,
            "total_decisions": 0,
            "active_violations": 0,
            "rollback_events": 0,
            "category_metrics": {},
        }
        resp = client.post("/admin/diff", json=baseline)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_snapshot_captures_category_metrics(self, client):
        for _ in range(5):
            client.post("/events", json={
                "case_id": str(uuid.uuid4()),
                "case_category": "billing_dispute",
                "decision": "resolve",
                "resolution_time_ms": 100,
            })
        resp = client.post("/admin/snapshot")
        data = resp.json()
        assert "billing_dispute" in data["category_metrics"]
