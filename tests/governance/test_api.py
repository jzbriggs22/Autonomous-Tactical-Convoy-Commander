"""Tests for the FastAPI governance API: endpoints, validation, rollback resume."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.config import (
    AlertSeverity,
    DriftThreshold,
    GovernanceConfig,
    MetricDirection,
    RollbackCondition,
)
from ai_governance.storage import GovernanceDB, RollbackRecord


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 10
    cfg.recent_window_size = 30
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 10
        thr.recent_window = 15
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


@pytest.fixture
def drift_client():
    """Client preconfigured with tight thresholds for drift testing."""
    cfg = GovernanceConfig(
        agent_id="api-test-agent",
        min_baseline_events=10,
        recent_window_size=30,
        drift_thresholds=[
            DriftThreshold(
                name="esc_spike",
                category="billing_dispute",
                metric="escalation_rate",
                max_delta=0.10,
                direction=MetricDirection.INCREASE,
                min_baseline_samples=10,
                recent_window=15,
                severity=AlertSeverity.CRITICAL,
            )
        ],
        rollback_conditions=[
            RollbackCondition(
                name="fraud_low_esc",
                description="Fraud not escalated",
                expression="fraud_claim_escalation_rate < 0.50 and fraud_claim_count >= 5",
                cooldown_seconds=60,
            ),
        ],
    )
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


def _post_events(client: TestClient, n: int, category: str, decision: str,
                  ts_offset_hours: int = 0):
    now = datetime.now(timezone.utc)
    for i in range(n):
        client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": category,
            "decision": decision,
            "resolution_time_ms": 500,
            "timestamp": (now - timedelta(hours=ts_offset_hours + n - i)).isoformat(),
        })


# ── ingest endpoints ─────────────────────────────────────────────────────────

class TestIngestEndpoint:
    def test_ingest_single_event(self, client):
        resp = client.post("/events", json={
            "case_id": "c1",
            "case_category": "billing_dispute",
            "decision": "resolve",
            "resolution_time_ms": 400,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["event_id"]
        assert data["is_high_risk"] is True  # billing_dispute matches pattern

    def test_ingest_with_metadata(self, client):
        resp = client.post("/events", json={
            "case_id": "c2",
            "case_category": "returns",
            "decision": "resolve",
            "resolution_time_ms": 200,
            "metadata": {"account_tier": "enterprise"},
        })
        assert resp.status_code == 201
        assert resp.json()["is_high_risk"] is True
        assert "high_value_account" in resp.json()["matched_patterns"]

    def test_ingest_validation_error(self, client):
        resp = client.post("/events", json={
            "case_id": "c3",
            "case_category": "billing_dispute",
            "decision": "invalid_decision",
            "resolution_time_ms": 100,
        })
        assert resp.status_code == 422

    def test_ingest_batch(self, client):
        events = [
            {
                "case_id": f"batch-{i}",
                "case_category": "returns",
                "decision": "resolve",
                "resolution_time_ms": 300,
            }
            for i in range(5)
        ]
        resp = client.post("/events/batch", json=events)
        assert resp.status_code == 201
        assert len(resp.json()) == 5

    def test_batch_validation_rejects_bad_event(self, client):
        events = [
            {"case_id": "ok", "case_category": "returns", "decision": "resolve",
             "resolution_time_ms": 300},
            {"case_id": "bad", "case_category": "returns", "decision": "nope",
             "resolution_time_ms": 300},
        ]
        resp = client.post("/events/batch", json=events)
        assert resp.status_code == 422


class TestGroundTruth:
    def test_add_ground_truth(self, client):
        resp = client.post("/events", json={
            "case_id": "gt1",
            "case_category": "fraud_claim",
            "decision": "resolve",
            "resolution_time_ms": 500,
        })
        event_id = resp.json()["event_id"]
        resp2 = client.post(f"/events/{event_id}/ground-truth",
                            json={"ground_truth": "escalate"})
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "updated"

    def test_ground_truth_not_found(self, client):
        resp = client.post("/events/nonexistent/ground-truth",
                           json={"ground_truth": "resolve"})
        assert resp.status_code == 404

    def test_ground_truth_invalid(self, client):
        resp = client.post("/events", json={
            "case_id": "gt2",
            "case_category": "returns",
            "decision": "resolve",
            "resolution_time_ms": 200,
        })
        event_id = resp.json()["event_id"]
        resp2 = client.post(f"/events/{event_id}/ground-truth",
                            json={"ground_truth": "unknown"})
        assert resp2.status_code == 422


# ── status & safety endpoints ────────────────────────────────────────────────

class TestStatusEndpoint:
    def test_safe_initially(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.json()["is_safe"] is True

    def test_unsafe_after_rollback(self, client):
        svc = init_services(
            GovernanceConfig.default_customer_service(),
            GovernanceDB(":memory:"),
        )
        svc.db.insert_rollback(RollbackRecord(
            rollback_id=str(uuid.uuid4()),
            agent_id=svc.config.agent_id,
            timestamp=datetime.now(timezone.utc),
            trigger_rule="test",
            reason="test",
        ))
        resp = TestClient(app).get("/status")
        assert resp.json()["is_safe"] is False


# ── drift endpoint ───────────────────────────────────────────────────────────

class TestDriftEndpoint:
    def test_drift_no_data(self, client):
        resp = client.get("/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_drift_score"] == 0.0
        assert data["violations"] == []

    def test_drift_detects_violations(self, drift_client):
        _post_events(drift_client, 15, "billing_dispute", "resolve", ts_offset_hours=30)
        drift_client.post("/baseline/compute", json={})
        _post_events(drift_client, 20, "billing_dispute", "escalate")

        resp = drift_client.get("/drift")
        data = resp.json()
        assert len(data["violations"]) >= 1
        assert data["alerts_fired"] >= 1

    def test_drift_fires_rollback_condition(self, drift_client):
        _post_events(drift_client, 10, "fraud_claim", "resolve")

        resp = drift_client.get("/drift")
        data = resp.json()
        rollback_alerts = [a for a in data.get("violations", [])
                          if a.get("severity") == "rollback"]
        # Check status — rollback condition should have triggered
        status_resp = drift_client.get("/status")
        assert status_resp.json()["is_safe"] is False


# ── baseline endpoint ────────────────────────────────────────────────────────

class TestBaselineEndpoint:
    def test_compute_baseline(self, client):
        _post_events(client, 15, "billing_dispute", "resolve", ts_offset_hours=10)
        resp = client.post("/baseline/compute", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "billing_dispute" in data["categories_computed"]

    def test_compute_baseline_with_filter(self, client):
        _post_events(client, 15, "billing_dispute", "resolve", ts_offset_hours=10)
        _post_events(client, 15, "returns", "resolve", ts_offset_hours=10)
        resp = client.post("/baseline/compute",
                           json={"categories": ["billing_dispute"]})
        assert "billing_dispute" in resp.json()["categories_computed"]
        assert "returns" not in resp.json()["categories_computed"]


# ── dashboard endpoint ───────────────────────────────────────────────────────

class TestDashboardEndpoint:
    def test_dashboard_returns_all_sections(self, client):
        _post_events(client, 10, "billing_dispute", "resolve")
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_safe" in data
        assert "normal_metrics" in data
        assert "governance_signals" in data
        assert "category_breakdown" in data
        assert "recent_alerts" in data

    def test_dashboard_empty_db(self, client):
        resp = client.get("/dashboard")
        data = resp.json()
        assert data["is_safe"] is True
        assert data["normal_metrics"]["total_decisions"] == 0


# ── alerts endpoint ──────────────────────────────────────────────────────────

class TestAlertsEndpoint:
    def test_alerts_empty(self, client):
        resp = client.get("/alerts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_acknowledge_alert(self, drift_client):
        _post_events(drift_client, 15, "billing_dispute", "resolve", ts_offset_hours=30)
        drift_client.post("/baseline/compute", json={})
        _post_events(drift_client, 20, "billing_dispute", "escalate")
        drift_client.get("/drift")  # triggers alerts

        alerts = drift_client.get("/alerts").json()
        assert len(alerts) >= 1
        alert_id = alerts[0]["alert_id"]

        ack_resp = drift_client.post(f"/alerts/{alert_id}/acknowledge")
        assert ack_resp.status_code == 200

        alerts_after = drift_client.get("/alerts").json()
        acked = next(a for a in alerts_after if a["alert_id"] == alert_id)
        assert acked["acknowledged"] is True

    def test_acknowledge_nonexistent(self, client):
        resp = client.post("/alerts/fake-id/acknowledge")
        assert resp.status_code == 404


# ── rollback management endpoints ────────────────────────────────────────────

class TestRollbackEndpoints:
    def test_rollbacks_empty(self, client):
        resp = client.get("/rollbacks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_resolve_rollback(self, drift_client):
        _post_events(drift_client, 10, "fraud_claim", "resolve")
        drift_client.get("/drift")  # triggers rollback condition

        status = drift_client.get("/status").json()
        assert status["is_safe"] is False

        rollbacks = drift_client.get("/rollbacks").json()
        assert len(rollbacks) >= 1
        rb_id = rollbacks[0]["rollback_id"]
        assert rollbacks[0]["resolved"] is False

        resolve_resp = drift_client.post(
            f"/rollbacks/{rb_id}/resolve",
            json={"resolved_by": "pm@company.com"},
        )
        assert resolve_resp.status_code == 200

        status_after = drift_client.get("/status").json()
        assert status_after["is_safe"] is True

        rollbacks_after = drift_client.get("/rollbacks").json()
        resolved = next(r for r in rollbacks_after if r["rollback_id"] == rb_id)
        assert resolved["resolved"] is True
        assert resolved["resolved_by"] == "pm@company.com"

    def test_resolve_all_rollbacks(self, drift_client):
        _post_events(drift_client, 10, "fraud_claim", "resolve")
        drift_client.get("/drift")

        resp = drift_client.post("/rollbacks/resolve-all",
                                 json={"resolved_by": "admin"})
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1
        assert drift_client.get("/status").json()["is_safe"] is True

    def test_resolve_nonexistent_rollback(self, client):
        resp = client.post("/rollbacks/fake-id/resolve",
                           json={"resolved_by": "admin"})
        assert resp.status_code == 404

    def test_resolve_already_resolved(self, drift_client):
        _post_events(drift_client, 10, "fraud_claim", "resolve")
        drift_client.get("/drift")
        rollbacks = drift_client.get("/rollbacks").json()
        rb_id = rollbacks[0]["rollback_id"]

        drift_client.post(f"/rollbacks/{rb_id}/resolve",
                          json={"resolved_by": "admin"})
        # Second resolve should 404 (already resolved)
        resp = drift_client.post(f"/rollbacks/{rb_id}/resolve",
                                 json={"resolved_by": "admin"})
        assert resp.status_code == 404


# ── metric history endpoint ──────────────────────────────────────────────────

class TestMetricHistoryEndpoint:
    def test_history_empty_when_no_drift_runs(self, client):
        resp = client.get("/metrics/history/billing_dispute/resolution_rate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "billing_dispute"
        assert data["metric"] == "resolution_rate"
        assert data["history"] == []
        assert data["baseline_value"] is None

    def test_history_populated_after_drift_detection(self, drift_client):
        _post_events(drift_client, 20, "billing_dispute", "resolve", ts_offset_hours=10)
        drift_client.post("/baseline/compute", json={})
        drift_client.get("/drift")  # triggers metric snapshot recording

        resp = drift_client.get("/metrics/history/billing_dispute/resolution_rate")
        data = resp.json()
        assert len(data["history"]) >= 1
        assert data["history"][0]["value"] == 1.0  # all resolve → 100% resolution
        assert data["baseline_value"] is not None

    def test_history_with_limit(self, drift_client):
        _post_events(drift_client, 20, "billing_dispute", "resolve")
        for _ in range(3):
            drift_client.get("/drift")  # record multiple snapshots

        resp = drift_client.get("/metrics/history/billing_dispute/resolution_rate?limit=2")
        data = resp.json()
        assert len(data["history"]) <= 2
