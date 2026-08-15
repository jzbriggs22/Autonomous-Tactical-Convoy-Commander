"""Tests for production features: health check, config endpoint, data export,
data retention, and policy versioning."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.config import GovernanceConfig
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import AlertRecord, DecisionRecord, GovernanceDB


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


def _post_events(client: TestClient, n: int, category: str, decision: str):
    now = datetime.now(timezone.utc)
    for i in range(n):
        client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": category,
            "decision": decision,
            "resolution_time_ms": 500,
            "timestamp": (now - timedelta(hours=n - i)).isoformat(),
        })


# ── health check ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["agent_id"] == "cs-agent-v1"
        assert data["is_safe"] is True
        assert data["audit_chain_valid"] is True
        assert "table_counts" in data

    def test_health_shows_table_counts(self, client):
        _post_events(client, 5, "billing_dispute", "resolve")
        resp = client.get("/health")
        data = resp.json()
        assert data["table_counts"]["decisions"] == 5

    def test_health_reflects_safety_state(self, client):
        resp = client.get("/health")
        assert resp.json()["is_safe"] is True


# ── config endpoint ──────────────────────────────────────────────────────────

class TestConfigEndpoint:
    def test_config_returns_active_config(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "cs-agent-v1"
        assert data["version"] == "1.0.0"
        assert len(data["high_risk_patterns"]) >= 3
        assert len(data["drift_thresholds"]) >= 3
        assert len(data["rollback_conditions"]) >= 1

    def test_config_shows_threshold_details(self, client):
        resp = client.get("/config")
        thresholds = resp.json()["drift_thresholds"]
        names = [t["name"] for t in thresholds]
        assert "fraud_escalation_drop" in names

    def test_config_shows_rollback_conditions(self, client):
        resp = client.get("/config")
        conditions = resp.json()["rollback_conditions"]
        assert len(conditions) >= 1
        assert "expression" in conditions[0]


# ── data export ──────────────────────────────────────────────────────────────

class TestExportEndpoint:
    def test_export_json_basic(self, client):
        _post_events(client, 5, "billing_dispute", "resolve")
        resp = client.get("/export/decisions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 5
        assert len(data["records"]) == 5
        assert "event_id" in data["records"][0]

    def test_export_csv(self, client):
        _post_events(client, 3, "fraud_claim", "escalate")
        resp = client.get("/export/decisions?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "event_id" in lines[0]
        assert "fraud_claim" in lines[1]

    def test_export_filter_by_category(self, client):
        _post_events(client, 5, "billing_dispute", "resolve")
        _post_events(client, 3, "fraud_claim", "escalate")
        resp = client.get("/export/decisions?category=fraud_claim")
        data = resp.json()
        assert data["count"] == 3
        assert all(r["case_category"] == "fraud_claim" for r in data["records"])

    def test_export_filter_by_time_range(self, client):
        _post_events(client, 10, "billing_dispute", "resolve")
        since = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        resp = client.get(f"/export/decisions?since={since}")
        data = resp.json()
        assert data["count"] <= 10

    def test_export_empty(self, client):
        resp = client.get("/export/decisions")
        data = resp.json()
        assert data["count"] == 0
        assert data["records"] == []

    def test_export_respects_limit(self, client):
        _post_events(client, 10, "billing_dispute", "resolve")
        resp = client.get("/export/decisions?limit=3")
        data = resp.json()
        assert data["count"] == 3


# ── data retention ───────────────────────────────────────────────────────────

class TestRetentionEndpoint:
    def test_purge_deletes_old_decisions(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        client = TestClient(app)

        now = datetime.now(timezone.utc)
        for i in range(5):
            client.post("/events", json={
                "case_id": f"old-{i}",
                "case_category": "billing_dispute",
                "decision": "resolve",
                "resolution_time_ms": 100,
                "timestamp": (now - timedelta(days=100 + i)).isoformat(),
            })
        for i in range(3):
            client.post("/events", json={
                "case_id": f"recent-{i}",
                "case_category": "billing_dispute",
                "decision": "resolve",
                "resolution_time_ms": 100,
                "timestamp": (now - timedelta(days=i)).isoformat(),
            })

        resp = client.post("/admin/purge?decisions_days=90")
        assert resp.status_code == 200
        data = resp.json()
        assert data["decisions_deleted"] == 5

        export = client.get("/export/decisions").json()
        assert export["count"] == 3

    def test_purge_preserves_unacknowledged_alerts(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        db.insert_alert(AlertRecord(
            alert_id="old-alert",
            agent_id=cfg.agent_id,
            timestamp=datetime.now(timezone.utc) - timedelta(days=200),
            rule_name="test",
            severity="warn",
            message="old alert",
            acknowledged=False,
        ))
        init_services(cfg, db)
        client = TestClient(app)

        resp = client.post("/admin/purge?alerts_days=180")
        data = resp.json()
        assert data["alerts_deleted"] == 0  # not acknowledged, so preserved

    def test_purge_logs_to_audit(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        client = TestClient(app)

        now = datetime.now(timezone.utc)
        client.post("/events", json={
            "case_id": "will-purge",
            "case_category": "returns",
            "decision": "resolve",
            "resolution_time_ms": 100,
            "timestamp": (now - timedelta(days=100)).isoformat(),
        })
        client.post("/admin/purge?decisions_days=90")

        audit = client.get("/audit?action=data.purged").json()
        assert len(audit) >= 1
        assert audit[0]["detail"]["decisions_deleted"] >= 1


# ── policy versioning ────────────────────────────────────────────────────────

class TestPolicyVersioning:
    def test_config_version_stored_with_decision(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)

        result = ing.ingest(IngestRequest(
            case_id="versioned",
            case_category="fraud_claim",
            decision="escalate",
            resolution_time_ms=200,
        ))
        recs = db.get_recent_decisions(cfg.agent_id, limit=1)
        assert recs[0].config_version.startswith("1.0.0:")
        assert len(recs[0].config_version.split(":")[1]) == 12  # fingerprint

    def test_different_configs_produce_different_fingerprints(self):
        cfg1 = GovernanceConfig.default_customer_service()
        cfg2 = GovernanceConfig.default_customer_service()
        cfg2.min_baseline_events = 50
        assert cfg1.fingerprint != cfg2.fingerprint

    def test_same_config_produces_same_fingerprint(self):
        cfg1 = GovernanceConfig.default_customer_service()
        cfg2 = GovernanceConfig.default_customer_service()
        assert cfg1.fingerprint == cfg2.fingerprint

    def test_config_fingerprint_ignores_agent_id(self):
        cfg1 = GovernanceConfig.default_customer_service()
        cfg2 = GovernanceConfig.default_customer_service()
        cfg2.agent_id = "different-agent"
        assert cfg1.fingerprint == cfg2.fingerprint

    def test_export_includes_config_version(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)
        ing.ingest(IngestRequest(
            case_id="export-test",
            case_category="returns",
            decision="resolve",
            resolution_time_ms=100,
        ))
        recs = db.export_decisions(cfg.agent_id)
        assert len(recs) == 1
        assert recs[0].config_version.startswith("1.0.0:")

    def test_structured_ingestion_stores_config_version(self):
        from ai_governance.structured import GovernanceDecision
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)

        gov = GovernanceDecision(
            case_category="fraud_claim",
            risk_level="critical",
            decision="escalate",
            confidence=0.95,
        )
        ing.ingest_structured(gov)
        recs = db.get_recent_decisions(cfg.agent_id, limit=1)
        assert recs[0].config_version.startswith("1.0.0:")


# ── storage methods ──────────────────────────────────────────────────────────

class TestStorageMethods:
    def test_get_table_counts(self):
        db = GovernanceDB(":memory:")
        counts = db.get_table_counts("test-agent")
        assert counts == {
            "decisions": 0,
            "baselines": 0,
            "alerts": 0,
            "rollbacks": 0,
            "metric_snapshots": 0,
        }

    def test_export_decisions_time_range(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)
        now = datetime.now(timezone.utc)

        for i in range(10):
            ing.ingest(IngestRequest(
                case_id=f"range-{i}",
                case_category="returns",
                decision="resolve",
                resolution_time_ms=100,
                timestamp=now - timedelta(days=10 - i),
            ))

        since = now - timedelta(days=5)
        recs = db.export_decisions(cfg.agent_id, since=since)
        assert len(recs) <= 6  # last 5 days + margin
        assert all(r.timestamp >= since for r in recs)

    def test_purge_old_snapshots(self):
        db = GovernanceDB(":memory:")
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=100)

        with db._tx() as c:
            c.execute(
                "INSERT INTO metric_snapshots (agent_id, timestamp, category, metric, value, sample_count) VALUES (?,?,?,?,?,?)",
                ("test", old.isoformat(), "cat", "met", 0.5, 10),
            )

        deleted = db.purge_old_snapshots("test", keep_days=90)
        assert deleted == 1
