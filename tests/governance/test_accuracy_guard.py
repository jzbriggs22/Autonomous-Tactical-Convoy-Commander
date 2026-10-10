"""Tests for the accuracy-based governance guard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_governance.accuracy_guard import (
    AccuracyGuard,
    AccuracyThresholds,
    GuardResult,
)
from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.feedback import FeedbackLabel, FeedbackPipeline
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


def _cfg():
    return GovernanceConfig.default_customer_service()


def _ingest_labeled(db, cfg, *, category, decision, ground_truth, count):
    """Ingest `count` events with a ground-truth label already attached."""
    layer = IngestionLayer(cfg, db)
    for i in range(count):
        layer.ingest(IngestRequest(
            case_id=f"{category}-{decision}-{i}",
            case_category=category,
            decision=decision,
            resolution_time_ms=500,
            ground_truth=ground_truth,
        ))


class TestAccuracyThresholds:
    def test_defaults_valid(self):
        t = AccuracyThresholds()
        assert t.min_overall_accuracy == 0.85
        assert t.min_high_risk_accuracy == 0.90
        assert t.rollback_on_high_risk_breach is True

    def test_invalid_overall_accuracy(self):
        with pytest.raises(ValueError):
            AccuracyThresholds(min_overall_accuracy=1.5)
        with pytest.raises(ValueError):
            AccuracyThresholds(min_overall_accuracy=-0.1)

    def test_invalid_high_risk_accuracy(self):
        with pytest.raises(ValueError):
            AccuracyThresholds(min_high_risk_accuracy=2.0)

    def test_invalid_min_labels(self):
        with pytest.raises(ValueError):
            AccuracyThresholds(min_labels=0)
        with pytest.raises(ValueError):
            AccuracyThresholds(min_high_risk_labels=0)


class TestAccuracyGuard:
    def test_not_enforced_below_min_labels(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        guard = AccuracyGuard(cfg, db, AccuracyThresholds(min_labels=20))
        result = guard.check()
        assert result.enforced is False
        assert result.passed is False  # not enforced ≠ passed
        assert result.violations == []
        assert "not enforced" in result.summary()

    def test_passes_with_good_accuracy(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        # 25 correct decisions — 100% accuracy
        _ingest_labeled(db, cfg, category="billing_dispute", decision="resolve",
                        ground_truth="resolve", count=25)
        guard = AccuracyGuard(cfg, db, AccuracyThresholds(min_labels=20))
        result = guard.check()
        assert result.enforced is True
        assert result.passed is True
        assert result.violations == []
        assert result.rollback_triggered is False
        assert "PASS" in result.summary()

    def test_overall_accuracy_breach_fires_critical_alert(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        # general_inquiry is NOT high-risk in the default config;
        # 10 correct + 15 wrong = 40% accuracy, below the 85% floor
        _ingest_labeled(db, cfg, category="general_inquiry", decision="resolve",
                        ground_truth="resolve", count=10)
        _ingest_labeled(db, cfg, category="general_inquiry", decision="resolve",
                        ground_truth="escalate", count=15)
        guard = AccuracyGuard(cfg, db, AccuracyThresholds(min_labels=20))
        result = guard.check()
        assert result.enforced is True
        assert result.passed is False
        kinds = [v.kind for v in result.violations]
        assert "overall_accuracy" in kinds
        assert len(result.alerts_fired) >= 1
        # overall breach alone is critical, not a rollback
        assert result.rollback_triggered is False
        alerts = db.get_recent_alerts(cfg.agent_id, limit=10)
        assert any(a.rule_name == "accuracy_guard_overall_accuracy" for a in alerts)
        assert any(a.severity == "critical" for a in alerts)

    def test_high_risk_accuracy_breach_triggers_rollback(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        # billing_dispute IS high-risk. 5 correct + 20 wrong = 20% accuracy.
        _ingest_labeled(db, cfg, category="billing_dispute", decision="resolve",
                        ground_truth="resolve", count=5)
        _ingest_labeled(db, cfg, category="billing_dispute", decision="resolve",
                        ground_truth="escalate", count=20)
        guard = AccuracyGuard(cfg, db, AccuracyThresholds(min_labels=20))
        result = guard.check()
        assert result.passed is False
        kinds = [v.kind for v in result.violations]
        assert "high_risk_accuracy" in kinds
        assert result.rollback_triggered is True
        assert db.has_active_rollback(cfg.agent_id) is True
        rollbacks = db.get_rollbacks(cfg.agent_id, limit=5)
        assert any(r.trigger_rule == "accuracy_guard_high_risk_accuracy"
                   for r in rollbacks)

    def test_rollback_disabled_by_threshold_flag(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        _ingest_labeled(db, cfg, category="billing_dispute", decision="resolve",
                        ground_truth="escalate", count=25)
        guard = AccuracyGuard(cfg, db, AccuracyThresholds(
            min_labels=20, rollback_on_high_risk_breach=False,
        ))
        result = guard.check()
        assert result.passed is False
        assert result.rollback_triggered is False
        assert db.has_active_rollback(cfg.agent_id) is False
        # alert still fires, just at critical severity
        alerts = db.get_recent_alerts(cfg.agent_id, limit=10)
        assert any(a.severity == "critical" for a in alerts)

    def test_dry_run_fires_nothing(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        _ingest_labeled(db, cfg, category="billing_dispute", decision="resolve",
                        ground_truth="escalate", count=25)
        guard = AccuracyGuard(cfg, db, AccuracyThresholds(min_labels=20))
        result = guard.check(dry_run=True)
        assert result.passed is False
        assert len(result.violations) >= 1
        assert result.alerts_fired == []
        assert result.rollback_triggered is False
        assert db.get_recent_alerts(cfg.agent_id, limit=10) == []
        assert db.has_active_rollback(cfg.agent_id) is False

    def test_high_risk_needs_min_high_risk_labels(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        # Only 2 high-risk labeled (both wrong) but 23 correct non-high-risk.
        # With min_high_risk_labels=5, the per-category high-risk check is
        # skipped — but note FeedbackPipeline itself needs 3+ high-risk labels
        # to even compute high_risk_accuracy.
        _ingest_labeled(db, cfg, category="billing_dispute", decision="resolve",
                        ground_truth="escalate", count=2)
        _ingest_labeled(db, cfg, category="general_inquiry", decision="resolve",
                        ground_truth="resolve", count=23)
        guard = AccuracyGuard(cfg, db, AccuracyThresholds(
            min_labels=20, min_high_risk_labels=5,
        ))
        result = guard.check()
        kinds = [v.kind for v in result.violations]
        assert "high_risk_accuracy" not in kinds

    def test_summary_shows_violations(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        _ingest_labeled(db, cfg, category="billing_dispute", decision="resolve",
                        ground_truth="escalate", count=25)
        guard = AccuracyGuard(cfg, db, AccuracyThresholds(min_labels=20))
        result = guard.check(dry_run=True)
        summary = result.summary()
        assert "FAIL" in summary

    def test_guard_result_summary_with_rollback(self):
        result = GuardResult(
            agent_id="a", checked_at="t", enforced=True, total_labeled=30,
        )
        from ai_governance.accuracy_guard import AccuracyViolation
        result.violations.append(AccuracyViolation(
            kind="high_risk_accuracy", category="billing",
            observed=0.5, threshold=0.9, labels=10, message="m",
        ))
        result.rollback_triggered = True
        assert "ROLLBACK TRIGGERED" in result.summary()


class TestAccuracyGuardAPI:
    @pytest.fixture
    def client(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        return TestClient(app), db, cfg

    def _ingest_via_api(self, client, category, decision, ground_truth, count):
        for i in range(count):
            resp = client.post("/events", json={
                "case_id": f"{category}-{i}",
                "case_category": category,
                "decision": decision,
                "resolution_time_ms": 500,
                "ground_truth": ground_truth,
            })
            assert resp.status_code == 201

    def test_guard_endpoint_not_enforced(self, client):
        c, db, cfg = client
        resp = c.post("/feedback/guard", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["enforced"] is False
        assert data["passed"] is False

    def test_guard_endpoint_pass(self, client):
        c, db, cfg = client
        self._ingest_via_api(c, "billing_dispute", "resolve", "resolve", 25)
        resp = c.post("/feedback/guard", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["enforced"] is True
        assert data["passed"] is True
        assert data["violations"] == []

    def test_guard_endpoint_breach_triggers_rollback(self, client):
        c, db, cfg = client
        self._ingest_via_api(c, "billing_dispute", "resolve", "escalate", 25)
        resp = c.post("/feedback/guard", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is False
        assert data["rollback_triggered"] is True
        # the safety gate must now report unsafe
        status = c.get("/status").json()
        assert status["is_safe"] is False

    def test_guard_endpoint_dry_run(self, client):
        c, db, cfg = client
        self._ingest_via_api(c, "billing_dispute", "resolve", "escalate", 25)
        resp = c.post("/feedback/guard", json={"dry_run": True})
        data = resp.json()
        assert data["passed"] is False
        assert data["rollback_triggered"] is False
        status = c.get("/status").json()
        assert status["is_safe"] is True

    def test_guard_endpoint_custom_thresholds(self, client):
        c, db, cfg = client
        # 80% accuracy: passes with floor 0.75, fails with default 0.85
        self._ingest_via_api(c, "general_inquiry", "resolve", "resolve", 20)
        self._ingest_via_api(c, "general_inquiry", "resolve", "escalate", 5)
        resp = c.post("/feedback/guard", json={"min_overall_accuracy": 0.75})
        assert resp.json()["passed"] is True
        resp = c.post("/feedback/guard", json={"min_overall_accuracy": 0.85,
                                               "dry_run": True})
        assert resp.json()["passed"] is False

    def test_guard_endpoint_rejects_bad_thresholds(self, client):
        c, db, cfg = client
        resp = c.post("/feedback/guard", json={"min_overall_accuracy": 1.5})
        assert resp.status_code == 422
        resp = c.post("/feedback/guard", json={"min_labels": 0})
        assert resp.status_code == 422
