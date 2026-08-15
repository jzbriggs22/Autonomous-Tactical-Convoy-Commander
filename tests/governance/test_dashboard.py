"""Tests for dashboard assembly: normal metrics, governance signals, category rows."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from ai_governance.config import GovernanceConfig
from ai_governance.ingestion import IngestRequest
from ai_governance.storage import DecisionRecord

from .conftest import make_request, seed_decisions


class TestNormalMetrics:
    def test_resolution_rate_matches_decisions(self, ingestion, dashboard, config, db):
        seed_decisions(ingestion, 8, "billing_dispute", "resolve")
        seed_decisions(ingestion, 2, "billing_dispute", "escalate")
        snap = dashboard.build()
        assert abs(snap.normal_metrics.resolution_rate - 0.80) < 0.01

    def test_total_decisions_counts_all(self, ingestion, dashboard, config, db):
        seed_decisions(ingestion, 15, "billing_dispute", "resolve")
        snap = dashboard.build()
        assert snap.normal_metrics.total_decisions == 15

    def test_avg_response_time(self, ingestion, dashboard, config, db):
        for rt in (100, 200, 300, 400):
            ingestion.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="resolve",
                resolution_time_ms=rt,
                timestamp=datetime.now(timezone.utc),
            ))
        snap = dashboard.build()
        assert abs(snap.normal_metrics.avg_response_time_ms - 250.0) < 1.0

    def test_decisions_last_hour_only_recent(self, ingestion, dashboard, config, db):
        now = datetime.now(timezone.utc)
        # 3 recent (within last hour)
        for i in range(3):
            ingestion.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="resolve",
                resolution_time_ms=400,
                timestamp=now - timedelta(minutes=10 + i),
            ))
        # 2 old (>1 hour ago)
        for i in range(2):
            ingestion.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="resolve",
                resolution_time_ms=400,
                timestamp=now - timedelta(hours=2 + i),
            ))
        snap = dashboard.build()
        assert snap.normal_metrics.decisions_last_hour == 3

    def test_empty_db_returns_zero_metrics(self, dashboard):
        snap = dashboard.build()
        assert snap.normal_metrics.total_decisions == 0
        assert snap.normal_metrics.resolution_rate == 0.0


class TestGovernanceSignals:
    def test_safe_when_no_rollbacks(self, ingestion, dashboard):
        seed_decisions(ingestion, 5, "billing_dispute", "resolve")
        snap = dashboard.build()
        assert snap.governance_signals.is_safe is True

    def test_unsafe_after_rollback(self, config, db, dashboard):
        from ai_governance.storage import RollbackRecord
        db.insert_rollback(RollbackRecord(
            rollback_id=str(uuid.uuid4()),
            agent_id=config.agent_id,
            timestamp=datetime.now(timezone.utc),
            trigger_rule="test_rule",
            reason="test",
        ))
        snap = dashboard.build()
        assert snap.governance_signals.is_safe is False

    def test_drift_score_zero_without_baseline(self, ingestion, dashboard):
        seed_decisions(ingestion, 30, "billing_dispute", "resolve")
        snap = dashboard.build()
        assert snap.governance_signals.overall_drift_score == 0.0

    def test_high_risk_decisions_counted(self, config, db, dashboard):
        now = datetime.now(timezone.utc)
        for i in range(5):
            db.insert_decision(DecisionRecord(
                event_id=str(uuid.uuid4()),
                agent_id=config.agent_id,
                timestamp=now - timedelta(minutes=i),
                case_id=str(uuid.uuid4()),
                case_category="fraud_claim",
                is_high_risk=True,
                high_risk_score=0.5,
                decision="escalate",
                resolution_time_ms=400,
            ))
        snap = dashboard.build()
        assert snap.governance_signals.high_risk_decisions_recent >= 5

    def test_active_violations_zero_without_baseline(self, ingestion, dashboard):
        seed_decisions(ingestion, 20, "billing_dispute", "resolve")
        snap = dashboard.build()
        assert snap.governance_signals.active_violations == 0


class TestCategoryBreakdown:
    def test_multiple_categories_appear_in_breakdown(self, ingestion, dashboard):
        seed_decisions(ingestion, 10, "billing_dispute", "resolve")
        seed_decisions(ingestion, 10, "fraud_claim", "escalate")
        snap = dashboard.build()
        cats = {r.category for r in snap.category_breakdown}
        assert "billing_dispute" in cats
        assert "fraud_claim" in cats

    def test_breakdown_sorted_by_drift_score_desc(self, ingestion, dashboard):
        seed_decisions(ingestion, 10, "billing_dispute", "resolve")
        seed_decisions(ingestion, 10, "fraud_claim", "escalate")
        snap = dashboard.build()
        scores = [r.drift_score for r in snap.category_breakdown]
        assert scores == sorted(scores, reverse=True)

    def test_severity_ok_when_no_violations(self, ingestion, dashboard):
        seed_decisions(ingestion, 10, "billing_dispute", "resolve")
        snap = dashboard.build()
        for row in snap.category_breakdown:
            assert row.severity == "ok"

    def test_category_resolution_rate(self, ingestion, dashboard):
        seed_decisions(ingestion, 6, "billing_dispute", "resolve")
        seed_decisions(ingestion, 4, "billing_dispute", "deny")
        snap = dashboard.build()
        billing = next(r for r in snap.category_breakdown if r.category == "billing_dispute")
        assert abs(billing.resolution_rate - 0.60) < 0.02


class TestDashboardSerialization:
    def test_to_dict_returns_all_keys(self, ingestion, dashboard):
        seed_decisions(ingestion, 5, "billing_dispute", "resolve")
        snap = dashboard.build()
        d = snap.to_dict()
        assert "agent_id" in d
        assert "is_safe" in d
        assert "safety_reason" in d
        assert "normal_metrics" in d
        assert "governance_signals" in d
        assert "category_breakdown" in d
        assert "recent_alerts" in d

    def test_to_dict_rates_are_rounded(self, ingestion, dashboard):
        seed_decisions(ingestion, 7, "billing_dispute", "resolve")
        seed_decisions(ingestion, 3, "billing_dispute", "escalate")
        snap = dashboard.build()
        d = snap.to_dict()
        assert isinstance(d["normal_metrics"]["resolution_rate"], float)

    def test_snapshot_time_is_utc(self, dashboard):
        snap = dashboard.build()
        assert snap.snapshot_time.tzinfo is not None
