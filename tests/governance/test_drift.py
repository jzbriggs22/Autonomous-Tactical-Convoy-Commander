"""Tests for drift detection: metric computation, baseline storage, violation detection."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from ai_governance.config import (
    AlertSeverity,
    DriftThreshold,
    GovernanceConfig,
    MetricDirection,
)
from ai_governance.drift import DriftDetector, _compute_metrics
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import DecisionRecord, GovernanceDB

from .conftest import make_request, seed_decisions


def _make_records(
    n: int,
    category: str = "billing_dispute",
    decision: str = "resolve",
    is_high_risk: bool = False,
    ground_truth: str = None,
    rt_ms: int = 500,
    agent_id: str = "cs-agent-v1",
) -> list[DecisionRecord]:
    now = datetime.now(timezone.utc)
    return [
        DecisionRecord(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            timestamp=now - timedelta(seconds=n - i),
            case_id=str(uuid.uuid4()),
            case_category=category,
            is_high_risk=is_high_risk,
            high_risk_score=0.3 if is_high_risk else 0.0,
            decision=decision,
            resolution_time_ms=rt_ms,
            ground_truth=ground_truth,
        )
        for i in range(n)
    ]


class TestComputeMetrics:
    def test_empty_returns_zeros(self):
        m = _compute_metrics([])
        assert m.total_events == 0
        assert m.resolution_rate == 0.0
        assert m.accuracy is None

    def test_resolution_rate(self):
        records = (
            _make_records(7, decision="resolve")
            + _make_records(3, decision="escalate")
        )
        m = _compute_metrics(records)
        assert abs(m.resolution_rate - 0.70) < 0.01
        assert abs(m.escalation_rate - 0.30) < 0.01

    def test_accuracy_with_ground_truth(self):
        records = _make_records(5, decision="resolve", ground_truth="resolve")
        m = _compute_metrics(records)
        assert m.accuracy == 1.0

    def test_accuracy_none_when_insufficient_gt(self):
        records = _make_records(4, decision="resolve", ground_truth="resolve")
        m = _compute_metrics(records)
        assert m.accuracy is None

    def test_high_risk_escalation_rate(self):
        hr_esc = _make_records(3, decision="escalate", is_high_risk=True)
        hr_res = _make_records(7, decision="resolve", is_high_risk=True)
        m = _compute_metrics(hr_esc + hr_res)
        assert abs(m.high_risk_escalation_rate - 0.30) < 0.01

    def test_entropy_uniform(self):
        # 5 resolves + 5 escalates → entropy should be ~1.0 bit
        records = _make_records(5, decision="resolve") + _make_records(5, decision="escalate")
        m = _compute_metrics(records)
        assert 0.9 < m.decision_entropy < 1.1

    def test_entropy_homogeneous(self):
        records = _make_records(10, decision="resolve")
        m = _compute_metrics(records)
        assert m.decision_entropy == 0.0

    def test_avg_response_time(self):
        records = _make_records(10, rt_ms=300)
        m = _compute_metrics(records)
        assert m.avg_response_time_ms == 300.0


class TestBaselineComputation:
    def test_baseline_stored_per_category(self, ingestion, detector, config, db):
        seed_decisions(ingestion, 35, "billing_dispute", "resolve")
        detector.compute_baseline()
        val = db.get_baseline(config.agent_id, "billing_dispute", "resolution_rate")
        assert val is not None
        rate, count = val
        assert abs(rate - 1.0) < 0.01
        assert count >= config.min_baseline_events

    def test_baseline_not_computed_if_insufficient_events(
        self, ingestion, detector, config, db
    ):
        seed_decisions(ingestion, 10, "billing_dispute", "resolve")
        detector.compute_baseline()
        val = db.get_baseline(config.agent_id, "billing_dispute", "resolution_rate")
        assert val is None

    def test_baseline_overwrites_on_second_call(self, ingestion, detector, config, db):
        seed_decisions(ingestion, 35, "fraud_claim", "escalate")
        detector.compute_baseline()
        first = db.get_baseline(config.agent_id, "fraud_claim", "escalation_rate")

        seed_decisions(ingestion, 35, "fraud_claim", "resolve")
        detector.compute_baseline()
        second = db.get_baseline(config.agent_id, "fraud_claim", "escalation_rate")

        # Baseline should now reflect mix (still dominated by original since oldest first)
        assert first is not None and second is not None

    def test_category_filter(self, ingestion, detector, config, db):
        seed_decisions(ingestion, 35, "billing_dispute", "resolve")
        seed_decisions(ingestion, 35, "fraud_claim", "escalate")
        detector.compute_baseline(categories=["billing_dispute"])
        billing = db.get_baseline(config.agent_id, "billing_dispute", "resolution_rate")
        fraud = db.get_baseline(config.agent_id, "fraud_claim", "escalation_rate")
        assert billing is not None
        assert fraud is None


class TestDriftDetection:
    def test_no_drift_when_behavior_matches_baseline(self, ingestion, detector, config):
        seed_decisions(ingestion, 35, "billing_dispute", "resolve")
        detector.compute_baseline()
        seed_decisions(ingestion, 55, "billing_dispute", "resolve")
        report = detector.detect()
        billing_violations = [v for v in report.violations if v.category == "billing_dispute"]
        assert not billing_violations

    def test_detects_resolution_rate_drop(self, config, db):
        """Insert high-resolution baseline, then low-resolution recent."""
        cfg = GovernanceConfig(
            agent_id="test-agent",
            min_baseline_events=10,
            recent_window_size=20,
            drift_thresholds=[
                DriftThreshold(
                    name="res_drop",
                    category="billing_dispute",
                    metric="resolution_rate",
                    max_delta=0.10,
                    direction=MetricDirection.DECREASE,
                    min_baseline_samples=10,
                    recent_window=20,
                    severity=AlertSeverity.WARN,
                )
            ],
        )
        ing = IngestionLayer(cfg, db)
        det = DriftDetector(cfg, db)

        # 20 "good" decisions (high resolution) — these become baseline
        now = datetime.now(timezone.utc)
        for i in range(20):
            ing.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="resolve",
                resolution_time_ms=400,
                timestamp=now - timedelta(hours=50 - i),
            ))

        det.compute_baseline()

        # 25 "drifted" decisions (low resolution) — recent window
        for i in range(25):
            ing.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="deny" if i % 2 == 0 else "escalate",
                resolution_time_ms=400,
                timestamp=now - timedelta(minutes=25 - i),
            ))

        report = det.detect()
        violations = [v for v in report.violations if v.metric == "resolution_rate"]
        assert len(violations) >= 1
        assert violations[0].delta < 0  # resolution rate went down

    def test_detects_escalation_spike(self, config, db):
        cfg = GovernanceConfig(
            agent_id="spike-agent",
            min_baseline_events=10,
            recent_window_size=20,
            drift_thresholds=[
                DriftThreshold(
                    name="esc_spike",
                    category="billing_dispute",
                    metric="escalation_rate",
                    max_delta=0.15,
                    direction=MetricDirection.INCREASE,
                    min_baseline_samples=10,
                    recent_window=20,
                    severity=AlertSeverity.CRITICAL,
                )
            ],
        )
        ing = IngestionLayer(cfg, db)
        det = DriftDetector(cfg, db)

        now = datetime.now(timezone.utc)
        for i in range(20):
            ing.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="resolve",
                resolution_time_ms=400,
                timestamp=now - timedelta(hours=50 - i),
            ))
        det.compute_baseline()

        for i in range(25):
            ing.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="escalate",
                resolution_time_ms=400,
                timestamp=now - timedelta(minutes=25 - i),
            ))

        report = det.detect()
        viols = [v for v in report.violations if v.metric == "escalation_rate"]
        assert len(viols) >= 1
        assert viols[0].severity == "critical"

    def test_insufficient_recent_events_skips_threshold(self, ingestion, detector, config):
        seed_decisions(ingestion, 35, "billing_dispute", "resolve")
        detector.compute_baseline()
        # Only 10 recent events — below the recent_window of 50 in the default threshold
        seed_decisions(ingestion, 10, "billing_dispute", "deny")
        report = detector.detect()
        billing_viols = [v for v in report.violations if v.category == "billing_dispute"]
        assert not billing_viols  # skipped because too few recent events

    def test_wildcard_category_threshold(self, config, db):
        cfg = GovernanceConfig(
            agent_id="wc-agent",
            min_baseline_events=10,
            recent_window_size=20,
            drift_thresholds=[
                DriftThreshold(
                    name="global_res",
                    category="*",
                    metric="resolution_rate",
                    max_delta=0.15,
                    direction=MetricDirection.DECREASE,
                    min_baseline_samples=10,
                    recent_window=20,
                    severity=AlertSeverity.WARN,
                )
            ],
        )
        ing = IngestionLayer(cfg, db)
        det = DriftDetector(cfg, db)

        now = datetime.now(timezone.utc)
        for i in range(20):
            for cat in ("billing_dispute", "fraud_claim"):
                ing.ingest(IngestRequest(
                    case_id=str(uuid.uuid4()),
                    case_category=cat,
                    decision="resolve",
                    resolution_time_ms=400,
                    timestamp=now - timedelta(hours=50 - i),
                ))
        det.compute_baseline()

        for i in range(25):
            for cat in ("billing_dispute", "fraud_claim"):
                ing.ingest(IngestRequest(
                    case_id=str(uuid.uuid4()),
                    case_category=cat,
                    decision="deny",
                    resolution_time_ms=400,
                    timestamp=now - timedelta(minutes=25 - i),
                ))

        report = det.detect()
        assert len(report.violations) > 0

    def test_drift_score_zero_when_no_baseline(self, ingestion, detector):
        seed_decisions(ingestion, 60, "billing_dispute", "resolve")
        report = detector.detect()
        assert report.overall_drift_score == 0.0

    def test_small_category_metrics_computed(self, config, db):
        cfg = GovernanceConfig(
            agent_id="small-agent",
            min_baseline_events=5,
            recent_window_size=10,
            drift_thresholds=[
                DriftThreshold(
                    name="small",
                    category="rare",
                    metric="resolution_rate",
                    max_delta=0.1,
                    direction=MetricDirection.DECREASE,
                    min_baseline_samples=5,
                    recent_window=10,
                    severity=AlertSeverity.WARN,
                )
            ],
        )
        ing = IngestionLayer(cfg, db)
        det = DriftDetector(cfg, db)
        now = datetime.now(timezone.utc)

        for i in range(6):
            ing.ingest(IngestRequest(
                case_id=f"rare-baseline-{i}",
                case_category="rare",
                decision="resolve",
                resolution_time_ms=100,
                timestamp=now - timedelta(hours=50 - i),
            ))
        det.compute_baseline()

        for i in range(12):
            ing.ingest(IngestRequest(
                case_id=f"rare-recent-{i}",
                case_category="rare",
                decision="deny",
                resolution_time_ms=100,
                timestamp=now - timedelta(minutes=12 - i),
            ))
        report = det.detect()
        assert "rare" in report.recent_metrics
        viols = [v for v in report.violations if v.category == "rare"]
        assert len(viols) >= 1

    def test_all_denied_category_metrics(self):
        records = _make_records(10, decision="deny")
        m = _compute_metrics(records)
        assert m.denial_rate == 1.0
        assert m.resolution_rate == 0.0
        assert m.escalation_rate == 0.0

    def test_no_high_risk_events(self):
        records = _make_records(10, decision="resolve", is_high_risk=False)
        m = _compute_metrics(records)
        assert m.high_risk_count == 0
        assert m.high_risk_escalation_rate == 0.0
        assert m.high_risk_accuracy is None

    def test_all_high_risk_no_escalations(self):
        records = _make_records(10, decision="resolve", is_high_risk=True)
        m = _compute_metrics(records)
        assert m.high_risk_count == 10
        assert m.high_risk_escalation_rate == 0.0

    def test_mixed_ground_truth_accuracy(self):
        correct = _make_records(6, decision="resolve", ground_truth="resolve")
        wrong = _make_records(4, decision="resolve", ground_truth="escalate")
        m = _compute_metrics(correct + wrong)
        assert m.accuracy is not None
        assert abs(m.accuracy - 0.6) < 0.01

    def test_overall_drift_score_increases_with_violations(self, config, db):
        cfg = GovernanceConfig(
            agent_id="score-agent",
            min_baseline_events=10,
            recent_window_size=20,
            drift_thresholds=[
                DriftThreshold(
                    name="t1",
                    category="billing_dispute",
                    metric="resolution_rate",
                    max_delta=0.05,  # tight threshold
                    direction=MetricDirection.DECREASE,
                    min_baseline_samples=10,
                    recent_window=20,
                    severity=AlertSeverity.WARN,
                )
            ],
        )
        ing = IngestionLayer(cfg, db)
        det = DriftDetector(cfg, db)

        now = datetime.now(timezone.utc)
        for i in range(20):
            ing.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="resolve",
                resolution_time_ms=400,
                timestamp=now - timedelta(hours=50 - i),
            ))
        det.compute_baseline()

        for i in range(25):
            ing.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="deny",
                resolution_time_ms=400,
                timestamp=now - timedelta(minutes=25 - i),
            ))

        report = det.detect()
        assert report.overall_drift_score > 0.0
