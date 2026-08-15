"""Tests for structured-decision drift metrics: agent_risk_calibration and avg_confidence."""

from __future__ import annotations

import uuid

import pytest

from ai_governance.config import GovernanceConfig, DriftThreshold, AlertSeverity, MetricDirection
from ai_governance.drift import DriftDetector, _compute_metrics
from ai_governance.ingestion import IngestionLayer, IngestRequest
from ai_governance.storage import DecisionRecord, GovernanceDB
from ai_governance.structured import GovernanceDecision
from datetime import datetime, timezone


def _make_structured_record(
    agent_id: str,
    category: str,
    decision: str,
    risk_level: str,
    confidence: float,
    is_high_risk: bool = False,
) -> DecisionRecord:
    return DecisionRecord(
        event_id=str(uuid.uuid4()),
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc),
        case_id=str(uuid.uuid4()),
        case_category=category,
        is_high_risk=is_high_risk,
        high_risk_score=0.7 if is_high_risk else 0.1,
        decision=decision,
        resolution_time_ms=100,
        metadata={"risk_level": risk_level, "confidence": confidence, "flags": []},
    )


class TestStructuredMetricsComputation:
    def test_calibration_computed_when_enough_structured_decisions(self):
        records = [
            _make_structured_record("ag", "fraud", "escalate", "critical", 0.9, is_high_risk=True)
            for _ in range(10)
        ]
        m = _compute_metrics(records, "fraud")
        assert m.agent_risk_calibration is not None
        assert 0.0 <= m.agent_risk_calibration <= 1.0

    def test_calibration_none_when_too_few_structured_decisions(self):
        records = [
            _make_structured_record("ag", "billing", "resolve", "low", 0.8, is_high_risk=False)
            for _ in range(4)
        ]
        m = _compute_metrics(records, "billing")
        assert m.agent_risk_calibration is None

    def test_calibration_perfect_when_all_agree(self):
        records = [
            _make_structured_record("ag", "fraud", "escalate", "critical", 0.9, is_high_risk=True)
            for _ in range(10)
        ]
        m = _compute_metrics(records, "fraud")
        assert m.agent_risk_calibration == 1.0

    def test_calibration_zero_when_all_disagree(self):
        records = [
            # Agent says high/critical, governance says NOT high-risk
            _make_structured_record("ag", "returns", "resolve", "critical", 0.9, is_high_risk=False)
            for _ in range(10)
        ]
        m = _compute_metrics(records, "returns")
        assert m.agent_risk_calibration == 0.0

    def test_calibration_partial_agreement(self):
        records = (
            [_make_structured_record("ag", "fraud", "escalate", "high", 0.9, is_high_risk=True)
             for _ in range(7)]
            +
            [_make_structured_record("ag", "fraud", "resolve", "critical", 0.8, is_high_risk=False)
             for _ in range(3)]  # agent says critical, governance says no
        )
        m = _compute_metrics(records, "fraud")
        assert m.agent_risk_calibration == pytest.approx(0.7)

    def test_avg_confidence_computed(self):
        records = [
            _make_structured_record("ag", "billing", "resolve", "low", conf, is_high_risk=False)
            for conf in [0.9, 0.8, 0.7, 0.85, 0.95, 0.6, 0.75, 0.8, 0.9, 0.7]
        ]
        m = _compute_metrics(records, "billing")
        assert m.avg_confidence is not None
        assert abs(m.avg_confidence - sum([0.9, 0.8, 0.7, 0.85, 0.95, 0.6, 0.75, 0.8, 0.9, 0.7]) / 10) < 0.001

    def test_avg_confidence_none_when_too_few(self):
        records = [
            _make_structured_record("ag", "billing", "resolve", "low", 0.9)
            for _ in range(4)
        ]
        m = _compute_metrics(records, "billing")
        assert m.avg_confidence is None

    def test_calibration_low_medium_treated_as_not_high_risk(self):
        records = [
            _make_structured_record("ag", "returns", "resolve", "low", 0.9, is_high_risk=False)
            for _ in range(5)
        ] + [
            _make_structured_record("ag", "returns", "resolve", "medium", 0.9, is_high_risk=False)
            for _ in range(5)
        ]
        m = _compute_metrics(records, "returns")
        assert m.agent_risk_calibration == 1.0

    def test_no_metadata_records_produce_none(self):
        records = []
        for i in range(10):
            r = DecisionRecord(
                event_id=str(uuid.uuid4()), agent_id="ag",
                timestamp=datetime.now(timezone.utc),
                case_id=str(uuid.uuid4()), case_category="returns",
                is_high_risk=False, high_risk_score=0.1,
                decision="resolve", resolution_time_ms=100,
                metadata={},
            )
            records.append(r)
        m = _compute_metrics(records, "returns")
        assert m.agent_risk_calibration is None
        assert m.avg_confidence is None


class TestStructuredDriftIntegration:
    def test_calibration_metric_usable_in_threshold(self):
        cfg = GovernanceConfig.default_customer_service()
        cfg.min_baseline_events = 5
        cfg.recent_window_size = 20
        cfg.drift_thresholds.append(DriftThreshold(
            name="risk_calibration_drop",
            category="*",
            metric="agent_risk_calibration",
            max_delta=0.15,
            direction=MetricDirection.DECREASE,
            min_baseline_samples=5,
            recent_window=10,
            severity=AlertSeverity.WARN,
        ))
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)

        # Baseline: well-calibrated agent
        for _ in range(10):
            gov = GovernanceDecision(
                case_category="fraud_claim",
                risk_level="critical",
                decision="escalate",
                confidence=0.9,
            )
            ing.ingest_structured(gov)

        det = DriftDetector(cfg, db)
        det.compute_baseline()

        report = det.detect()
        # With perfect calibration in both baseline and recent window, no violations
        assert all(v.rule_name != "risk_calibration_drop" for v in report.violations)

    def test_avg_confidence_in_category_metrics(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)

        for conf in [0.9, 0.85, 0.8, 0.95, 0.88, 0.92, 0.78, 0.83, 0.9, 0.87]:
            gov = GovernanceDecision(
                case_category="billing_dispute",
                risk_level="medium",
                decision="resolve",
                confidence=conf,
            )
            ing.ingest_structured(gov)

        det = DriftDetector(cfg, db)
        report = det.detect()
        billing = report.recent_metrics.get("billing_dispute")
        assert billing is not None
        assert billing.avg_confidence is not None
        assert 0.7 < billing.avg_confidence < 1.0

    def test_structured_and_raw_decisions_coexist(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)

        for _ in range(5):
            gov = GovernanceDecision(
                case_category="fraud_claim",
                risk_level="high",
                decision="escalate",
                confidence=0.85,
            )
            ing.ingest_structured(gov)

        for _ in range(5):
            ing.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="fraud_claim",
                decision="escalate",
                resolution_time_ms=100,
            ))

        det = DriftDetector(cfg, db)
        report = det.detect()
        fraud = report.recent_metrics.get("fraud_claim")
        assert fraud is not None
        assert fraud.total_events == 10
        # Calibration only from structured portion (5 records)
        assert fraud.agent_risk_calibration is not None

    def test_dashboard_exposes_calibration(self):
        from ai_governance.api import app, init_services
        from ai_governance.auth import reset as reset_auth
        from fastapi.testclient import TestClient

        reset_auth()
        cfg = GovernanceConfig.default_customer_service()
        cfg.min_baseline_events = 5
        cfg.recent_window_size = 20
        for thr in cfg.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 5
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        client = TestClient(app)

        for _ in range(10):
            client.post("/events/structured", json={
                "decision": {
                    "case_category": "fraud_claim",
                    "risk_level": "critical",
                    "decision": "escalate",
                    "confidence": 0.9,
                    "flags": [],
                }
            })

        resp = client.get("/dashboard")
        data = resp.json()
        fraud_rows = [c for c in data["category_breakdown"] if c["category"] == "fraud_claim"]
        assert len(fraud_rows) == 1
        assert "agent_risk_calibration" in fraud_rows[0]
        assert "avg_confidence" in fraud_rows[0]
        reset_auth()
