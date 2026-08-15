"""Tests for alert engine: alert firing, rollback conditions, cooldown, safety gate."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from ai_governance.alerts import AlertEngine
from ai_governance.config import (
    AlertSeverity,
    DriftThreshold,
    GovernanceConfig,
    MetricDirection,
    RollbackCondition,
)
from ai_governance.drift import DriftDetector
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import DecisionRecord, GovernanceDB


def _ingest_n(
    ing: IngestionLayer,
    n: int,
    category: str,
    decision: str,
    base_dt: datetime,
    offset_hours: int = 0,
    ground_truth: str = None,
) -> list:
    return [
        ing.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category=category,
            decision=decision,
            resolution_time_ms=400,
            ground_truth=ground_truth,
            timestamp=base_dt - timedelta(hours=offset_hours + n - i),
        ))
        for i in range(n)
    ]


def _insert_hr_records(
    db: GovernanceDB,
    agent_id: str,
    n: int,
    category: str,
    decision: str,
    ground_truth: str,
    now: datetime,
    offset_hours: int = 0,
) -> None:
    """Insert records directly (with is_high_risk=True and ground_truth) for accuracy tests."""
    for i in range(n):
        db.insert_decision(DecisionRecord(
            event_id=str(uuid.uuid4()),
            agent_id=agent_id,
            timestamp=now - timedelta(hours=offset_hours + n - i),
            case_id=str(uuid.uuid4()),
            case_category=category,
            is_high_risk=True,
            high_risk_score=0.5,
            decision=decision,
            resolution_time_ms=400,
            ground_truth=ground_truth,
        ))


def _setup(
    agent_id: str = "alert-test",
    extra_thresholds: list = None,
    extra_conditions: list = None,
) -> tuple[GovernanceConfig, GovernanceDB, IngestionLayer, DriftDetector, AlertEngine]:
    cfg = GovernanceConfig(
        agent_id=agent_id,
        min_baseline_events=10,
        recent_window_size=20,
        drift_thresholds=extra_thresholds or [],
        rollback_conditions=extra_conditions or [],
    )
    db = GovernanceDB(":memory:")
    ing = IngestionLayer(cfg, db)
    det = DriftDetector(cfg, db)
    eng = AlertEngine(cfg, db)
    return cfg, db, ing, det, eng


class TestSafetyGate:
    def test_safe_when_no_rollbacks(self, engine):
        safe, _ = engine.is_agent_safe()
        assert safe is True

    def test_unsafe_after_rollback_record(self, config, db, engine):
        from ai_governance.storage import RollbackRecord
        db.insert_rollback(RollbackRecord(
            rollback_id=str(uuid.uuid4()),
            agent_id=config.agent_id,
            timestamp=datetime.now(timezone.utc),
            trigger_rule="fraud_underescalation",
            reason="test rollback",
        ))
        safe, reason = engine.is_agent_safe()
        assert safe is False
        assert "fraud_underescalation" in reason

    def test_reason_contains_rule_name(self, config, db, engine):
        from ai_governance.storage import RollbackRecord
        db.insert_rollback(RollbackRecord(
            rollback_id=str(uuid.uuid4()),
            agent_id=config.agent_id,
            timestamp=datetime.now(timezone.utc),
            trigger_rule="my_custom_rule",
            reason="custom reason text",
        ))
        _, reason = engine.is_agent_safe()
        assert "my_custom_rule" in reason


class TestThresholdAlerts:
    def test_violation_fires_alert(self):
        cfg, db, ing, det, eng = _setup(
            extra_thresholds=[
                DriftThreshold(
                    name="esc_spike",
                    category="billing_dispute",
                    metric="escalation_rate",
                    max_delta=0.10,
                    direction=MetricDirection.INCREASE,
                    min_baseline_samples=10,
                    recent_window=20,
                    severity=AlertSeverity.CRITICAL,
                )
            ]
        )
        now = datetime.now(timezone.utc)
        _ingest_n(ing, 15, "billing_dispute", "resolve", now, offset_hours=20)
        det.compute_baseline()
        _ingest_n(ing, 25, "billing_dispute", "escalate", now)

        report = det.detect()
        fired = eng.evaluate(report)
        assert len(fired) >= 1
        assert any(f.severity == "critical" for f in fired)

    def test_rollback_severity_writes_rollback_record(self):
        """Threshold at ROLLBACK severity creates a rollback record when breached."""
        cfg, db, ing, det, eng = _setup(
            extra_thresholds=[
                DriftThreshold(
                    name="hr_acc_floor",
                    category="fraud_claim",
                    metric="high_risk_accuracy",
                    max_delta=0.05,
                    direction=MetricDirection.DECREASE,
                    min_baseline_samples=10,
                    recent_window=15,
                    severity=AlertSeverity.ROLLBACK,
                )
            ]
        )
        now = datetime.now(timezone.utc)
        # Baseline: all high-risk with perfect accuracy (inserted directly for control)
        _insert_hr_records(db, cfg.agent_id, 15, "fraud_claim", "escalate", "escalate", now, 30)
        det.compute_baseline()

        # Recent: high-risk with wrong decisions (accuracy → 0)
        _insert_hr_records(db, cfg.agent_id, 20, "fraud_claim", "resolve", "escalate", now)

        report = det.detect()
        fired = eng.evaluate(report)
        rollback_fired = [f for f in fired if f.triggered_rollback]
        assert len(rollback_fired) >= 1
        assert db.has_active_rollback(cfg.agent_id)

    def test_no_alert_when_below_threshold(self):
        cfg, db, ing, det, eng = _setup(
            extra_thresholds=[
                DriftThreshold(
                    name="big_threshold",
                    category="billing_dispute",
                    metric="escalation_rate",
                    max_delta=0.50,  # very permissive
                    direction=MetricDirection.INCREASE,
                    min_baseline_samples=10,
                    recent_window=20,
                    severity=AlertSeverity.WARN,
                )
            ]
        )
        now = datetime.now(timezone.utc)
        _ingest_n(ing, 15, "billing_dispute", "resolve", now, offset_hours=20)
        det.compute_baseline()
        # Small increase — within threshold
        for i in range(25):
            ing.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="escalate" if i % 10 == 0 else "resolve",
                resolution_time_ms=400,
                timestamp=now - timedelta(minutes=25 - i),
            ))
        report = det.detect()
        fired = eng.evaluate(report)
        assert len(fired) == 0


class TestRollbackConditions:
    def test_condition_triggers_rollback_on_expression_true(self):
        cfg, db, ing, det, eng = _setup(
            extra_conditions=[
                RollbackCondition(
                    name="fraud_esc_too_low",
                    description="fraud not escalated enough",
                    expression=(
                        "fraud_claim_escalation_rate < 0.70 "
                        "and fraud_claim_count >= 10"
                    ),
                    cooldown_seconds=60,
                )
            ]
        )
        now = datetime.now(timezone.utc)
        _ingest_n(ing, 15, "fraud_claim", "resolve", now)  # 0% escalation

        report = det.detect()
        fired = eng.evaluate(report)
        rollbacks = [f for f in fired if f.triggered_rollback]
        assert len(rollbacks) >= 1
        assert db.has_active_rollback(cfg.agent_id)

    def test_condition_does_not_trigger_when_false(self):
        cfg, db, ing, det, eng = _setup(
            extra_conditions=[
                RollbackCondition(
                    name="impossible_rule",
                    description="never triggers",
                    expression="fraud_claim_count > 1000000",
                    cooldown_seconds=60,
                )
            ]
        )
        now = datetime.now(timezone.utc)
        _ingest_n(ing, 10, "fraud_claim", "escalate", now)
        report = det.detect()
        fired = eng.evaluate(report)
        assert not fired
        assert not db.has_active_rollback(cfg.agent_id)

    def test_cooldown_prevents_duplicate_alerts(self):
        cfg, db, ing, det, eng = _setup(
            extra_conditions=[
                RollbackCondition(
                    name="always_fires",
                    description="always",
                    expression="overall_drift_score >= 0",  # always True
                    cooldown_seconds=3600,
                )
            ]
        )
        now = datetime.now(timezone.utc)
        _ingest_n(ing, 5, "billing_dispute", "resolve", now)
        report = det.detect()
        fired1 = eng.evaluate(report)
        fired2 = eng.evaluate(report)  # second call within cooldown
        first_rollbacks = [f for f in fired1 if f.triggered_rollback]
        second_rollbacks = [f for f in fired2 if f.triggered_rollback]
        assert len(first_rollbacks) >= 1
        assert len(second_rollbacks) == 0  # cooldown suppresses re-fire

    def test_expression_runtime_error_does_not_crash(self):
        cfg, db, ing, det, eng = _setup()
        cfg.rollback_conditions.append(
            RollbackCondition(
                name="div_by_zero",
                description="errors at runtime",
                expression="1 > 0",  # valid syntax but we'll corrupt it manually
                cooldown_seconds=60,
            )
        )
        # Manually corrupt the expression after validation
        cfg.rollback_conditions[-1].__dict__["expression"] = "1 / 0 > 0"
        now = datetime.now(timezone.utc)
        _ingest_n(ing, 5, "billing_dispute", "resolve", now)
        report = det.detect()
        fired = eng.evaluate(report)  # must not raise
        bad = [f for f in fired if f.rule_name == "div_by_zero"]
        assert not bad  # error suppressed, no spurious alert

    def test_alerts_written_to_db(self):
        cfg, db, ing, det, eng = _setup(
            extra_thresholds=[
                DriftThreshold(
                    name="watch",
                    category="billing_dispute",
                    metric="escalation_rate",
                    max_delta=0.10,
                    direction=MetricDirection.INCREASE,
                    min_baseline_samples=10,
                    recent_window=20,
                    severity=AlertSeverity.WARN,
                )
            ]
        )
        now = datetime.now(timezone.utc)
        _ingest_n(ing, 15, "billing_dispute", "resolve", now, offset_hours=20)
        det.compute_baseline()
        _ingest_n(ing, 25, "billing_dispute", "escalate", now)

        report = det.detect()
        fired = eng.evaluate(report)
        alerts = db.get_recent_alerts(cfg.agent_id)
        assert len(alerts) == len(fired)
        assert all(a.agent_id == cfg.agent_id for a in alerts)

    def test_context_exposes_category_metrics(self):
        """Verify rollback expression has access to per-category variables."""
        cfg, db, ing, det, eng = _setup(
            extra_conditions=[
                RollbackCondition(
                    name="billing_high_esc",
                    description="billing escalation too high",
                    expression="billing_dispute_escalation_rate > 0.80 and billing_dispute_count >= 5",
                    cooldown_seconds=60,
                )
            ]
        )
        now = datetime.now(timezone.utc)
        # 10 billing escalations
        _ingest_n(ing, 10, "billing_dispute", "escalate", now)
        report = det.detect()
        fired = eng.evaluate(report)
        rollbacks = [f for f in fired if f.triggered_rollback]
        assert len(rollbacks) >= 1
