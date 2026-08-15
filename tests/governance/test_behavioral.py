"""Behavioral / drift regression test suite.

50 fixed agent decisions across five case categories serve as the
governance regression dataset.  Every commit must pass these tests.
If they fail, the drift logic has regressed — or the dataset itself
has exposed a new violation that should be actioned.

Dataset layout
--------------
  Billing Disputes  (BD-01 … BD-10)  — medium/high risk, balanced decisions
  Fraud Claims      (FC-01 … FC-10)  — high/critical risk, must be escalated
  Policy Issues     (PI-01 … PI-10)  — high risk, sensitive decisions
  Routine Cases     (RT-01 … RT-10)  — low risk, fast resolves
  Edge Cases        (EC-01 … EC-10)  — ambiguous; tests boundary conditions

Assertions per test class
-------------------------
  1. Every decision parses to a valid GovernanceDecision (structural test)
  2. High-risk categories carry high/critical risk levels (categorisation test)
  3. Healthy-baseline → no drift detected (no violations, no alerts)
  4. Drifted recent window → violations correctly fired (regression test)
  5. Rollback engine triggers when conditions are met (safety test)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

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
from ai_governance.ingestion import IngestionLayer
from ai_governance.storage import GovernanceDB
from ai_governance.structured import DecisionDecoder, GovernanceDecision


# ── fixed dataset ─────────────────────────────────────────────────────────────

FIXED_DATASET: list[dict[str, Any]] = [
    # ── Billing Disputes (10) ────────────────────────────────────────────────
    # Healthy pattern: ~70% resolve, ~30% escalate, medium-high risk
    {"id": "BD-01", "case_category": "billing_dispute", "risk_level": "high",
     "decision": "escalate", "confidence": 0.92, "flags": ["overdue_90_days"]},
    {"id": "BD-02", "case_category": "billing_dispute", "risk_level": "medium",
     "decision": "resolve", "confidence": 0.85, "flags": []},
    {"id": "BD-03", "case_category": "billing_dispute", "risk_level": "high",
     "decision": "escalate", "confidence": 0.88, "flags": ["duplicate_charge"]},
    {"id": "BD-04", "case_category": "billing_dispute", "risk_level": "medium",
     "decision": "resolve", "confidence": 0.79, "flags": []},
    {"id": "BD-05", "case_category": "billing_dispute", "risk_level": "medium",
     "decision": "partial_resolve", "confidence": 0.72, "flags": ["partial_credit"]},
    {"id": "BD-06", "case_category": "billing_dispute", "risk_level": "medium",
     "decision": "resolve", "confidence": 0.91, "flags": []},
    {"id": "BD-07", "case_category": "billing_dispute", "risk_level": "high",
     "decision": "defer", "confidence": 0.60, "flags": ["awaiting_bank_confirmation"]},
    {"id": "BD-08", "case_category": "billing_dispute", "risk_level": "medium",
     "decision": "resolve", "confidence": 0.83, "flags": []},
    {"id": "BD-09", "case_category": "billing_dispute", "risk_level": "medium",
     "decision": "resolve", "confidence": 0.77, "flags": []},
    {"id": "BD-10", "case_category": "billing_dispute", "risk_level": "high",
     "decision": "escalate", "confidence": 0.94, "flags": ["enterprise_account", "legal_threat"]},

    # ── Fraud Claims (10) ────────────────────────────────────────────────────
    # Healthy pattern: ≥80% escalate, high/critical risk
    {"id": "FC-01", "case_category": "fraud_claim", "risk_level": "critical",
     "decision": "escalate", "confidence": 0.98, "flags": ["confirmed_fraud_pattern"]},
    {"id": "FC-02", "case_category": "fraud_claim", "risk_level": "critical",
     "decision": "escalate", "confidence": 0.96, "flags": ["card_not_present", "geo_mismatch"]},
    {"id": "FC-03", "case_category": "fraud_claim", "risk_level": "high",
     "decision": "escalate", "confidence": 0.89, "flags": ["velocity_flag"]},
    {"id": "FC-04", "case_category": "fraud_claim", "risk_level": "critical",
     "decision": "escalate", "confidence": 0.99, "flags": ["identity_theft_suspected"]},
    {"id": "FC-05", "case_category": "fraud_claim", "risk_level": "high",
     "decision": "escalate", "confidence": 0.87, "flags": ["new_account"]},
    {"id": "FC-06", "case_category": "fraud_claim", "risk_level": "high",
     "decision": "escalate", "confidence": 0.91, "flags": ["unusual_merchant"]},
    {"id": "FC-07", "case_category": "fraud_claim", "risk_level": "critical",
     "decision": "escalate", "confidence": 0.97, "flags": ["chargeback_history"]},
    {"id": "FC-08", "case_category": "fraud_claim", "risk_level": "high",
     "decision": "escalate", "confidence": 0.84, "flags": []},
    # Two resolved fraud claims (low confidence, genuinely benign)
    {"id": "FC-09", "case_category": "fraud_claim", "risk_level": "medium",
     "decision": "resolve", "confidence": 0.61, "flags": ["customer_confirmed_transaction"]},
    {"id": "FC-10", "case_category": "fraud_claim", "risk_level": "medium",
     "decision": "resolve", "confidence": 0.68, "flags": ["merchant_verified"]},

    # ── Policy Issues (10) ───────────────────────────────────────────────────
    # High risk, policy/legal/compliance — mixed decisions, mostly escalate/defer
    {"id": "PI-01", "case_category": "policy_issue", "risk_level": "high",
     "decision": "escalate", "confidence": 0.90, "flags": ["gdpr_data_request"]},
    {"id": "PI-02", "case_category": "legal_matter", "risk_level": "critical",
     "decision": "defer", "confidence": 0.80, "flags": ["legal_hold", "attorney_contact"]},
    {"id": "PI-03", "case_category": "compliance_breach", "risk_level": "high",
     "decision": "escalate", "confidence": 0.93, "flags": ["pci_scope"]},
    {"id": "PI-04", "case_category": "policy_issue", "risk_level": "high",
     "decision": "deny", "confidence": 0.88, "flags": ["tos_violation"]},
    {"id": "PI-05", "case_category": "policy_issue", "risk_level": "medium",
     "decision": "resolve", "confidence": 0.74, "flags": ["low_severity_infraction"]},
    {"id": "PI-06", "case_category": "legal_matter", "risk_level": "critical",
     "decision": "defer", "confidence": 0.85, "flags": ["pending_litigation"]},
    {"id": "PI-07", "case_category": "compliance_breach", "risk_level": "high",
     "decision": "escalate", "confidence": 0.95, "flags": ["sox_related"]},
    {"id": "PI-08", "case_category": "policy_issue", "risk_level": "high",
     "decision": "escalate", "confidence": 0.82, "flags": ["repeat_offender"]},
    {"id": "PI-09", "case_category": "policy_issue", "risk_level": "medium",
     "decision": "partial_resolve", "confidence": 0.70, "flags": []},
    {"id": "PI-10", "case_category": "policy_issue", "risk_level": "high",
     "decision": "deny", "confidence": 0.89, "flags": ["abuse_pattern"]},

    # ── Routine Cases (10) ───────────────────────────────────────────────────
    # Low risk, fast resolves — should never trigger governance alerts
    {"id": "RT-01", "case_category": "shipping_inquiry", "risk_level": "low",
     "decision": "resolve", "confidence": 0.99, "flags": []},
    {"id": "RT-02", "case_category": "product_question", "risk_level": "low",
     "decision": "resolve", "confidence": 0.98, "flags": []},
    {"id": "RT-03", "case_category": "shipping_inquiry", "risk_level": "low",
     "decision": "resolve", "confidence": 0.97, "flags": []},
    {"id": "RT-04", "case_category": "account_update", "risk_level": "low",
     "decision": "resolve", "confidence": 0.99, "flags": []},
    {"id": "RT-05", "case_category": "product_question", "risk_level": "low",
     "decision": "resolve", "confidence": 0.96, "flags": []},
    {"id": "RT-06", "case_category": "shipping_inquiry", "risk_level": "low",
     "decision": "resolve", "confidence": 0.95, "flags": []},
    {"id": "RT-07", "case_category": "account_update", "risk_level": "low",
     "decision": "resolve", "confidence": 0.98, "flags": []},
    {"id": "RT-08", "case_category": "product_question", "risk_level": "low",
     "decision": "resolve", "confidence": 0.97, "flags": []},
    {"id": "RT-09", "case_category": "shipping_inquiry", "risk_level": "low",
     "decision": "resolve", "confidence": 0.99, "flags": []},
    {"id": "RT-10", "case_category": "account_update", "risk_level": "medium",
     "decision": "defer", "confidence": 0.65, "flags": ["verification_required"]},

    # ── Edge Cases (10) ──────────────────────────────────────────────────────
    # Boundary conditions, mixed signals, multiple high-risk flags
    {"id": "EC-01", "case_category": "billing_dispute", "risk_level": "critical",
     "decision": "escalate", "confidence": 0.50,
     "flags": ["enterprise_account", "legal_threat", "repeat_escalation"]},
    {"id": "EC-02", "case_category": "fraud_claim", "risk_level": "high",
     "decision": "defer", "confidence": 0.55,
     "flags": ["awaiting_police_report"]},
    {"id": "EC-03", "case_category": "shipping_inquiry", "risk_level": "medium",
     "decision": "escalate", "confidence": 0.62,
     "flags": ["high_value_shipment", "insurance_claim"]},
    {"id": "EC-04", "case_category": "account_update", "risk_level": "high",
     "decision": "deny", "confidence": 0.71,
     "flags": ["suspicious_ip", "multiple_failed_attempts"]},
    {"id": "EC-05", "case_category": "fraud_claim", "risk_level": "low",
     "decision": "resolve", "confidence": 0.90,
     "flags": ["customer_verified", "amount_below_threshold"]},
    {"id": "EC-06", "case_category": "policy_issue", "risk_level": "critical",
     "decision": "escalate", "confidence": 0.99,
     "flags": ["regulatory_report_required", "pci_scope", "gdpr_data_request"]},
    {"id": "EC-07", "case_category": "billing_dispute", "risk_level": "low",
     "decision": "resolve", "confidence": 0.95,
     "flags": []},
    {"id": "EC-08", "case_category": "product_question", "risk_level": "high",
     "decision": "escalate", "confidence": 0.78,
     "flags": ["product_safety_concern"]},
    {"id": "EC-09", "case_category": "fraud_claim", "risk_level": "critical",
     "decision": "escalate", "confidence": 0.0,
     "flags": ["model_uncertain", "human_review_required"]},
    {"id": "EC-10", "case_category": "account_update", "risk_level": "medium",
     "decision": "partial_resolve", "confidence": 0.55,
     "flags": ["partial_verification"]},
]

assert len(FIXED_DATASET) == 50, f"Expected 50 decisions, got {len(FIXED_DATASET)}"

_decoder = DecisionDecoder()


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_governance_decision(raw: dict) -> GovernanceDecision:
    payload = {k: v for k, v in raw.items() if k != "id"}
    return _decoder.decode(payload)


def _make_tight_config(baseline_events: int = 10, recent_window: int = 15) -> GovernanceConfig:
    """Config tuned for behavioral tests (small windows so 10-event batches work)."""
    return GovernanceConfig(
        agent_id="behavioral-test-agent",
        min_baseline_events=baseline_events,
        recent_window_size=recent_window * 2,
        high_risk_patterns=GovernanceConfig.default_customer_service().high_risk_patterns,
        drift_thresholds=[
            DriftThreshold(
                name="fraud_esc_drop",
                category="fraud_claim",
                metric="escalation_rate",
                max_delta=0.15,
                direction=MetricDirection.DECREASE,
                min_baseline_samples=baseline_events,
                recent_window=recent_window,
                severity=AlertSeverity.CRITICAL,
            ),
            DriftThreshold(
                name="billing_resolution_drop",
                category="billing_dispute",
                metric="resolution_rate",
                max_delta=0.20,
                direction=MetricDirection.DECREASE,
                min_baseline_samples=baseline_events,
                recent_window=recent_window,
                severity=AlertSeverity.WARN,
            ),
        ],
        rollback_conditions=[
            RollbackCondition(
                name="fraud_underescalation",
                description="Fraud not being escalated — financial risk threshold breached",
                expression="fraud_claim_escalation_rate < 0.50 and fraud_claim_count >= 5",
                cooldown_seconds=60,
            ),
        ],
    )


def _seed_from_dataset(
    ingestion: IngestionLayer,
    decisions: list[dict],
    hours_offset: int = 0,
) -> None:
    """Inject a list of FIXED_DATASET rows into the ingestion layer with temporal offsets."""
    now = datetime.now(timezone.utc)
    for i, row in enumerate(decisions):
        gov = _to_governance_decision(row)
        ts = now - timedelta(hours=hours_offset + len(decisions) - i)
        ingestion.ingest_structured(
            gov,
            case_id=f"{row['id']}-{uuid.uuid4().hex[:6]}",
            timestamp=ts,
        )


# ── 1. Structural tests: all 50 decisions must parse ────────────────────────

class TestFixedDatasetStructure:
    """Every decision in the fixed dataset must be a parseable GovernanceDecision."""

    def test_all_50_parse_to_governance_decision(self):
        for row in FIXED_DATASET:
            decision = _to_governance_decision(row)
            assert isinstance(decision, GovernanceDecision), \
                f"{row['id']} failed to parse"

    def test_all_50_have_non_empty_category(self):
        for row in FIXED_DATASET:
            d = _to_governance_decision(row)
            assert d.case_category.strip(), f"{row['id']} has empty case_category"

    def test_all_50_confidence_in_bounds(self):
        for row in FIXED_DATASET:
            d = _to_governance_decision(row)
            assert 0.0 <= d.confidence <= 1.0, \
                f"{row['id']} confidence {d.confidence} out of bounds"

    def test_all_50_valid_risk_levels(self):
        valid = {"low", "medium", "high", "critical"}
        for row in FIXED_DATASET:
            d = _to_governance_decision(row)
            assert d.risk_level in valid, \
                f"{row['id']} has invalid risk_level {d.risk_level!r}"

    def test_all_50_valid_decisions(self):
        valid = {"resolve", "escalate", "deny", "defer", "partial_resolve"}
        for row in FIXED_DATASET:
            d = _to_governance_decision(row)
            assert d.decision in valid, \
                f"{row['id']} has invalid decision {d.decision!r}"

    def test_dataset_has_exactly_50_decisions(self):
        assert len(FIXED_DATASET) == 50

    def test_dataset_has_expected_category_counts(self):
        by_prefix = {}
        for row in FIXED_DATASET:
            prefix = row["id"][:2]
            by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
        assert by_prefix["BD"] == 10, "Expected 10 billing dispute entries"
        assert by_prefix["FC"] == 10, "Expected 10 fraud claim entries"
        assert by_prefix["PI"] == 10, "Expected 10 policy issue entries"
        assert by_prefix["RT"] == 10, "Expected 10 routine entries"
        assert by_prefix["EC"] == 10, "Expected 10 edge case entries"


# ── 2. High-risk categorisation tests ───────────────────────────────────────

class TestHighRiskCategorisation:
    """High-risk case types must have high/critical risk levels in the dataset."""

    def _rows_by_prefix(self, prefix: str) -> list[dict]:
        return [r for r in FIXED_DATASET if r["id"].startswith(prefix)]

    def test_fraud_claims_are_high_or_critical(self):
        fraud = self._rows_by_prefix("FC")
        for row in fraud:
            d = _to_governance_decision(row)
            # At least 8 of 10 must be high/critical (2 are confirmed-benign)
        high_count = sum(
            1 for r in fraud if _to_governance_decision(r).risk_level in ("high", "critical")
        )
        assert high_count >= 8, \
            f"Only {high_count}/10 fraud claims rated high/critical"

    def test_billing_disputes_not_all_low(self):
        billing = self._rows_by_prefix("BD")
        low_count = sum(
            1 for r in billing if _to_governance_decision(r).risk_level == "low"
        )
        assert low_count < 5, \
            f"{low_count}/10 billing disputes rated low — too many"

    def test_routine_cases_mostly_low(self):
        routine = self._rows_by_prefix("RT")
        low_count = sum(
            1 for r in routine if _to_governance_decision(r).risk_level == "low"
        )
        assert low_count >= 8, \
            f"Only {low_count}/10 routine cases rated low"

    def test_fraud_claims_mostly_escalated(self):
        fraud = self._rows_by_prefix("FC")
        escalated = sum(
            1 for r in fraud if _to_governance_decision(r).decision == "escalate"
        )
        assert escalated >= 7, \
            f"Only {escalated}/10 fraud claims escalated — governance risk!"

    def test_policy_issues_not_routine(self):
        policy = self._rows_by_prefix("PI")
        low_count = sum(
            1 for r in policy if _to_governance_decision(r).risk_level == "low"
        )
        assert low_count == 0, \
            f"{low_count} policy issues rated low — misclassification!"

    def test_is_high_risk_property_consistent(self):
        for row in FIXED_DATASET:
            d = _to_governance_decision(row)
            if d.risk_level in ("high", "critical"):
                assert d.is_high_risk is True, f"{row['id']} should be is_high_risk"
            else:
                assert d.is_high_risk is False, f"{row['id']} should not be is_high_risk"


# ── 3. Drift score tests: healthy baseline → no violations ───────────────────

class TestDriftScoresHealthyBaseline:
    """Feeding the full healthy dataset as both baseline and recent should
    produce zero violations — drift scores within thresholds."""

    @pytest.fixture
    def system(self):
        cfg = _make_tight_config(baseline_events=10, recent_window=15)
        db = GovernanceDB(":memory:")
        ingestion = IngestionLayer(cfg, db)
        detector = DriftDetector(cfg, db)
        engine = AlertEngine(cfg, db)
        return ingestion, detector, engine, cfg

    def test_no_drift_when_baseline_and_recent_match(self, system):
        ingestion, detector, engine, cfg = system
        fraud_rows = [r for r in FIXED_DATASET if r["id"].startswith("FC")]

        # Seed baseline: all 10 fraud rows as old events
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=50)
        detector.compute_baseline()

        # Seed recent: same distribution (8 escalate, 2 resolve)
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=0)

        report = detector.detect()
        violations = [v for v in report.violations if v.category == "fraud_claim"]
        assert len(violations) == 0, \
            f"Unexpected violations on matching baseline: {[v.rule_name for v in violations]}"

    def test_overall_drift_score_near_zero_healthy(self, system):
        ingestion, detector, engine, cfg = system
        billing_rows = [r for r in FIXED_DATASET if r["id"].startswith("BD")]

        _seed_from_dataset(ingestion, billing_rows, hours_offset=50)
        detector.compute_baseline()
        _seed_from_dataset(ingestion, billing_rows, hours_offset=0)

        report = detector.detect()
        assert report.overall_drift_score < 0.30, \
            f"Healthy dataset produced unexpectedly high drift: {report.overall_drift_score:.3f}"

    def test_routine_cases_never_trigger_alerts(self, system):
        ingestion, detector, engine, cfg = system
        routine = [r for r in FIXED_DATASET if r["id"].startswith("RT")]

        _seed_from_dataset(ingestion, routine, hours_offset=50)
        detector.compute_baseline()
        _seed_from_dataset(ingestion, routine, hours_offset=0)

        report = detector.detect()
        fired = engine.evaluate(report)
        assert len(fired) == 0, \
            f"Routine cases fired alerts: {[f.rule_name for f in fired]}"


# ── 4. Drift detection: drifted recent window triggers violations ─────────────

class TestDriftDetectionDrifted:
    """After a healthy baseline, a drifted recent window must fire violations."""

    @pytest.fixture
    def system(self):
        cfg = _make_tight_config(baseline_events=10, recent_window=15)
        db = GovernanceDB(":memory:")
        ingestion = IngestionLayer(cfg, db)
        detector = DriftDetector(cfg, db)
        engine = AlertEngine(cfg, db)
        return ingestion, detector, engine, cfg

    def _drifted_fraud_rows(self) -> list[dict]:
        """Replace escalate with resolve for fraud — simulates the drift pattern."""
        drifted = []
        for row in FIXED_DATASET:
            if not row["id"].startswith("FC"):
                continue
            drifted.append({**row, "decision": "resolve", "risk_level": "medium"})
        return drifted

    def test_fraud_drift_fires_critical_alert(self, system):
        ingestion, detector, engine, cfg = system

        # Healthy baseline
        fraud_rows = [r for r in FIXED_DATASET if r["id"].startswith("FC")]
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=50)
        detector.compute_baseline()

        # Drifted recent: all fraud resolved instead of escalated
        _seed_from_dataset(ingestion, self._drifted_fraud_rows(), hours_offset=0)

        report = detector.detect()
        violations = [v for v in report.violations if v.category == "fraud_claim"]
        assert len(violations) >= 1, "Expected fraud escalation drop violation"
        assert any(v.severity in ("critical", "rollback") for v in violations), \
            f"Violation severity not critical/rollback: {[v.severity for v in violations]}"

    def test_drifted_fraud_triggers_rollback_condition(self, system):
        ingestion, detector, engine, cfg = system

        fraud_rows = [r for r in FIXED_DATASET if r["id"].startswith("FC")]
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=50)
        detector.compute_baseline()
        _seed_from_dataset(ingestion, self._drifted_fraud_rows(), hours_offset=0)

        report = detector.detect()
        fired = engine.evaluate(report)
        rollback_alerts = [f for f in fired if f.triggered_rollback]
        assert len(rollback_alerts) >= 1, \
            "Expected rollback to trigger on drifted fraud dataset"

    def test_agent_unsafe_after_drift(self, system):
        ingestion, detector, engine, cfg = system

        fraud_rows = [r for r in FIXED_DATASET if r["id"].startswith("FC")]
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=50)
        detector.compute_baseline()
        _seed_from_dataset(ingestion, self._drifted_fraud_rows(), hours_offset=0)

        report = detector.detect()
        engine.evaluate(report)
        is_safe, reason = engine.is_agent_safe()
        assert is_safe is False, f"Agent should be unsafe after fraud drift; reason: {reason}"

    def test_drift_delta_direction(self, system):
        ingestion, detector, engine, cfg = system

        fraud_rows = [r for r in FIXED_DATASET if r["id"].startswith("FC")]
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=50)
        detector.compute_baseline()
        _seed_from_dataset(ingestion, self._drifted_fraud_rows(), hours_offset=0)

        report = detector.detect()
        fraud_violations = [v for v in report.violations if v.category == "fraud_claim"]
        for v in fraud_violations:
            assert v.delta < 0, \
                f"Escalation rate should have decreased (delta<0), got {v.delta}"


# ── 5. Rollback engine safety tests ──────────────────────────────────────────

class TestRollbackEngineSafety:
    """The rollback engine must fire and block agent when safety conditions met."""

    @pytest.fixture
    def system(self):
        cfg = _make_tight_config(baseline_events=10, recent_window=15)
        db = GovernanceDB(":memory:")
        ingestion = IngestionLayer(cfg, db)
        detector = DriftDetector(cfg, db)
        engine = AlertEngine(cfg, db)
        return ingestion, detector, engine, db

    def test_safe_before_any_drift(self, system):
        _, _, engine, _ = system
        is_safe, _ = engine.is_agent_safe()
        assert is_safe is True

    def test_rollback_blocks_agent_until_resolved(self, system):
        ingestion, detector, engine, db = system

        # Healthy baseline
        fraud_rows = [r for r in FIXED_DATASET if r["id"].startswith("FC")]
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=50)
        detector.compute_baseline()

        # All fraud cases resolved — rollback condition triggers
        drifted = [{**r, "decision": "resolve"} for r in fraud_rows]
        _seed_from_dataset(ingestion, drifted, hours_offset=0)
        report = detector.detect()
        engine.evaluate(report)

        is_safe, reason = engine.is_agent_safe()
        assert is_safe is False
        assert "rollback" in reason.lower()

        # PM resolves the rollback
        rollbacks = db.get_rollbacks("behavioral-test-agent", limit=5)
        assert len(rollbacks) >= 1
        db.resolve_rollback("behavioral-test-agent", rollbacks[0].rollback_id, "pm@test.com")

        is_safe_after, _ = engine.is_agent_safe()
        assert is_safe_after is True

    def test_cooldown_prevents_repeat_rollback(self, system):
        ingestion, detector, engine, db = system

        fraud_rows = [r for r in FIXED_DATASET if r["id"].startswith("FC")]
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=50)
        detector.compute_baseline()

        drifted = [{**r, "decision": "resolve"} for r in fraud_rows]
        _seed_from_dataset(ingestion, drifted, hours_offset=0)

        report = detector.detect()
        fired_first = engine.evaluate(report)
        rb_count_first = sum(1 for f in fired_first if f.triggered_rollback)

        # Second evaluation without resolving — cooldown prevents duplicate rollback
        report2 = detector.detect()
        fired_second = engine.evaluate(report2)
        rb_count_second = sum(1 for f in fired_second if f.triggered_rollback)

        assert rb_count_second == 0, \
            f"Cooldown should prevent repeat rollback; got {rb_count_second}"


# ── 6. Structured ingestion flow through full system ─────────────────────────

class TestStructuredIngestionEndToEnd:
    """GovernanceDecision objects flow correctly through the full governance stack."""

    def test_all_50_ingest_via_ingest_structured(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ingestion = IngestionLayer(cfg, db)

        for row in FIXED_DATASET:
            gov = _to_governance_decision(row)
            result = ingestion.ingest_structured(gov, case_id=row["id"])
            assert result.event_id, f"{row['id']} produced no event_id"

        decisions = db.get_recent_decisions(cfg.agent_id, limit=200)
        assert len(decisions) == 50

    def test_high_risk_decisions_flagged_in_storage(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ingestion = IngestionLayer(cfg, db)

        for row in FIXED_DATASET:
            gov = _to_governance_decision(row)
            ingestion.ingest_structured(gov, case_id=row["id"])

        all_decisions = db.get_recent_decisions(cfg.agent_id, limit=200)
        high_risk_in_db = [d for d in all_decisions if d.is_high_risk]

        # fraud (FC) + billing (BD) + policy (PI) — most are high risk
        assert len(high_risk_in_db) >= 20, \
            f"Expected ≥20 high-risk decisions in DB, got {len(high_risk_in_db)}"

    def test_fraud_escalation_rate_computed_correctly(self):
        cfg = _make_tight_config(baseline_events=10, recent_window=15)
        db = GovernanceDB(":memory:")
        ingestion = IngestionLayer(cfg, db)
        detector = DriftDetector(cfg, db)

        fraud_rows = [r for r in FIXED_DATASET if r["id"].startswith("FC")]
        _seed_from_dataset(ingestion, fraud_rows, hours_offset=0)

        report = detector.detect()
        if "fraud_claim" in report.recent_metrics:
            m = report.recent_metrics["fraud_claim"]
            # 8 escalate / 10 total = 0.80 escalation rate
            assert abs(m.escalation_rate - 0.80) < 0.05, \
                f"Expected ~80% fraud escalation rate, got {m.escalation_rate:.2%}"
