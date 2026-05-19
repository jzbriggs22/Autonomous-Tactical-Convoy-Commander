"""Alert and rollback engine.

Evaluates drift reports against configured thresholds and rollback conditions.
Fires alerts, writes rollback records, and exposes the `is_agent_safe` gate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import AlertSeverity, GovernanceConfig, RollbackCondition
from .drift import DriftReport
from .storage import AlertRecord, GovernanceDB, RollbackRecord


@dataclass
class AlertFired:
    alert_id: str
    rule_name: str
    severity: str
    message: str
    triggered_rollback: bool


class AlertEngine:
    """Evaluates a DriftReport, fires alerts, and triggers rollbacks."""

    def __init__(self, config: GovernanceConfig, db: GovernanceDB) -> None:
        self._config = config
        self._db = db

    def evaluate(self, report: DriftReport) -> list[AlertFired]:
        """Process a drift report. Returns all alerts fired during this evaluation."""
        fired: list[AlertFired] = []

        for violation in report.violations:
            fired.append(self._fire_threshold_alert(violation))

        ctx = self._build_expression_context(report)
        for condition in self._config.rollback_conditions:
            if not self._cooldown_ok(condition):
                continue
            try:
                triggered = bool(eval(condition.expression, {"__builtins__": {}}, ctx))  # noqa: S307
            except Exception:
                triggered = False
            if triggered:
                fired.append(self._fire_rollback(condition, ctx))

        return fired

    def is_agent_safe(self) -> tuple[bool, str]:
        """Returns (safe, reason). The PM dashboard calls this as the safety gate."""
        if self._db.has_active_rollback(self._config.agent_id):
            rollbacks = self._db.get_rollbacks(self._config.agent_id, limit=1)
            if rollbacks:
                r = rollbacks[0]
                return (
                    False,
                    f"Rollback active — rule '{r.trigger_rule}': {r.reason}",
                )
            return False, "Rollback active (no detail available)"
        return True, "All governance thresholds within bounds"

    # ── private ──────────────────────────────────────────────────────────────

    def _fire_threshold_alert(self, violation) -> AlertFired:
        alert_id = str(uuid.uuid4())
        direction = "increased" if violation.delta > 0 else "decreased"
        msg = (
            f"[{violation.severity.upper()}] {violation.rule_name}: "
            f"'{violation.metric}' {direction} by {abs(violation.delta):.3f} "
            f"(baseline={violation.baseline_value:.3f}, "
            f"recent={violation.recent_value:.3f}, "
            f"max_allowed_delta={violation.max_allowed:.3f}) "
            f"in category '{violation.category}'"
        )
        self._db.insert_alert(AlertRecord(
            alert_id=alert_id,
            agent_id=self._config.agent_id,
            timestamp=datetime.now(timezone.utc),
            rule_name=violation.rule_name,
            severity=violation.severity,
            message=msg,
            metrics={
                "baseline": violation.baseline_value,
                "recent": violation.recent_value,
                "delta": violation.delta,
                "category": violation.category,
                "metric": violation.metric,
            },
        ))
        is_rollback = violation.severity == AlertSeverity.ROLLBACK.value
        if is_rollback:
            self._db.insert_rollback(RollbackRecord(
                rollback_id=str(uuid.uuid4()),
                agent_id=self._config.agent_id,
                timestamp=datetime.now(timezone.utc),
                trigger_rule=violation.rule_name,
                reason=f"Metric threshold breached: {violation.metric} delta={violation.delta:.3f}",
                metrics={
                    "baseline": violation.baseline_value,
                    "recent": violation.recent_value,
                    "delta": violation.delta,
                },
            ))
        return AlertFired(
            alert_id=alert_id, rule_name=violation.rule_name,
            severity=violation.severity, message=msg,
            triggered_rollback=is_rollback,
        )

    def _fire_rollback(
        self, condition: RollbackCondition, ctx: dict
    ) -> AlertFired:
        alert_id = str(uuid.uuid4())
        msg = (
            f"[ROLLBACK] '{condition.name}' triggered — {condition.description}. "
            f"expression: {condition.expression!r}"
        )
        self._db.insert_alert(AlertRecord(
            alert_id=alert_id,
            agent_id=self._config.agent_id,
            timestamp=datetime.now(timezone.utc),
            rule_name=condition.name,
            severity=condition.action.value,
            message=msg,
            metrics=ctx,
        ))
        self._db.insert_rollback(RollbackRecord(
            rollback_id=str(uuid.uuid4()),
            agent_id=self._config.agent_id,
            timestamp=datetime.now(timezone.utc),
            trigger_rule=condition.name,
            reason=condition.description,
            metrics=ctx,
        ))
        return AlertFired(
            alert_id=alert_id,
            rule_name=condition.name,
            severity=condition.action.value,
            message=msg,
            triggered_rollback=True,
        )

    def _build_expression_context(self, report: DriftReport) -> dict:
        """
        Build the safe variable namespace for rollback condition expressions.

        Variables exposed:
          overall_drift_score
          high_risk_count                — total high-risk events across all categories
          high_risk_accuracy             — weighted average accuracy on high-risk events
          {category}_count               — total recent events in that category
          {category}_resolution_rate
          {category}_escalation_rate
          {category}_denial_rate
          {category}_high_risk_count
          {category}_high_risk_escalation_rate
          {category}_accuracy            — only if ground-truth data available
        """
        ctx: dict = {"overall_drift_score": report.overall_drift_score}

        for cat, m in report.recent_metrics.items():
            pfx = cat.replace("-", "_").replace(" ", "_")
            ctx[f"{pfx}_count"] = m.total_events
            ctx[f"{pfx}_resolution_rate"] = m.resolution_rate
            ctx[f"{pfx}_escalation_rate"] = m.escalation_rate
            ctx[f"{pfx}_denial_rate"] = m.denial_rate
            ctx[f"{pfx}_high_risk_count"] = m.high_risk_count
            ctx[f"{pfx}_high_risk_escalation_rate"] = m.high_risk_escalation_rate
            if m.accuracy is not None:
                ctx[f"{pfx}_accuracy"] = m.accuracy
            if m.high_risk_accuracy is not None:
                ctx[f"{pfx}_high_risk_accuracy"] = m.high_risk_accuracy

        ctx["high_risk_count"] = sum(
            m.high_risk_count for m in report.recent_metrics.values()
        )
        hr_accuracies = [
            m.high_risk_accuracy
            for m in report.recent_metrics.values()
            if m.high_risk_accuracy is not None
        ]
        ctx["high_risk_accuracy"] = (
            sum(hr_accuracies) / len(hr_accuracies) if hr_accuracies else 1.0
        )
        return ctx

    def _cooldown_ok(self, condition: RollbackCondition) -> bool:
        last = self._db.get_last_alert_time(self._config.agent_id, condition.name)
        if last is None:
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= condition.cooldown_seconds
