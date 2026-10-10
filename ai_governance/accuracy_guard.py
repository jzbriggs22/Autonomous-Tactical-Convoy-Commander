"""Accuracy-based governance guard.

Closes the feedback loop: ground-truth labels → accuracy metrics →
governance alerts → rollback. An agent whose labeled accuracy drops
below thresholds is unsafe even if its drift metrics look stable —
drift detection sees behavioral change, the accuracy guard sees
being *wrong*.

Thresholds are only enforced once enough labels exist (min_labels),
so a sparse labeling effort never fires spurious rollbacks.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import GovernanceConfig
from .feedback import AccuracyReport, FeedbackPipeline
from .storage import AlertRecord, GovernanceDB, RollbackRecord

logger = logging.getLogger(__name__)


@dataclass
class AccuracyThresholds:
    """Minimum acceptable accuracy levels. Breaches fire governance alerts."""
    min_overall_accuracy: float = 0.85
    min_high_risk_accuracy: float = 0.90
    min_labels: int = 20
    min_high_risk_labels: int = 5
    rollback_on_high_risk_breach: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_overall_accuracy <= 1.0:
            raise ValueError("min_overall_accuracy must be in [0, 1]")
        if not 0.0 <= self.min_high_risk_accuracy <= 1.0:
            raise ValueError("min_high_risk_accuracy must be in [0, 1]")
        if self.min_labels < 1:
            raise ValueError("min_labels must be >= 1")
        if self.min_high_risk_labels < 1:
            raise ValueError("min_high_risk_labels must be >= 1")


@dataclass
class AccuracyViolation:
    kind: str  # "overall_accuracy" | "high_risk_accuracy"
    category: Optional[str]  # None for agent-wide violations
    observed: float
    threshold: float
    labels: int
    message: str


@dataclass
class GuardResult:
    agent_id: str
    checked_at: str
    enforced: bool  # False when below min_labels — nothing evaluated
    total_labeled: int
    violations: list[AccuracyViolation] = field(default_factory=list)
    alerts_fired: list[str] = field(default_factory=list)
    rollback_triggered: bool = False

    @property
    def passed(self) -> bool:
        return self.enforced and not self.violations

    def summary(self) -> str:
        if not self.enforced:
            return (
                f"Accuracy guard: not enforced — {self.total_labeled} labels "
                f"(need more to evaluate)"
            )
        if not self.violations:
            return f"Accuracy guard: PASS ({self.total_labeled} labels evaluated)"
        lines = [f"Accuracy guard: FAIL — {len(self.violations)} violation(s)"]
        for v in self.violations:
            scope = f" in '{v.category}'" if v.category else " (agent-wide)"
            lines.append(
                f"  {v.kind}{scope}: {v.observed:.1%} < {v.threshold:.1%} "
                f"({v.labels} labels)"
            )
        if self.rollback_triggered:
            lines.append("  ROLLBACK TRIGGERED")
        return "\n".join(lines)


class AccuracyGuard:
    """Evaluates labeled accuracy against thresholds; fires alerts and rollbacks."""

    def __init__(
        self,
        config: GovernanceConfig,
        db: GovernanceDB,
        thresholds: Optional[AccuracyThresholds] = None,
    ) -> None:
        self._config = config
        self._db = db
        self._thresholds = thresholds or AccuracyThresholds()
        self._pipeline = FeedbackPipeline(config, db)

    def check(self, *, dry_run: bool = False) -> GuardResult:
        """Evaluate accuracy thresholds. Fires alerts/rollbacks unless dry_run."""
        now = datetime.now(timezone.utc)
        report = self._pipeline.compute_accuracy()
        t = self._thresholds

        result = GuardResult(
            agent_id=self._config.agent_id,
            checked_at=now.isoformat(),
            enforced=report.total_labeled >= t.min_labels,
            total_labeled=report.total_labeled,
        )
        if not result.enforced:
            return result

        result.violations.extend(self._find_violations(report))

        if not dry_run and result.violations:
            self._enforce(result)

        return result

    # ── private ──────────────────────────────────────────────────────────────

    def _find_violations(self, report: AccuracyReport) -> list[AccuracyViolation]:
        t = self._thresholds
        violations: list[AccuracyViolation] = []

        if (
            report.overall_accuracy is not None
            and report.overall_accuracy < t.min_overall_accuracy
        ):
            violations.append(AccuracyViolation(
                kind="overall_accuracy",
                category=None,
                observed=report.overall_accuracy,
                threshold=t.min_overall_accuracy,
                labels=report.total_labeled,
                message=(
                    f"Agent-wide accuracy {report.overall_accuracy:.1%} is below "
                    f"the {t.min_overall_accuracy:.1%} floor "
                    f"({report.total_labeled} labeled decisions)"
                ),
            ))

        for cat in report.categories:
            if (
                cat.high_risk_accuracy is not None
                and cat.high_risk_labeled >= t.min_high_risk_labels
                and cat.high_risk_accuracy < t.min_high_risk_accuracy
            ):
                violations.append(AccuracyViolation(
                    kind="high_risk_accuracy",
                    category=cat.category,
                    observed=cat.high_risk_accuracy,
                    threshold=t.min_high_risk_accuracy,
                    labels=cat.high_risk_labeled,
                    message=(
                        f"High-risk accuracy in '{cat.category}' is "
                        f"{cat.high_risk_accuracy:.1%}, below the "
                        f"{t.min_high_risk_accuracy:.1%} floor "
                        f"({cat.high_risk_correct}/{cat.high_risk_labeled} correct)"
                    ),
                ))

        return violations

    def _enforce(self, result: GuardResult) -> None:
        t = self._thresholds
        now = datetime.now(timezone.utc)

        for v in result.violations:
            is_high_risk_breach = v.kind == "high_risk_accuracy"
            triggers_rollback = is_high_risk_breach and t.rollback_on_high_risk_breach
            severity = "rollback" if triggers_rollback else "critical"
            rule_name = f"accuracy_guard_{v.kind}"

            alert_id = str(uuid.uuid4())
            self._db.insert_alert(AlertRecord(
                alert_id=alert_id,
                agent_id=self._config.agent_id,
                timestamp=now,
                rule_name=rule_name,
                severity=severity,
                message=f"[{severity.upper()}] {v.message}",
                metrics={
                    "kind": v.kind,
                    "category": v.category,
                    "observed": round(v.observed, 4),
                    "threshold": v.threshold,
                    "labels": v.labels,
                },
            ))
            result.alerts_fired.append(alert_id)

            if triggers_rollback:
                self._db.insert_rollback(RollbackRecord(
                    rollback_id=str(uuid.uuid4()),
                    agent_id=self._config.agent_id,
                    timestamp=now,
                    trigger_rule=rule_name,
                    reason=v.message,
                    metrics={
                        "kind": v.kind,
                        "category": v.category,
                        "observed": round(v.observed, 4),
                        "threshold": v.threshold,
                    },
                ))
                result.rollback_triggered = True
                logger.warning(
                    "Accuracy guard triggered rollback for %s: %s",
                    self._config.agent_id, v.message,
                )
