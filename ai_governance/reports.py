"""Compliance report generator.

Produces structured governance reports covering a time period:
safety status, drift trends, alert history, violation counts,
and audit chain integrity. Output as JSON or human-readable text.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .alerts import AlertEngine
from .audit import AuditLog
from .config import GovernanceConfig
from .drift import DriftDetector
from .storage import GovernanceDB


@dataclass
class ComplianceReport:
    agent_id: str
    generated_at: str
    config_version: str
    config_fingerprint: str
    period_start: Optional[str]
    period_end: str

    is_safe: bool
    safety_reason: str
    overall_drift_score: float

    total_decisions: int
    high_risk_decisions: int
    high_risk_percentage: float

    total_alerts: int
    unacknowledged_alerts: int
    total_rollbacks: int
    active_rollbacks: int

    audit_chain_valid: bool
    audit_chain_broken_at: Optional[int]
    total_audit_entries: int

    categories: list[CategorySummary]
    alert_breakdown: list[AlertSummary]
    violations: list[ViolationSummary]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_text(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  GOVERNANCE COMPLIANCE REPORT",
            f"  Agent: {self.agent_id}",
            f"  Generated: {self.generated_at}",
            f"  Config: v{self.config_version} ({self.config_fingerprint})",
            f"{'=' * 60}",
            "",
            f"  Safety Status: {'SAFE' if self.is_safe else 'UNSAFE'}",
            f"  Reason: {self.safety_reason}",
            f"  Overall Drift Score: {self.overall_drift_score:.4f}",
            "",
            "  --- Decision Summary ---",
            f"  Total decisions: {self.total_decisions}",
            f"  High-risk decisions: {self.high_risk_decisions} ({self.high_risk_percentage:.1%})",
            "",
            "  --- Alert Summary ---",
            f"  Total alerts: {self.total_alerts}",
            f"  Unacknowledged: {self.unacknowledged_alerts}",
            f"  Rollbacks: {self.total_rollbacks} (active: {self.active_rollbacks})",
            "",
            "  --- Audit Chain ---",
            f"  Chain valid: {self.audit_chain_valid}",
            f"  Total entries: {self.total_audit_entries}",
        ]
        if not self.audit_chain_valid:
            lines.append(f"  Chain broken at seq: {self.audit_chain_broken_at}")

        if self.categories:
            lines.append("")
            lines.append("  --- Categories ---")
            for cat in self.categories:
                lines.append(f"  [{cat.category}] {cat.total_events} events, "
                             f"drift={cat.drift_score:.4f}, "
                             f"violations={cat.violation_count}")

        if self.violations:
            lines.append("")
            lines.append("  --- Active Violations ---")
            for v in self.violations:
                lines.append(f"  [{v.severity.upper()}] {v.rule_name}: {v.metric} "
                             f"delta={v.delta:+.4f} in '{v.category}'")

        if self.alert_breakdown:
            lines.append("")
            lines.append("  --- Recent Alerts ---")
            for a in self.alert_breakdown[:10]:
                ack = " [ACK]" if a.acknowledged else ""
                lines.append(f"  [{a.severity.upper()}{ack}] {a.timestamp}: "
                             f"{a.rule_name}")

        lines.append("")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


@dataclass
class CategorySummary:
    category: str
    total_events: int
    high_risk_count: int
    resolution_rate: float
    escalation_rate: float
    drift_score: float
    violation_count: int


@dataclass
class AlertSummary:
    alert_id: str
    timestamp: str
    rule_name: str
    severity: str
    acknowledged: bool


@dataclass
class ViolationSummary:
    category: str
    metric: str
    rule_name: str
    baseline_value: float
    recent_value: float
    delta: float
    severity: str


class ReportGenerator:
    """Generates governance compliance reports from current system state."""

    def __init__(
        self,
        config: GovernanceConfig,
        db: GovernanceDB,
        detector: DriftDetector,
        engine: AlertEngine,
        audit: AuditLog,
    ) -> None:
        self._config = config
        self._db = db
        self._detector = detector
        self._engine = engine
        self._audit = audit

    def generate(self, since: Optional[datetime] = None) -> ComplianceReport:
        now = datetime.now(timezone.utc)
        is_safe, reason = self._engine.is_agent_safe()
        report = self._detector.detect()

        total = self._db.count_decisions(self._config.agent_id)
        high_risk = self._db.count_high_risk_decisions(self._config.agent_id)

        alerts = self._db.get_recent_alerts(self._config.agent_id, limit=100)
        unacked = sum(1 for a in alerts if not a.acknowledged)

        rollbacks = self._db.get_rollbacks(self._config.agent_id, limit=100)
        active_rollbacks = sum(1 for r in rollbacks if not r.resolved)

        valid, broken_seq = self._audit.verify_chain(self._config.agent_id)
        audit_entries = self._audit.get_entries(self._config.agent_id, limit=1)
        total_audit = audit_entries[0].seq if audit_entries else 0

        categories = []
        for cat, metrics in report.recent_metrics.items():
            cat_violations = [v for v in report.violations if v.category == cat]
            categories.append(CategorySummary(
                category=cat,
                total_events=metrics.total_events,
                high_risk_count=metrics.high_risk_count,
                resolution_rate=round(metrics.resolution_rate, 4),
                escalation_rate=round(metrics.escalation_rate, 4),
                drift_score=round(report.category_drift_scores.get(cat, 0.0), 4),
                violation_count=len(cat_violations),
            ))

        alert_breakdown = [
            AlertSummary(
                alert_id=a.alert_id,
                timestamp=a.timestamp.isoformat(),
                rule_name=a.rule_name,
                severity=a.severity,
                acknowledged=a.acknowledged,
            )
            for a in alerts[:20]
        ]

        violations = [
            ViolationSummary(
                category=v.category,
                metric=v.metric,
                rule_name=v.rule_name,
                baseline_value=round(v.baseline_value, 4),
                recent_value=round(v.recent_value, 4),
                delta=round(v.delta, 4),
                severity=v.severity,
            )
            for v in report.violations
        ]

        return ComplianceReport(
            agent_id=self._config.agent_id,
            generated_at=now.isoformat(),
            config_version=self._config.version,
            config_fingerprint=self._config.fingerprint,
            period_start=since.isoformat() if since else None,
            period_end=now.isoformat(),
            is_safe=is_safe,
            safety_reason=reason,
            overall_drift_score=round(report.overall_drift_score, 4),
            total_decisions=total,
            high_risk_decisions=high_risk,
            high_risk_percentage=high_risk / max(total, 1),
            total_alerts=len(alerts),
            unacknowledged_alerts=unacked,
            total_rollbacks=len(rollbacks),
            active_rollbacks=active_rollbacks,
            audit_chain_valid=valid,
            audit_chain_broken_at=broken_seq,
            total_audit_entries=total_audit,
            categories=categories,
            alert_breakdown=alert_breakdown,
            violations=violations,
        )
