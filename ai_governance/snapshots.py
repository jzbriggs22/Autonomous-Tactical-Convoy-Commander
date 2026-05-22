"""Governance state snapshots: save, load, and diff.

Captures the full governance state at a point in time for comparison.
Answers the PM question: "what changed between yesterday and today?"
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import GovernanceConfig
from .dashboard import DashboardBuilder, DashboardSnapshot
from .storage import GovernanceDB


@dataclass
class StateSnapshot:
    agent_id: str
    timestamp: str
    config_version: str
    config_fingerprint: str
    is_safe: bool
    safety_reason: str
    overall_drift_score: float
    total_decisions: int
    active_violations: int
    rollback_events: int
    category_metrics: dict[str, dict]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "StateSnapshot":
        data = json.loads(text)
        return cls(**data)

    @classmethod
    def from_dashboard(cls, snap: DashboardSnapshot, config: GovernanceConfig) -> "StateSnapshot":
        cats = {}
        for row in snap.category_breakdown:
            cats[row.category] = {
                "total_recent": row.total_recent,
                "resolution_rate": round(row.resolution_rate, 4),
                "escalation_rate": round(row.escalation_rate, 4),
                "high_risk_count": row.high_risk_count,
                "drift_score": round(row.drift_score, 4),
                "violations": row.violations,
                "severity": row.severity,
            }
        return cls(
            agent_id=snap.agent_id,
            timestamp=snap.snapshot_time.isoformat(),
            config_version=config.version,
            config_fingerprint=config.fingerprint,
            is_safe=snap.governance_signals.is_safe,
            safety_reason=snap.governance_signals.safety_reason,
            overall_drift_score=round(snap.governance_signals.overall_drift_score, 4),
            total_decisions=snap.normal_metrics.total_decisions,
            active_violations=snap.governance_signals.active_violations,
            rollback_events=snap.governance_signals.rollback_events,
            category_metrics=cats,
        )


@dataclass
class DiffItem:
    field: str
    category: Optional[str]
    old_value: object
    new_value: object
    severity: str  # "info", "warn", "critical"

    @property
    def delta(self) -> Optional[float]:
        if isinstance(self.old_value, (int, float)) and isinstance(self.new_value, (int, float)):
            return self.new_value - self.old_value
        return None


@dataclass
class SnapshotDiff:
    old_timestamp: str
    new_timestamp: str
    items: list[DiffItem]
    safety_changed: bool
    config_changed: bool

    @property
    def has_regressions(self) -> bool:
        return any(d.severity in ("warn", "critical") for d in self.items)

    def summary(self) -> str:
        lines = [f"Diff: {self.old_timestamp} -> {self.new_timestamp}"]
        if self.config_changed:
            lines.append("  [!] Config changed between snapshots")
        if self.safety_changed:
            lines.append("  [!] Safety status changed")
        for d in self.items:
            tag = {"info": " ", "warn": "!", "critical": "X"}[d.severity]
            delta_str = f" ({d.delta:+.4f})" if d.delta is not None else ""
            cat_str = f" [{d.category}]" if d.category else ""
            lines.append(
                f"  [{tag}]{cat_str} {d.field}: {d.old_value} -> {d.new_value}{delta_str}"
            )
        return "\n".join(lines)


def diff_snapshots(old: StateSnapshot, new: StateSnapshot) -> SnapshotDiff:
    items: list[DiffItem] = []
    safety_changed = old.is_safe != new.is_safe
    config_changed = old.config_fingerprint != new.config_fingerprint

    if safety_changed:
        sev = "critical" if old.is_safe and not new.is_safe else "info"
        items.append(DiffItem("is_safe", None, old.is_safe, new.is_safe, sev))

    _diff_scalar(items, "overall_drift_score", None,
                 old.overall_drift_score, new.overall_drift_score,
                 warn_threshold=0.1, critical_threshold=0.3)
    _diff_scalar(items, "active_violations", None,
                 old.active_violations, new.active_violations,
                 warn_threshold=1, critical_threshold=3)
    _diff_scalar(items, "rollback_events", None,
                 old.rollback_events, new.rollback_events,
                 warn_threshold=1, critical_threshold=2)
    _diff_scalar(items, "total_decisions", None,
                 old.total_decisions, new.total_decisions,
                 warn_threshold=float("inf"), critical_threshold=float("inf"))

    all_cats = set(old.category_metrics) | set(new.category_metrics)
    for cat in sorted(all_cats):
        old_cat = old.category_metrics.get(cat, {})
        new_cat = new.category_metrics.get(cat, {})
        if cat not in old.category_metrics:
            items.append(DiffItem("category_added", cat, None, cat, "info"))
            continue
        if cat not in new.category_metrics:
            items.append(DiffItem("category_removed", cat, cat, None, "warn"))
            continue

        _diff_scalar(items, "drift_score", cat,
                     old_cat.get("drift_score", 0), new_cat.get("drift_score", 0),
                     warn_threshold=0.1, critical_threshold=0.3)
        _diff_scalar(items, "resolution_rate", cat,
                     old_cat.get("resolution_rate", 0), new_cat.get("resolution_rate", 0),
                     warn_threshold=0.1, critical_threshold=0.2)
        _diff_scalar(items, "violations", cat,
                     old_cat.get("violations", 0), new_cat.get("violations", 0),
                     warn_threshold=1, critical_threshold=3)

        old_sev = old_cat.get("severity", "ok")
        new_sev = new_cat.get("severity", "ok")
        if old_sev != new_sev:
            sev_order = {"ok": 0, "warn": 1, "critical": 2, "rollback": 3}
            is_worse = sev_order.get(new_sev, 0) > sev_order.get(old_sev, 0)
            items.append(DiffItem(
                "severity", cat, old_sev, new_sev,
                "critical" if is_worse else "info",
            ))

    return SnapshotDiff(
        old_timestamp=old.timestamp,
        new_timestamp=new.timestamp,
        items=items,
        safety_changed=safety_changed,
        config_changed=config_changed,
    )


def _diff_scalar(
    items: list[DiffItem], field: str, category: Optional[str],
    old_val, new_val, *, warn_threshold: float, critical_threshold: float,
) -> None:
    if old_val == new_val:
        return
    delta = abs(new_val - old_val) if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)) else 0
    if delta >= critical_threshold:
        severity = "critical"
    elif delta >= warn_threshold:
        severity = "warn"
    else:
        severity = "info"
    items.append(DiffItem(field, category, old_val, new_val, severity))
