"""PM-facing dashboard: side-by-side normal metrics vs governance signals.

Renders to terminal via `rich`. Also provides raw dict output for API use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .alerts import AlertEngine
from .config import GovernanceConfig
from .drift import CategoryMetrics, DriftDetector, DriftReport
from .storage import GovernanceDB


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class NormalMetrics:
    """Standard observability metrics — what most dashboards already show."""

    total_decisions: int
    resolution_rate: float
    escalation_rate: float
    denial_rate: float
    avg_response_time_ms: float
    decisions_last_hour: int
    decisions_last_24h: int


@dataclass
class GovernanceSignals:
    """What the governance layer adds that normal metrics can't see."""

    overall_drift_score: float
    active_violations: int
    rollback_events: int
    high_risk_decisions_recent: int
    high_risk_escalation_rate: float
    high_risk_accuracy: Optional[float]
    categories_in_drift: list[str]
    is_safe: bool
    safety_reason: str


@dataclass
class CategoryRow:
    category: str
    total_recent: int
    resolution_rate: float
    escalation_rate: float
    high_risk_count: int
    drift_score: float
    violations: int
    severity: str  # "ok" | "warn" | "critical" | "rollback"
    agent_risk_calibration: Optional[float] = None
    avg_confidence: Optional[float] = None


@dataclass
class DashboardSnapshot:
    agent_id: str
    snapshot_time: datetime
    normal_metrics: NormalMetrics
    governance_signals: GovernanceSignals
    category_breakdown: list[CategoryRow]
    recent_alerts: list[dict]
    drift_report: DriftReport

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "snapshot_time": self.snapshot_time.isoformat(),
            "is_safe": self.governance_signals.is_safe,
            "safety_reason": self.governance_signals.safety_reason,
            "normal_metrics": {
                "total_decisions": self.normal_metrics.total_decisions,
                "resolution_rate": round(self.normal_metrics.resolution_rate, 4),
                "escalation_rate": round(self.normal_metrics.escalation_rate, 4),
                "denial_rate": round(self.normal_metrics.denial_rate, 4),
                "avg_response_time_ms": round(self.normal_metrics.avg_response_time_ms, 1),
                "decisions_last_hour": self.normal_metrics.decisions_last_hour,
                "decisions_last_24h": self.normal_metrics.decisions_last_24h,
            },
            "governance_signals": {
                "overall_drift_score": round(self.governance_signals.overall_drift_score, 4),
                "active_violations": self.governance_signals.active_violations,
                "rollback_events": self.governance_signals.rollback_events,
                "high_risk_decisions_recent": self.governance_signals.high_risk_decisions_recent,
                "high_risk_escalation_rate": round(
                    self.governance_signals.high_risk_escalation_rate, 4
                ),
                "high_risk_accuracy": (
                    round(self.governance_signals.high_risk_accuracy, 4)
                    if self.governance_signals.high_risk_accuracy is not None else None
                ),
                "categories_in_drift": self.governance_signals.categories_in_drift,
            },
            "category_breakdown": [
                {
                    "category": r.category,
                    "total_recent": r.total_recent,
                    "resolution_rate": round(r.resolution_rate, 4),
                    "escalation_rate": round(r.escalation_rate, 4),
                    "high_risk_count": r.high_risk_count,
                    "drift_score": round(r.drift_score, 4),
                    "violations": r.violations,
                    "severity": r.severity,
                    "agent_risk_calibration": (
                        round(r.agent_risk_calibration, 4)
                        if r.agent_risk_calibration is not None else None
                    ),
                    "avg_confidence": (
                        round(r.avg_confidence, 4)
                        if r.avg_confidence is not None else None
                    ),
                }
                for r in self.category_breakdown
            ],
            "recent_alerts": self.recent_alerts,
        }


# ── builder ───────────────────────────────────────────────────────────────────

class DashboardBuilder:
    """Assembles a DashboardSnapshot from live governance data."""

    def __init__(
        self,
        config: GovernanceConfig,
        db: GovernanceDB,
        detector: DriftDetector,
        alert_engine: AlertEngine,
    ) -> None:
        self._config = config
        self._db = db
        self._detector = detector
        self._engine = alert_engine

    def build(self) -> DashboardSnapshot:
        drift = self._detector.detect()
        is_safe, safety_reason = self._engine.is_agent_safe()

        recent = self._db.get_recent_decisions(
            self._config.agent_id, limit=self._config.recent_window_size
        )
        rollbacks = self._db.get_rollbacks(self._config.agent_id, limit=10)
        alerts = self._db.get_recent_alerts(self._config.agent_id, limit=20)

        return DashboardSnapshot(
            agent_id=self._config.agent_id,
            snapshot_time=datetime.now(timezone.utc),
            normal_metrics=self._normal(drift, recent),
            governance_signals=self._governance(drift, rollbacks, is_safe, safety_reason),
            category_breakdown=self._categories(drift),
            recent_alerts=[
                {
                    "time": a.timestamp.strftime("%H:%M:%S"),
                    "rule": a.rule_name,
                    "severity": a.severity,
                    "message": a.message[:120],
                }
                for a in alerts[:10]
            ],
            drift_report=drift,
        )

    def _normal(self, drift: DriftReport, recent) -> NormalMetrics:
        total = self._db.count_decisions(self._config.agent_id)
        if not recent:
            return NormalMetrics(
                total_decisions=total,
                resolution_rate=0.0, escalation_rate=0.0, denial_rate=0.0,
                avg_response_time_ms=0.0, decisions_last_hour=0, decisions_last_24h=0,
            )
        n = len(recent)
        now = datetime.now(timezone.utc)
        return NormalMetrics(
            total_decisions=total,
            resolution_rate=sum(1 for r in recent if r.decision == "resolve") / n,
            escalation_rate=sum(1 for r in recent if r.decision == "escalate") / n,
            denial_rate=sum(1 for r in recent if r.decision == "deny") / n,
            avg_response_time_ms=sum(r.resolution_time_ms for r in recent) / n,
            decisions_last_hour=sum(
                1 for r in recent if (now - r.timestamp).total_seconds() < 3600
            ),
            decisions_last_24h=sum(
                1 for r in recent if (now - r.timestamp).total_seconds() < 86400
            ),
        )

    def _governance(self, drift, rollbacks, is_safe, safety_reason) -> GovernanceSignals:
        hr = self._db.get_recent_decisions(
            self._config.agent_id,
            limit=self._config.recent_window_size,
            high_risk_only=True,
        )
        hr_esc = sum(1 for r in hr if r.decision == "escalate")
        hr_gt = [r for r in hr if r.ground_truth is not None]
        hr_acc = (
            sum(1 for r in hr_gt if r.decision == r.ground_truth) / len(hr_gt)
            if len(hr_gt) >= 5 else None
        )
        categories_drifting = [
            c for c, s in drift.category_drift_scores.items() if s > 0.3
        ]
        return GovernanceSignals(
            overall_drift_score=drift.overall_drift_score,
            active_violations=len(drift.violations),
            rollback_events=len(rollbacks),
            high_risk_decisions_recent=len(hr),
            high_risk_escalation_rate=hr_esc / max(len(hr), 1),
            high_risk_accuracy=hr_acc,
            categories_in_drift=categories_drifting,
            is_safe=is_safe,
            safety_reason=safety_reason,
        )

    def _categories(self, drift: DriftReport) -> list[CategoryRow]:
        viols_by_cat: dict[str, list] = {}
        for v in drift.violations:
            viols_by_cat.setdefault(v.category, []).append(v)

        rows: list[CategoryRow] = []
        for cat, m in drift.recent_metrics.items():
            viols = viols_by_cat.get(cat, [])
            severity = "ok"
            for v in viols:
                if v.severity == "rollback":
                    severity = "rollback"
                    break
                if v.severity == "critical" and severity not in ("rollback",):
                    severity = "critical"
                elif v.severity == "warn" and severity == "ok":
                    severity = "warn"
            rows.append(CategoryRow(
                category=cat,
                total_recent=m.total_events,
                resolution_rate=m.resolution_rate,
                escalation_rate=m.escalation_rate,
                high_risk_count=m.high_risk_count,
                drift_score=drift.category_drift_scores.get(cat, 0.0),
                violations=len(viols),
                severity=severity,
                agent_risk_calibration=m.agent_risk_calibration,
                avg_confidence=m.avg_confidence,
            ))
        return sorted(rows, key=lambda r: r.drift_score, reverse=True)


# ── terminal renderer ─────────────────────────────────────────────────────────

_SEVERITY_STYLE = {
    "ok": "green",
    "warn": "yellow",
    "critical": "red",
    "rollback": "bold red",
    "info": "cyan",
}


class TerminalDashboard:
    """Renders a DashboardSnapshot to the terminal using rich."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self._console = console or Console()

    def render(self, snap: DashboardSnapshot) -> None:
        c = self._console
        c.rule(f"[bold]AI Governance Dashboard — {snap.agent_id}[/bold]")
        c.print(f"[dim]Snapshot: {snap.snapshot_time.strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]")
        c.print()

        self._render_safety_banner(snap)
        c.print()
        self._render_side_by_side(snap)
        c.print()
        self._render_category_table(snap)
        c.print()
        self._render_alerts(snap)

    def _render_safety_banner(self, snap: DashboardSnapshot) -> None:
        sig = snap.governance_signals
        if sig.is_safe:
            style = "bold green"
            icon = "✓"
            label = "AGENT SAFE TO RUN"
        else:
            style = "bold red"
            icon = "✗"
            label = "AGENT HALTED — ROLLBACK ACTIVE"
        self._console.print(
            Panel(
                f"[{style}]{icon}  {label}[/{style}]\n"
                f"[dim]{sig.safety_reason}[/dim]",
                border_style=style,
            )
        )

    def _render_side_by_side(self, snap: DashboardSnapshot) -> None:
        c = self._console
        nm = snap.normal_metrics
        gs = snap.governance_signals

        normal_table = Table(
            title="Normal Metrics",
            box=box.SIMPLE_HEAVY,
            title_style="bold cyan",
        )
        normal_table.add_column("Metric", style="dim")
        normal_table.add_column("Value", justify="right")

        normal_table.add_row("Total decisions", str(nm.total_decisions))
        normal_table.add_row("Resolution rate", f"{nm.resolution_rate:.1%}")
        normal_table.add_row("Escalation rate", f"{nm.escalation_rate:.1%}")
        normal_table.add_row("Denial rate", f"{nm.denial_rate:.1%}")
        normal_table.add_row("Avg response time", f"{nm.avg_response_time_ms:.0f} ms")
        normal_table.add_row("Last hour", str(nm.decisions_last_hour))
        normal_table.add_row("Last 24h", str(nm.decisions_last_24h))

        gov_table = Table(
            title="Governance Signals",
            box=box.SIMPLE_HEAVY,
            title_style="bold magenta",
        )
        gov_table.add_column("Signal", style="dim")
        gov_table.add_column("Value", justify="right")

        drift_color = (
            "green" if gs.overall_drift_score < 0.3
            else "yellow" if gs.overall_drift_score < 0.6
            else "red"
        )
        gov_table.add_row(
            "Overall drift score",
            f"[{drift_color}]{gs.overall_drift_score:.3f}[/{drift_color}]",
        )
        gov_table.add_row(
            "Active violations",
            f"[{'red' if gs.active_violations else 'green'}]{gs.active_violations}[/]",
        )
        gov_table.add_row(
            "Rollback events",
            f"[{'red' if gs.rollback_events else 'green'}]{gs.rollback_events}[/]",
        )
        gov_table.add_row("High-risk events (recent)", str(gs.high_risk_decisions_recent))
        gov_table.add_row(
            "High-risk escalation rate", f"{gs.high_risk_escalation_rate:.1%}"
        )
        gov_table.add_row(
            "High-risk accuracy",
            f"{gs.high_risk_accuracy:.1%}" if gs.high_risk_accuracy is not None else "n/a",
        )
        cats_drifting = ", ".join(gs.categories_in_drift) or "none"
        gov_table.add_row("Categories in drift", cats_drifting)

        from rich.columns import Columns
        c.print(Columns([normal_table, gov_table]))

    def _render_category_table(self, snap: DashboardSnapshot) -> None:
        table = Table(
            title="Category Breakdown",
            box=box.ROUNDED,
            title_style="bold white",
        )
        table.add_column("Category", style="white")
        table.add_column("Events", justify="right")
        table.add_column("Resolve%", justify="right")
        table.add_column("Escalate%", justify="right")
        table.add_column("High-Risk", justify="right")
        table.add_column("Drift", justify="right")
        table.add_column("Violations", justify="right")
        table.add_column("Status", justify="center")

        for row in snap.category_breakdown:
            style = _SEVERITY_STYLE.get(row.severity, "white")
            drift_pct = f"{row.drift_score:.0%}"
            table.add_row(
                row.category,
                str(row.total_recent),
                f"{row.resolution_rate:.1%}",
                f"{row.escalation_rate:.1%}",
                str(row.high_risk_count),
                f"[{style}]{drift_pct}[/{style}]",
                f"[{style}]{row.violations}[/{style}]" if row.violations else "0",
                f"[{style}]{row.severity.upper()}[/{style}]",
            )

        self._console.print(table)

    def _render_alerts(self, snap: DashboardSnapshot) -> None:
        if not snap.recent_alerts:
            self._console.print("[dim]No recent alerts.[/dim]")
            return

        table = Table(
            title="Recent Alerts",
            box=box.SIMPLE,
            title_style="bold yellow",
        )
        table.add_column("Time")
        table.add_column("Rule")
        table.add_column("Severity")
        table.add_column("Message")

        for a in snap.recent_alerts:
            sev = a["severity"]
            style = _SEVERITY_STYLE.get(sev, "white")
            table.add_row(
                a["time"],
                a["rule"],
                f"[{style}]{sev.upper()}[/{style}]",
                a["message"][:100],
            )
        self._console.print(table)
