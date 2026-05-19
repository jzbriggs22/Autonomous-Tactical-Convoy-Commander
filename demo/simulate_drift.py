#!/usr/bin/env python3
"""
AI Agent Governance MVP — drift simulation demo.

Simulates a customer-service agent that looks healthy on standard metrics
for weeks, then silently drifts on high-risk cases (fraud_claim).

Phases:
  1. WARMUP  (100 events): healthy behavior → compute baseline
  2. HEALTHY (50 events):  no drift → dashboard shows green
  3. DRIFT   (80 events):  fraud escalation rate drops from 85% → 20%
                           normal metrics (resolution rate) actually *improve*
  4. ALERT               : drift detector fires critical + rollback alerts
  5. ROLLBACK            : safety gate halts agent
"""

from __future__ import annotations

import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.rule import Rule

from ai_governance.alerts import AlertEngine
from ai_governance.config import GovernanceConfig
from ai_governance.dashboard import DashboardBuilder, TerminalDashboard
from ai_governance.drift import DriftDetector
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import GovernanceDB

console = Console()
rng = random.Random(42)

# ── decision generators ───────────────────────────────────────────────────────

CATEGORIES = ["billing_dispute", "fraud_claim", "policy_question", "returns", "shipping"]
CATEGORY_WEIGHTS = [0.25, 0.20, 0.15, 0.25, 0.15]


def _healthy_decision(category: str) -> str:
    """Simulate a well-calibrated agent: escalates fraud, resolves billing."""
    if category == "fraud_claim":
        return rng.choices(["escalate", "resolve", "defer"], weights=[85, 10, 5])[0]
    if category == "billing_dispute":
        return rng.choices(["resolve", "escalate", "deny", "partial_resolve"], weights=[65, 15, 10, 10])[0]
    if category == "policy_question":
        return rng.choices(["resolve", "escalate", "defer"], weights=[60, 25, 15])[0]
    return rng.choices(["resolve", "deny", "defer"], weights=[75, 15, 10])[0]


def _drifted_decision(category: str) -> str:
    """Simulate an agent that has drifted: stops escalating fraud."""
    if category == "fraud_claim":
        # Escalation drops from 85% to ~20% — the dangerous drift
        return rng.choices(["escalate", "resolve", "deny"], weights=[20, 70, 10])[0]
    if category == "billing_dispute":
        # Resolution rate slightly improves (looks healthy on normal metrics!)
        return rng.choices(["resolve", "escalate", "deny", "partial_resolve"], weights=[72, 10, 8, 10])[0]
    return _healthy_decision(category)


def _make_event(
    category: str,
    decision: str,
    ts: datetime,
    ground_truth: str = None,
    metadata: dict = None,
) -> IngestRequest:
    return IngestRequest(
        case_id=str(uuid.uuid4()),
        case_category=category,
        decision=decision,
        resolution_time_ms=rng.randint(200, 2000),
        ground_truth=ground_truth,
        timestamp=ts,
        metadata=metadata or {},
    )


# ── simulation ────────────────────────────────────────────────────────────────

def simulate(
    n_warmup: int = 120,
    n_healthy: int = 60,
    n_drift: int = 80,
) -> None:
    config = GovernanceConfig.default_customer_service()
    # Tune for demo: smaller windows so we need fewer events to trigger
    config.min_baseline_events = 15
    config.recent_window_size = 50
    for thr in config.drift_thresholds:
        thr.min_baseline_samples = 15
        thr.recent_window = 20
    # Lower rollback threshold for demo visibility (real deployment: 0.60)
    for cond in config.rollback_conditions:
        cond.cooldown_seconds = 60
        if cond.name == "fraud_underescalation":
            cond.__dict__["expression"] = (
                "fraud_claim_escalation_rate < 0.75 and fraud_claim_count >= 10"
            )

    db = GovernanceDB(":memory:")
    ingestion = IngestionLayer(config, db)
    detector = DriftDetector(config, db)
    engine = AlertEngine(config, db)
    dashboard_builder = DashboardBuilder(config, db, detector, engine)
    renderer = TerminalDashboard(console)

    now = datetime.now(timezone.utc)

    # ── Phase 1: Warmup ───────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]PHASE 1 — Warmup (establishing baseline)[/bold cyan]"))
    console.print(f"[dim]Injecting {n_warmup} events over simulated 48h window...[/dim]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Warmup events", total=n_warmup)
        for i in range(n_warmup):
            cat = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS)[0]
            decision = _healthy_decision(cat)
            gt = decision if rng.random() < 0.6 else None  # 60% have ground truth
            ts = now - timedelta(hours=48 - (i * 48 / n_warmup))
            ingestion.ingest(_make_event(cat, decision, ts, ground_truth=gt))
            progress.advance(task)

    baseline_results = detector.compute_baseline()
    console.print(
        f"[green]✓ Baseline frozen for {len(baseline_results)} categories: "
        f"{', '.join(baseline_results.keys())}[/green]"
    )
    for cat, m in baseline_results.items():
        if cat == "fraud_claim":
            console.print(
                f"  [dim]fraud_claim baseline: "
                f"escalation_rate={m.escalation_rate:.1%}, "
                f"resolution_rate={m.resolution_rate:.1%}[/dim]"
            )

    # ── Phase 2: Healthy operation ────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold green]PHASE 2 — Healthy operation (no drift)[/bold green]"))

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Healthy events", total=n_healthy)
        for i in range(n_healthy):
            cat = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS)[0]
            decision = _healthy_decision(cat)
            gt = decision if rng.random() < 0.5 else None
            ts = now - timedelta(hours=24 - (i * 24 / n_healthy))
            ingestion.ingest(_make_event(cat, decision, ts, ground_truth=gt))
            progress.advance(task)

    console.print("[dim]Running drift detection on healthy window...[/dim]")
    report = detector.detect()
    fired = engine.evaluate(report)
    console.print(
        f"[green]✓ Drift score: {report.overall_drift_score:.3f} | "
        f"Violations: {len(report.violations)} | Alerts fired: {len(fired)}[/green]"
    )

    console.print()
    console.print("[bold]DASHBOARD — Healthy state:[/bold]")
    renderer.render(dashboard_builder.build())

    # ── Phase 3: Silent drift ─────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold yellow]PHASE 3 — Silent drift (agent drifts on fraud cases)[/bold yellow]"))
    console.print(
        "[yellow]Note: standard resolution rate will IMPROVE while fraud safety degrades.[/yellow]"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Drifted events", total=n_drift)
        for i in range(n_drift):
            cat = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS)[0]
            decision = _drifted_decision(cat)
            gt = decision if rng.random() < 0.5 else None
            # Recent: all events in the last hour
            ts = now - timedelta(minutes=60 - (i * 60 / n_drift))
            ingestion.ingest(_make_event(cat, decision, ts, ground_truth=gt))
            progress.advance(task)

    # ── Phase 4: Detection ────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold red]PHASE 4 — Drift detection + alert firing[/bold red]"))

    report = detector.detect()
    fired = engine.evaluate(report)

    console.print(f"[bold]Overall drift score: [red]{report.overall_drift_score:.3f}[/red][/bold]")
    console.print(f"[bold]Violations detected: [red]{len(report.violations)}[/red][/bold]")
    console.print(f"[bold]Alerts fired:        [red]{len(fired)}[/red][/bold]")
    console.print()

    if report.violations:
        console.print("[bold]Violated thresholds:[/bold]")
        for v in report.violations:
            direction = "▲" if v.delta > 0 else "▼"
            console.print(
                f"  [red]•[/red] [{v.severity.upper()}] [bold]{v.rule_name}[/bold]  "
                f"category=[cyan]{v.category}[/cyan]  "
                f"metric=[cyan]{v.metric}[/cyan]  "
                f"baseline=[dim]{v.baseline_value:.3f}[/dim] → "
                f"recent=[red]{v.recent_value:.3f}[/red]  "
                f"delta={direction}{abs(v.delta):.3f} (max={v.max_allowed:.3f})"
            )

    if fired:
        console.print()
        console.print("[bold]Alerts fired:[/bold]")
        for f in fired:
            icon = "🔴" if f.triggered_rollback else "⚠️"
            console.print(f"  {icon}  {f.message[:120]}")

    # ── Phase 5: PM dashboard showing the divergence ──────────────────────────
    console.print()
    console.print(Rule("[bold]PHASE 5 — PM Dashboard: normal vs governance[/bold]"))
    console.print(
        Panel(
            "[bold yellow]KEY INSIGHT[/bold yellow]\n\n"
            "Normal metrics show resolution rate improving (✓).\n"
            "Governance signals reveal fraud escalation rate has collapsed (✗).\n"
            "The agent looks healthy to standard monitoring — but is creating "
            "financial risk on every fraud case it quietly resolves.",
            border_style="yellow",
        )
    )
    renderer.render(dashboard_builder.build())

    # Summary
    is_safe, reason = engine.is_agent_safe()
    console.print()
    if not is_safe:
        console.print(
            Panel(
                f"[bold red]AGENT HALTED[/bold red]\n\n{reason}\n\n"
                "[dim]The governance system has blocked further agent operation.\n"
                "A human reviewer must approve re-enabling the agent.[/dim]",
                border_style="red",
                title="ROLLBACK ACTIVE",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]Agent is safe to run.[/bold green]\n\n"
                "[dim]No rollback conditions were triggered. Review alerts above.[/dim]",
                border_style="green",
            )
        )

    # Metrics divergence summary
    console.print()
    console.print(Rule("[dim]Metrics divergence summary[/dim]"))
    console.print(
        "[dim]This is what makes silent drift dangerous — "
        "standard metrics give false confidence:[/dim]"
    )
    snap = dashboard_builder.build()
    nm = snap.normal_metrics
    gs = snap.governance_signals
    fraud_recent = report.recent_metrics.get("fraud_claim")
    fraud_baseline = db.get_baseline(config.agent_id, "fraud_claim", "escalation_rate")

    console.print(f"\n  STANDARD MONITORING")
    console.print(f"    Overall resolution rate:    {nm.resolution_rate:.1%}  ← [green]looks fine or improved[/green]")
    console.print(f"    Overall escalation rate:    {nm.escalation_rate:.1%}")
    console.print(f"    Avg response time:          {nm.avg_response_time_ms:.0f} ms")
    console.print()
    console.print(f"  GOVERNANCE LAYER")
    if fraud_baseline:
        console.print(
            f"    fraud_claim escalation (baseline): {fraud_baseline[0]:.1%}  ← [green]was good[/green]"
        )
    if fraud_recent:
        console.print(
            f"    fraud_claim escalation (recent):   {fraud_recent.escalation_rate:.1%}  ← [red]has collapsed[/red]"
        )
    console.print(f"    Overall drift score:        {gs.overall_drift_score:.3f}")
    console.print(f"    Active violations:          {gs.active_violations}")
    console.print(f"    Rollback events:            {gs.rollback_events}")
    if gs.high_risk_decisions_recent:
        console.print(
            f"    High-risk escalation rate:  {gs.high_risk_escalation_rate:.1%}  "
            f"← [{'red' if gs.high_risk_escalation_rate < 0.6 else 'green'}]"
            f"{'DANGER' if gs.high_risk_escalation_rate < 0.6 else 'OK'}[/]"
        )
    console.print()


if __name__ == "__main__":
    simulate()
