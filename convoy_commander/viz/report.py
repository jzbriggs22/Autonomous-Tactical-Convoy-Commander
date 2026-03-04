"""Generate markdown report from simulation results.

The report includes a safety audit section that surfaces all CRITICAL and
WARNING events from the structured event log, plus a summary of model
assumptions and their implications.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from convoy_commander.core.event_log import EventLog, Severity
from convoy_commander.metrics.collector import MetricsCollector, SimMetrics
from convoy_commander.sim.runner import SimResult
from convoy_commander.viz.plots import (
    plot_comms_graph,
    plot_metrics_summary,
    plot_position_errors,
    plot_trajectories,
)


def generate_report(result: SimResult, output_dir: Path) -> Path:
    """Generate full report with plots, metrics, and safety audit.

    Returns path to report.md.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_trajectories(result.world, result.vehicles, plots_dir / "trajectories.png")
    plot_position_errors(result.vehicles, result.config.dt, plots_dir / "position_errors.png")
    plot_metrics_summary(
        result.collector, result.vehicles, result.config.dt, plots_dir / "metrics_summary.png"
    )

    # Final comms graph snapshot
    positions = {v.id: (v.state.x, v.state.y) for v in result.vehicles}
    adj = result.comms.get_adjacency(positions)
    plot_comms_graph(result.vehicles, adj, result.world, plots_dir / "comms_graph.png")

    # Compute metrics
    metrics = result.collector.compute_final(
        result.vehicles,
        result.comms.total_sent,
        result.comms.total_delivered,
        result.comms.total_dropped,
        result.config.duration,
        comms_by_type=result.comms.get_stats_by_type(),
    )

    # Save artifacts
    result.collector.save_metrics(metrics, output_dir / "metrics.json")
    result.collector.save_time_series(output_dir / "time_series.jsonl")
    result.event_log.save(output_dir / "event_log.jsonl")

    # Save reproducibility stamp
    if result.stamp is not None:
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(result.stamp.to_dict(), f, indent=2, default=str)

    # Generate markdown
    report_path = output_dir / "report.md"
    md = _build_markdown(metrics, result, plots_dir)
    report_path.write_text(md)

    return report_path


def _build_markdown(metrics: SimMetrics, result: SimResult, plots_dir: Path) -> str:
    """Build markdown report content."""
    cfg = result.config
    m = metrics
    elog = result.event_log

    lines = [
        "# Convoy Commander Simulation Report",
        "",
        "## Configuration",
        f"- **Scenario:** {cfg.scenario}",
        f"- **Seed:** {cfg.seed}",
        f"- **Vehicles:** {cfg.num_vehicles}",
        f"- **Duration:** {cfg.duration}s",
        f"- **GPS Available:** {cfg.gps_available}",
        f"- **Packet Loss:** {cfg.comms.packet_loss:.0%}",
        f"- **Latency:** {cfg.comms.latency_mean_ms:.0f}ms",
        "",
        "",
    ]

    # Reproducibility stamp
    if result.stamp is not None:
        s = result.stamp
        lines += [
            "## Reproducibility",
            f"- **Git commit:** `{s.git_commit}`{'  (dirty)' if s.git_dirty else ''}",
            f"- **Python:** {s.python_version}",
            f"- **Platform:** {s.platform_info}",
            f"- **Package:** convoy_commander {s.package_version}",
            f"- **Full config:** see `config.json`",
        ]

    lines += [
        "",
        "## Mission Summary",
        f"- **Mission Success:** {'YES' if m.mission_success else 'NO'}",
        f"- **Vehicles Arrived:** {m.vehicles_arrived}/{m.vehicles_total}",
        f"- **Average Time to Destination:** {m.avg_time_to_destination:.1f}s",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Fuel Used | {m.total_fuel_used:.1f} |",
        f"| Avg Fuel Used | {m.avg_fuel_used:.1f} |",
        f"| Convoy Cohesion (avg dist) | {m.convoy_cohesion_score:.1f}m |",
        f"| Near Misses | {m.near_miss_count} |",
        f"| Collisions | {m.collision_count} |",
        f"| Comms Sent | {m.comms_total_sent} |",
        f"| Comms Delivered | {m.comms_total_delivered} |",
        f"| Comms Dropped | {m.comms_total_dropped} |",
        f"| Delivery Ratio | {m.comms_delivery_ratio:.1%} |",
        f"| Avg Position Error | {m.avg_position_error:.2f}m |",
        f"| Max Position Error | {m.max_position_error:.2f}m |",
        f"| Total Distance | {m.total_distance_traveled:.0f}m |",
        f"| Leader Elections | {m.num_leader_elections} |",
        f"| Safe Mode Activations | {m.num_safe_mode_activations} |",
        "",
    ]

    # Per-message-type bandwidth table
    if m.comms_by_type:
        lines += [
            "### Communications Bandwidth by Message Type",
            "",
            "| Message Type | Sent | Delivered | Dropped | Delivery % |",
            "|--------------|------|-----------|---------|------------|",
        ]
        for mtype, stats in sorted(m.comms_by_type.items()):
            s = stats.get("sent", 0)
            d = stats.get("delivered", 0)
            dr = stats.get("dropped", 0)
            ratio = f"{d / s:.0%}" if s > 0 else "N/A"
            lines.append(f"| {mtype} | {s} | {d} | {dr} | {ratio} |")
        lines.append("")

    # --- Safety Audit ---
    lines += _build_safety_audit(elog, cfg)

    # --- Performance notes ---
    lines += _build_performance_notes(result)

    # --- Assumptions ---
    lines += _build_assumptions_section()

    # --- Plots ---
    lines += [
        "## Plots",
        "",
        "### Trajectories (True vs Estimated)",
        "![Trajectories](plots/trajectories.png)",
        "",
        "### Position Estimation Error",
        "![Position Errors](plots/position_errors.png)",
        "",
        "### Speed, Fuel & Uncertainty",
        "![Metrics Summary](plots/metrics_summary.png)",
        "",
        "### Communications Graph (Final State)",
        "![Comms Graph](plots/comms_graph.png)",
        "",
    ]

    return "\n".join(lines)


def _build_performance_notes(result: SimResult) -> list[str]:
    """Document performance characteristics and complexity."""
    cfg = result.config
    n = cfg.num_vehicles
    steps = int(cfg.duration / cfg.dt)

    return [
        "## Performance Notes",
        "",
        f"- **Sim duration:** {cfg.duration}s at dt={cfg.dt}s = {steps:,} steps",
        f"- **Vehicles:** {n}",
        f"- **Per-step complexity:** O(N^2) for collision detection, O(N) for planning/control",
        f"- **Total step-vehicle evaluations:** {steps * n:,}",
        "",
        "### Complexity Drivers",
        "- Collision/near-miss detection: pairwise O(N^2) per step",
        "- Comms broadcast: O(N^2) adjacency check per broadcast interval",
        "- A* route planning: O(E log V) on road graph; called once per vehicle + on replan",
        "- CBBA auction: O(N * S) per re-allocation (every 10s), S = number of slots",
        "",
    ]


def _build_safety_audit(elog: EventLog, cfg: object) -> list[str]:
    """Build the safety audit section from the event log."""
    lines = [
        "## Safety Audit",
        "",
    ]

    # Event count summary
    by_severity = elog.count_by_severity()
    lines.append("### Event Summary by Severity")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    for sev in ["CRITICAL", "WARNING", "INFO", "DEBUG"]:
        count = by_severity.get(sev, 0)
        lines.append(f"| {sev} | {count} |")
    lines.append("")

    # Event count by kind
    by_kind = elog.count_by_kind()
    if by_kind:
        lines.append("### Event Summary by Kind")
        lines.append("")
        lines.append("| Event | Count |")
        lines.append("|-------|-------|")
        for kind, count in sorted(by_kind.items()):
            lines.append(f"| {kind} | {count} |")
        lines.append("")

    # CRITICAL events (full detail)
    critical = elog.filter(severity_min=Severity.CRITICAL)
    if critical:
        lines.append("### CRITICAL Events (require investigation)")
        lines.append("")
        lines.append("| Time (s) | Event | Vehicle | Message |")
        lines.append("|----------|-------|---------|---------|")
        for e in critical:
            vid = str(e.vehicle_id) if e.vehicle_id is not None else "-"
            lines.append(f"| {e.time:.1f} | {e.kind} | {vid} | {e.message} |")
        lines.append("")
    else:
        lines.append("### CRITICAL Events")
        lines.append("")
        lines.append("None. No critical safety events were recorded.")
        lines.append("")

    # WARNING events (first 50)
    warnings = elog.filter(severity_min=Severity.WARNING)
    # Exclude the ones already shown as CRITICAL
    warnings = [w for w in warnings if w.severity != "CRITICAL"]
    if warnings:
        shown = warnings[:50]
        lines.append(f"### WARNING Events (showing {len(shown)} of {len(warnings)})")
        lines.append("")
        lines.append("| Time (s) | Event | Vehicle | Message |")
        lines.append("|----------|-------|---------|---------|")
        for e in shown:
            vid = str(e.vehicle_id) if e.vehicle_id is not None else "-"
            msg = e.message[:100] + "..." if len(e.message) > 100 else e.message
            lines.append(f"| {e.time:.1f} | {e.kind} | {vid} | {msg} |")
        if len(warnings) > 50:
            lines.append(f"| ... | ... | ... | ({len(warnings) - 50} more warnings in event_log.jsonl) |")
        lines.append("")
    else:
        lines.append("### WARNING Events")
        lines.append("")
        lines.append("None.")
        lines.append("")

    return lines


def _build_assumptions_section() -> list[str]:
    """Document explicit model assumptions in the report."""
    return [
        "## Model Assumptions & Limitations",
        "",
        "This simulation makes the following explicit assumptions.  Results "
        "should be interpreted within these bounds.",
        "",
        "### Physics",
        "- 2-D kinematics only (no roll, pitch, terrain elevation).",
        "- First-order Euler integration at fixed dt.  Acceptable for "
        "  dt <= 0.1s and speeds <= 15 m/s.",
        "- Speed is non-negative; no reverse motion.",
        "- Fuel consumption is linear in speed; transient effects not modelled.",
        "",
        "### Position Estimation",
        "- IMU drift: additive Gaussian noise + slow bias random walk.",
        "- Real IMU errors are non-Gaussian and correlated; this model "
        "  *underestimates* worst-case drift.",
        "- Complementary filter (scalar gain) is an approximation of a Kalman "
        "  filter.  No full covariance maintained.",
        "- Uncertainty is a scalar 1-sigma proxy, optimistic in cross-track.",
        "- Landmark/GPS fixes use ground-truth position + noise.  Real "
        "  landmark detection can fail or be spoofed; not modelled.",
        "",
        "### Communications",
        "- Line-of-sight with distance-squared degradation; no multipath or fading.",
        "- Per-packet independent loss; no burst-error model.",
        "- No frequency, bandwidth, or queuing model.",
        "",
        "### Coordination",
        "- Formation is single-file behind leader; no lateral offsets.",
        "- Collision radius is centre-to-centre distance; swept-volume "
        "  overlap is not modelled.",
        "- Leader election assumes all non-failed vehicles can eventually "
        "  communicate (multi-hop not modelled).",
        "",
        "### Safe Mode Policy",
        "- Conservative: enters on ANY single trigger (high uncertainty OR "
        "  comms timeout), exits only when ALL conditions clear.",
        "- Speed reduced to 30% of max; spacing increased by 2.5x.",
        "- A vehicle in safe mode still navigates locally; it does not stop.",
        "",
    ]
