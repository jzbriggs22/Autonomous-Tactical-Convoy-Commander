"""Generate markdown report from simulation results."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from convoy_commander.metrics.collector import MetricsCollector, SimMetrics
from convoy_commander.sim.runner import SimResult
from convoy_commander.viz.plots import (
    plot_comms_graph,
    plot_metrics_summary,
    plot_position_errors,
    plot_trajectories,
)


def generate_report(result: SimResult, output_dir: Path) -> Path:
    """Generate full report with plots and markdown summary.

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
    )

    # Save metrics JSON
    result.collector.save_metrics(metrics, output_dir / "metrics.json")
    result.collector.save_time_series(output_dir / "time_series.jsonl")

    # Generate markdown
    report_path = output_dir / "report.md"
    md = _build_markdown(metrics, result, plots_dir)
    report_path.write_text(md)

    return report_path


def _build_markdown(metrics: SimMetrics, result: SimResult, plots_dir: Path) -> str:
    """Build markdown report content."""
    cfg = result.config
    m = metrics

    lines = [
        f"# Convoy Commander Simulation Report",
        f"",
        f"## Configuration",
        f"- **Scenario:** {cfg.scenario}",
        f"- **Seed:** {cfg.seed}",
        f"- **Vehicles:** {cfg.num_vehicles}",
        f"- **Duration:** {cfg.duration}s",
        f"- **GPS Available:** {cfg.gps_available}",
        f"- **Packet Loss:** {cfg.comms.packet_loss:.0%}",
        f"- **Latency:** {cfg.comms.latency_mean_ms:.0f}ms",
        f"",
        f"## Mission Summary",
        f"- **Mission Success:** {'YES' if m.mission_success else 'NO'}",
        f"- **Vehicles Arrived:** {m.vehicles_arrived}/{m.vehicles_total}",
        f"- **Average Time to Destination:** {m.avg_time_to_destination:.1f}s",
        f"",
        f"## Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
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
        f"",
        f"## Plots",
        f"",
        f"### Trajectories (True vs Estimated)",
        f"![Trajectories](plots/trajectories.png)",
        f"",
        f"### Position Estimation Error",
        f"![Position Errors](plots/position_errors.png)",
        f"",
        f"### Speed, Fuel & Uncertainty",
        f"![Metrics Summary](plots/metrics_summary.png)",
        f"",
        f"### Communications Graph (Final State)",
        f"![Comms Graph](plots/comms_graph.png)",
        f"",
    ]

    return "\n".join(lines)
