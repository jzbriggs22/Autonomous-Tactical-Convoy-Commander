"""Evaluation harness: run multiple scenarios x seeds, aggregate results.

Produces a summary.md with a single table that shows how the fleet performs
across all operational contexts.  Designed to be the first artifact a
reviewer reads.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from convoy_commander.metrics.collector import SimMetrics
from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.scenarios import get_scenario
from convoy_commander.stamp import collect_stamp
from convoy_commander.viz.report import generate_report


# Default evaluation matrix
DEFAULT_SCENARIOS: list[str] = [
    "baseline",
    "gps_denied",
    "comms_degraded",
    "leader_failure",
    "comms_blackout",
    "platooning",
    "mesh_relay",
]

DEFAULT_SEEDS: list[int] = [42, 123, 7]

EVAL_DURATION: float = 60.0
EVAL_VEHICLES: int = 8


@dataclass
class RunRecord:
    """One row in the evaluation summary."""

    scenario: str
    seed: int
    success: bool
    vehicles_arrived: int
    vehicles_total: int
    collisions: int
    near_misses: int
    avg_time_to_dest: float
    total_fuel: float
    comms_delivery_pct: float
    safe_mode_activations: int
    avg_position_error: float
    wall_clock_seconds: float
    run_dir: str


@dataclass
class EvalSummary:
    """Aggregated evaluation results."""

    timestamp: str
    total_runs: int
    scenarios: list[str]
    seeds: list[int]
    duration: float
    num_vehicles: int
    runs: list[RunRecord]
    overall_success_rate: float = 0.0
    total_collisions: int = 0
    total_near_misses: int = 0
    avg_fuel: float = 0.0
    avg_comms_delivery: float = 0.0
    avg_eta: float = 0.0


def run_evaluation(
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    duration: float = EVAL_DURATION,
    num_vehicles: int = EVAL_VEHICLES,
    output_base: Path | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> tuple[EvalSummary, Path]:
    """Run the full evaluation matrix and generate summary.

    Returns (summary, summary_dir_path).
    """
    scenarios = scenarios or list(DEFAULT_SCENARIOS)
    seeds = seeds or list(DEFAULT_SEEDS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_base is None:
        output_base = Path("eval_results")
    summary_dir = output_base / f"sweep_{timestamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunRecord] = []
    total = len(scenarios) * len(seeds)
    completed = 0

    for scenario_name in scenarios:
        for seed in seeds:
            config = get_scenario(
                scenario_name,
                seed=seed,
                vehicles=num_vehicles,
                duration=duration,
            )

            run_dir = summary_dir / f"{scenario_name}_seed{seed}"

            runner = SimRunner(config)
            wall_start = time.time()
            result = runner.run()
            wall_elapsed = time.time() - wall_start

            # Generate per-run report (includes stamp via SimResult)
            generate_report(result, run_dir)

            # Extract metrics
            metrics = result.collector.compute_final(
                result.vehicles,
                result.comms.total_sent,
                result.comms.total_delivered,
                result.comms.total_dropped,
                config.duration,
                comms_by_type=result.comms.get_stats_by_type(),
            )

            record = RunRecord(
                scenario=scenario_name,
                seed=seed,
                success=metrics.mission_success,
                vehicles_arrived=metrics.vehicles_arrived,
                vehicles_total=metrics.vehicles_total,
                collisions=metrics.collision_count,
                near_misses=metrics.near_miss_count,
                avg_time_to_dest=metrics.avg_time_to_destination,
                total_fuel=metrics.total_fuel_used,
                comms_delivery_pct=metrics.comms_delivery_ratio * 100,
                safe_mode_activations=metrics.num_safe_mode_activations,
                avg_position_error=metrics.avg_position_error,
                wall_clock_seconds=wall_elapsed,
                run_dir=str(run_dir.relative_to(summary_dir)),
            )
            runs.append(record)

            completed += 1
            if progress_callback:
                progress_callback(completed, total, scenario_name, seed)

    # Build summary
    summary = EvalSummary(
        timestamp=timestamp,
        total_runs=len(runs),
        scenarios=scenarios,
        seeds=seeds,
        duration=duration,
        num_vehicles=num_vehicles,
        runs=runs,
    )

    # Compute aggregates
    if runs:
        successes = sum(1 for r in runs if r.success)
        summary.overall_success_rate = successes / len(runs) * 100
        summary.total_collisions = sum(r.collisions for r in runs)
        summary.total_near_misses = sum(r.near_misses for r in runs)
        summary.avg_fuel = sum(r.total_fuel for r in runs) / len(runs)
        summary.avg_comms_delivery = sum(r.comms_delivery_pct for r in runs) / len(runs)
        eta_runs = [r for r in runs if r.avg_time_to_dest > 0]
        summary.avg_eta = (
            sum(r.avg_time_to_dest for r in eta_runs) / len(eta_runs) if eta_runs else 0.0
        )

    # Write summary artifacts
    _write_summary_md(summary, summary_dir)
    _write_summary_json(summary, summary_dir)

    return summary, summary_dir


def _write_summary_md(summary: EvalSummary, summary_dir: Path) -> None:
    """Write the main summary.md — the evaluation artifact."""
    lines = [
        "# Convoy Commander Evaluation Summary",
        "",
        f"**Date:** {summary.timestamp}",
        f"**Runs:** {summary.total_runs} "
        f"({len(summary.scenarios)} scenarios x {len(summary.seeds)} seeds)",
        f"**Duration per run:** {summary.duration}s | **Vehicles:** {summary.num_vehicles}",
        "",
        "## Overall Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Success Rate | {summary.overall_success_rate:.0f}% "
        f"({sum(1 for r in summary.runs if r.success)}/{len(summary.runs)}) |",
        f"| Total Collisions | {summary.total_collisions} |",
        f"| Total Near Misses | {summary.total_near_misses} |",
        f"| Avg Fuel Used | {summary.avg_fuel:.1f} |",
        f"| Avg Comms Delivery | {summary.avg_comms_delivery:.1f}% |",
        f"| Avg Time to Dest | {summary.avg_eta:.1f}s |",
        "",
        "## Per-Run Results",
        "",
        "| Scenario | Seed | Result | Arrived | Collisions | Near Miss "
        "| ETA (s) | Fuel | Comms % | Safe Mode | Pos Err (m) | Wall (s) |",
        "|----------|------|--------|---------|------------|---------- "
        "|---------|------|---------|-----------|-------------|----------|",
    ]

    for r in summary.runs:
        ok = "PASS" if r.success else "FAIL"
        lines.append(
            f"| {r.scenario} | {r.seed} | {ok} | "
            f"{r.vehicles_arrived}/{r.vehicles_total} | "
            f"{r.collisions} | {r.near_misses} | "
            f"{r.avg_time_to_dest:.1f} | {r.total_fuel:.1f} | "
            f"{r.comms_delivery_pct:.0f}% | {r.safe_mode_activations} | "
            f"{r.avg_position_error:.2f} | {r.wall_clock_seconds:.1f} |"
        )

    # Per-scenario aggregates
    lines += [
        "",
        "## Per-Scenario Averages",
        "",
        "| Scenario | Success Rate | Avg Collisions | Avg Near Miss "
        "| Avg ETA | Avg Fuel | Avg Comms % |",
        "|----------|-------------|----------------|-------------- "
        "|---------|----------|-------------|",
    ]

    for scenario in summary.scenarios:
        s_runs = [r for r in summary.runs if r.scenario == scenario]
        n = len(s_runs)
        if n == 0:
            continue
        sr = sum(1 for r in s_runs if r.success) / n * 100
        ac = sum(r.collisions for r in s_runs) / n
        anm = sum(r.near_misses for r in s_runs) / n
        aeta = sum(r.avg_time_to_dest for r in s_runs) / n
        af = sum(r.total_fuel for r in s_runs) / n
        acd = sum(r.comms_delivery_pct for r in s_runs) / n
        lines.append(
            f"| {scenario} | {sr:.0f}% | {ac:.1f} | {anm:.1f} | "
            f"{aeta:.1f} | {af:.1f} | {acd:.0f}% |"
        )

    lines += [
        "",
        "---",
        f"*Generated by `convoy_commander evaluate` at {summary.timestamp}*",
        "",
    ]

    (summary_dir / "summary.md").write_text("\n".join(lines))


def _write_summary_json(summary: EvalSummary, summary_dir: Path) -> None:
    """Write machine-readable summary."""
    data = {
        "timestamp": summary.timestamp,
        "total_runs": summary.total_runs,
        "scenarios": summary.scenarios,
        "seeds": summary.seeds,
        "duration": summary.duration,
        "num_vehicles": summary.num_vehicles,
        "overall_success_rate": summary.overall_success_rate,
        "total_collisions": summary.total_collisions,
        "total_near_misses": summary.total_near_misses,
        "avg_fuel": summary.avg_fuel,
        "avg_comms_delivery": summary.avg_comms_delivery,
        "avg_eta": summary.avg_eta,
        "runs": [asdict(r) for r in summary.runs],
    }
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(data, f, indent=2)
