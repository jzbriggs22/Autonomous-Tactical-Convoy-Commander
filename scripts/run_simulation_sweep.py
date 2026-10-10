#!/usr/bin/env python3
"""Comprehensive simulation sweep: run all scenarios, collect data, identify issues.

Runs both single-convoy and multi-convoy scenarios across multiple seeds,
collects detailed metrics, and outputs a JSON analysis report.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.multi_runner import MultiConvoyRunner
from convoy_commander.sim.scenarios import (
    get_scenario,
    get_multi_convoy_scenario,
    MULTI_CONVOY_SCENARIOS,
)

# All single-convoy scenarios
SINGLE_SCENARIOS = [
    "baseline",
    "gps_denied",
    "comms_degraded",
    "leader_failure",
    "obstacle_pop",
    "gps_spoofed",
    "silent_running",
    "comms_blackout",
    "sensor_drift_spike",
    "platooning",
    "mesh_relay",
    "heavy_rain",
    "winter_storm",
    "jammed_corridor",
    "mobile_jammer",
    "multi_threat",
]

MULTI_SCENARIOS = [
    "two_convoy_crossing",
    "convoy_merge",
    "convoy_split_reroute",
    "multi_convoy_contested",
]

SEEDS = [42, 123, 7, 256, 999]
DURATION = 120.0  # seconds per run
NUM_VEHICLES = 8


def run_single_scenario(scenario: str, seed: int) -> dict:
    """Run a single-convoy scenario and return metrics dict."""
    config = get_scenario(scenario, seed=seed, vehicles=NUM_VEHICLES, duration=DURATION)
    runner = SimRunner(config)
    t0 = time.time()
    result = runner.run()
    wall = time.time() - t0

    metrics = result.collector.compute_final(
        result.vehicles,
        result.comms.total_sent,
        result.comms.total_delivered,
        result.comms.total_dropped,
        config.duration,
        comms_by_type=result.comms.get_stats_by_type(),
    )

    # EW metrics
    result.collector.populate_ew_metrics(metrics, result.event_log)

    # Event counts
    by_sev = result.event_log.count_by_severity()
    by_kind = result.event_log.count_by_kind()

    return {
        "type": "single",
        "scenario": scenario,
        "seed": seed,
        "wall_seconds": round(wall, 2),
        "mission_success": metrics.mission_success,
        "vehicles_arrived": metrics.vehicles_arrived,
        "vehicles_total": metrics.vehicles_total,
        "collisions": metrics.collision_count,
        "near_misses": metrics.near_miss_count,
        "avg_time_to_dest": round(metrics.avg_time_to_destination, 2),
        "total_fuel": round(metrics.total_fuel_used, 2),
        "avg_fuel": round(metrics.avg_fuel_used, 2),
        "cohesion": round(metrics.convoy_cohesion_score, 2),
        "comms_delivery_pct": round(metrics.comms_delivery_ratio * 100, 1),
        "avg_position_error": round(metrics.avg_position_error, 3),
        "max_position_error": round(metrics.max_position_error, 3),
        "total_distance": round(metrics.total_distance_traveled, 1),
        "leader_elections": metrics.num_leader_elections,
        "safe_mode_activations": metrics.num_safe_mode_activations,
        "string_stability_max": round(metrics.string_stability_max, 4),
        "jammers_detected": metrics.jammers_detected,
        "ecm_activations": metrics.ecm_activations,
        "threat_replans": metrics.threat_replans,
        "weather_source": metrics.weather_source,
        "avg_friction": round(metrics.avg_friction_factor, 3),
        "events_critical": by_sev.get("CRITICAL", 0),
        "events_warning": by_sev.get("WARNING", 0),
        "event_kinds": {k: v for k, v in by_kind.items()
                        if v > 0 and k not in ("sim_start", "sim_end", "vehicle_spawned")},
    }


def run_multi_scenario(scenario: str, seed: int) -> dict:
    """Run a multi-convoy scenario and return metrics dict."""
    mc_config = get_multi_convoy_scenario(scenario, seed=seed, duration=DURATION)
    runner = MultiConvoyRunner(mc_config)
    t0 = time.time()
    result = runner.run()
    wall = time.time() - t0

    agg = result.aggregate_metrics
    by_sev = result.event_log.count_by_severity()
    by_kind = result.event_log.count_by_kind()

    arrived = sum(1 for v in result.all_vehicles if v.has_reached_destination())
    total_v = len(result.all_vehicles)
    collisions = sum(v.collision_count for v in result.all_vehicles)
    near_misses = sum(v.near_miss_count for v in result.all_vehicles)

    return {
        "type": "multi",
        "scenario": scenario,
        "seed": seed,
        "wall_seconds": round(wall, 2),
        "mission_success": agg.overall_mission_success if agg else False,
        "vehicles_arrived": arrived,
        "vehicles_total": total_v,
        "collisions": collisions,
        "near_misses": near_misses,
        "inter_convoy_collisions": agg.inter_convoy_collisions if agg else 0,
        "merge_count": agg.merge_count if agg else 0,
        "split_count": agg.split_count if agg else 0,
        "right_of_way_yields": agg.right_of_way_yields if agg else 0,
        "events_critical": by_sev.get("CRITICAL", 0),
        "events_warning": by_sev.get("WARNING", 0),
        "event_kinds": {k: v for k, v in by_kind.items()
                        if v > 0 and k not in ("sim_start", "sim_end", "vehicle_spawned")},
    }


def analyze_results(results: list[dict]) -> dict:
    """Analyze all results to identify issues and improvement targets."""
    analysis = {
        "total_runs": len(results),
        "overall_success_rate": 0,
        "total_collisions": 0,
        "total_near_misses": 0,
        "scenarios_with_failures": [],
        "scenarios_with_collisions": [],
        "scenarios_with_high_near_misses": [],
        "scenarios_with_low_comms": [],
        "scenarios_with_high_position_error": [],
        "scenarios_with_safe_mode_issues": [],
        "per_scenario_summary": {},
    }

    # Group by scenario
    by_scenario: dict[str, list[dict]] = {}
    for r in results:
        by_scenario.setdefault(r["scenario"], []).append(r)

    successes = sum(1 for r in results if r["mission_success"])
    analysis["overall_success_rate"] = round(successes / len(results) * 100, 1) if results else 0
    analysis["total_collisions"] = sum(r["collisions"] for r in results)
    analysis["total_near_misses"] = sum(r["near_misses"] for r in results)

    for scenario, runs in sorted(by_scenario.items()):
        n = len(runs)
        success_rate = sum(1 for r in runs if r["mission_success"]) / n * 100
        avg_collisions = sum(r["collisions"] for r in runs) / n
        avg_near_misses = sum(r["near_misses"] for r in runs) / n
        avg_arrived = sum(r["vehicles_arrived"] for r in runs) / n
        avg_total = sum(r["vehicles_total"] for r in runs) / n

        summary = {
            "success_rate": round(success_rate, 1),
            "avg_collisions": round(avg_collisions, 1),
            "avg_near_misses": round(avg_near_misses, 1),
            "avg_arrived": round(avg_arrived, 1),
            "avg_total": round(avg_total, 1),
            "avg_wall_seconds": round(sum(r["wall_seconds"] for r in runs) / n, 2),
            "events_critical_total": sum(r["events_critical"] for r in runs),
            "events_warning_total": sum(r["events_warning"] for r in runs),
        }

        # Single-convoy specific
        single_runs = [r for r in runs if r["type"] == "single"]
        if single_runs:
            summary["avg_fuel"] = round(sum(r["avg_fuel"] for r in single_runs) / len(single_runs), 2)
            summary["avg_comms_delivery"] = round(
                sum(r["comms_delivery_pct"] for r in single_runs) / len(single_runs), 1
            )
            summary["avg_position_error"] = round(
                sum(r["avg_position_error"] for r in single_runs) / len(single_runs), 3
            )
            summary["avg_cohesion"] = round(
                sum(r["cohesion"] for r in single_runs) / len(single_runs), 1
            )
            eta_runs = [r for r in single_runs if r["avg_time_to_dest"] > 0]
            summary["avg_eta"] = round(
                sum(r["avg_time_to_dest"] for r in eta_runs) / len(eta_runs), 1
            ) if eta_runs else 0
            summary["avg_safe_mode"] = round(
                sum(r["safe_mode_activations"] for r in single_runs) / len(single_runs), 1
            )

        analysis["per_scenario_summary"][scenario] = summary

        # Flag issues
        if success_rate < 100:
            analysis["scenarios_with_failures"].append(
                {"scenario": scenario, "success_rate": success_rate}
            )
        if avg_collisions > 0:
            analysis["scenarios_with_collisions"].append(
                {"scenario": scenario, "avg_collisions": avg_collisions}
            )
        if avg_near_misses > 5:
            analysis["scenarios_with_high_near_misses"].append(
                {"scenario": scenario, "avg_near_misses": avg_near_misses}
            )
        if single_runs:
            avg_comms = sum(r["comms_delivery_pct"] for r in single_runs) / len(single_runs)
            if avg_comms < 80:
                analysis["scenarios_with_low_comms"].append(
                    {"scenario": scenario, "avg_comms_delivery": round(avg_comms, 1)}
                )
            avg_pos_err = sum(r["avg_position_error"] for r in single_runs) / len(single_runs)
            if avg_pos_err > 2.0:
                analysis["scenarios_with_high_position_error"].append(
                    {"scenario": scenario, "avg_pos_error": round(avg_pos_err, 3)}
                )
            avg_sm = sum(r["safe_mode_activations"] for r in single_runs) / len(single_runs)
            if avg_sm > 10:
                analysis["scenarios_with_safe_mode_issues"].append(
                    {"scenario": scenario, "avg_safe_mode_activations": round(avg_sm, 1)}
                )

    return analysis


def main():
    results = []
    total = len(SINGLE_SCENARIOS) * len(SEEDS) + len(MULTI_SCENARIOS) * len(SEEDS)
    done = 0

    print(f"=== Simulation Sweep: {total} runs ===\n")

    # Single-convoy scenarios
    for scenario in SINGLE_SCENARIOS:
        for seed in SEEDS:
            done += 1
            print(f"  [{done}/{total}] {scenario} seed={seed} ... ", end="", flush=True)
            try:
                r = run_single_scenario(scenario, seed)
                status = "PASS" if r["mission_success"] else "FAIL"
                print(f"{status} ({r['wall_seconds']:.1f}s) "
                      f"arr={r['vehicles_arrived']}/{r['vehicles_total']} "
                      f"col={r['collisions']} nm={r['near_misses']}")
                results.append(r)
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "type": "single", "scenario": scenario, "seed": seed,
                    "error": str(e), "mission_success": False,
                    "vehicles_arrived": 0, "vehicles_total": NUM_VEHICLES,
                    "collisions": 0, "near_misses": 0,
                    "wall_seconds": 0, "events_critical": 0, "events_warning": 0,
                })

    # Multi-convoy scenarios
    for scenario in MULTI_SCENARIOS:
        for seed in SEEDS:
            done += 1
            print(f"  [{done}/{total}] {scenario} seed={seed} ... ", end="", flush=True)
            try:
                r = run_multi_scenario(scenario, seed)
                status = "PASS" if r["mission_success"] else "FAIL"
                print(f"{status} ({r['wall_seconds']:.1f}s) "
                      f"arr={r['vehicles_arrived']}/{r['vehicles_total']} "
                      f"col={r['collisions']} nm={r['near_misses']}")
                results.append(r)
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "type": "multi", "scenario": scenario, "seed": seed,
                    "error": str(e), "mission_success": False,
                    "vehicles_arrived": 0, "vehicles_total": 0,
                    "collisions": 0, "near_misses": 0,
                    "wall_seconds": 0, "events_critical": 0, "events_warning": 0,
                })

    # Analyze
    analysis = analyze_results(results)

    # Write results
    output = {
        "sweep_config": {
            "single_scenarios": SINGLE_SCENARIOS,
            "multi_scenarios": MULTI_SCENARIOS,
            "seeds": SEEDS,
            "duration": DURATION,
            "num_vehicles": NUM_VEHICLES,
        },
        "results": results,
        "analysis": analysis,
    }

    out_path = Path("sim_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE: {len(results)} runs")
    print(f"{'='*60}")
    print(f"Overall success rate: {analysis['overall_success_rate']}%")
    print(f"Total collisions: {analysis['total_collisions']}")
    print(f"Total near misses: {analysis['total_near_misses']}")

    if analysis["scenarios_with_failures"]:
        print(f"\nSCENARIOS WITH FAILURES:")
        for s in analysis["scenarios_with_failures"]:
            print(f"  {s['scenario']}: {s['success_rate']}% success")

    if analysis["scenarios_with_collisions"]:
        print(f"\nSCENARIOS WITH COLLISIONS:")
        for s in analysis["scenarios_with_collisions"]:
            print(f"  {s['scenario']}: avg {s['avg_collisions']:.1f} collisions")

    if analysis["scenarios_with_high_near_misses"]:
        print(f"\nSCENARIOS WITH HIGH NEAR MISSES (>5 avg):")
        for s in analysis["scenarios_with_high_near_misses"]:
            print(f"  {s['scenario']}: avg {s['avg_near_misses']:.1f}")

    if analysis["scenarios_with_low_comms"]:
        print(f"\nSCENARIOS WITH LOW COMMS (<80%):")
        for s in analysis["scenarios_with_low_comms"]:
            print(f"  {s['scenario']}: {s['avg_comms_delivery']}%")

    if analysis["scenarios_with_high_position_error"]:
        print(f"\nSCENARIOS WITH HIGH POSITION ERROR (>2m):")
        for s in analysis["scenarios_with_high_position_error"]:
            print(f"  {s['scenario']}: {s['avg_pos_error']}m")

    if analysis["scenarios_with_safe_mode_issues"]:
        print(f"\nSCENARIOS WITH EXCESSIVE SAFE MODE (>10 avg):")
        for s in analysis["scenarios_with_safe_mode_issues"]:
            print(f"  {s['scenario']}: avg {s['avg_safe_mode_activations']:.1f}")

    print(f"\nDetailed results: {out_path}")


if __name__ == "__main__":
    main()
