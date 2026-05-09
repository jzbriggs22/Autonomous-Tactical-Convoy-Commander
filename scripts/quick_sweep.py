#!/usr/bin/env python3
"""Quick simulation sweep: focused diagnostic across key scenarios."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.multi_runner import MultiConvoyRunner
from convoy_commander.sim.scenarios import get_scenario, get_multi_convoy_scenario

SCENARIOS = [
    "baseline", "gps_denied", "comms_degraded", "leader_failure",
    "platooning", "mesh_relay", "heavy_rain", "winter_storm",
    "jammed_corridor", "multi_threat",
]
MULTI_SCENARIOS = ["two_convoy_crossing", "convoy_merge"]
SEEDS = [42, 7]
DURATION = 180.0
NUM_VEHICLES = 8


def run_single(scenario, seed):
    config = get_scenario(scenario, seed=seed, vehicles=NUM_VEHICLES)
    config.duration = max(config.duration, DURATION)
    runner = SimRunner(config)
    t0 = time.time()
    result = runner.run()
    wall = time.time() - t0
    m = result.collector.compute_final(
        result.vehicles, result.comms.total_sent,
        result.comms.total_delivered, result.comms.total_dropped,
        config.duration, comms_by_type=result.comms.get_stats_by_type(),
    )
    result.collector.populate_ew_metrics(m, result.event_log)
    by_sev = result.event_log.count_by_severity()
    by_kind = result.event_log.count_by_kind()
    return {
        "scenario": scenario, "seed": seed, "wall": round(wall, 1),
        "success": m.mission_success,
        "arrived": f"{m.vehicles_arrived}/{m.vehicles_total}",
        "collisions": m.collision_count, "near_misses": m.near_miss_count,
        "eta": round(m.avg_time_to_destination, 1),
        "fuel": round(m.total_fuel_used, 1),
        "cohesion": round(m.convoy_cohesion_score, 1),
        "comms_pct": round(m.comms_delivery_ratio * 100, 1),
        "pos_err": round(m.avg_position_error, 3),
        "safe_mode": m.num_safe_mode_activations,
        "leader_elections": m.num_leader_elections,
        "critical": by_sev.get("CRITICAL", 0),
        "warnings": by_sev.get("WARNING", 0),
        "boundary_violations": by_kind.get("boundary_violation", 0),
        "planner_fallbacks": by_kind.get("planner_fallback", 0),
    }


def run_multi(scenario, seed):
    mc_config = get_multi_convoy_scenario(scenario, seed=seed, duration=DURATION)
    runner = MultiConvoyRunner(mc_config)
    t0 = time.time()
    result = runner.run()
    wall = time.time() - t0
    agg = result.aggregate_metrics
    arrived = sum(1 for v in result.all_vehicles if v.has_reached_destination())
    total_v = len(result.all_vehicles)
    collisions = sum(v.collision_count for v in result.all_vehicles)
    near_misses = sum(v.near_miss_count for v in result.all_vehicles)
    by_sev = result.event_log.count_by_severity()
    return {
        "scenario": scenario, "seed": seed, "wall": round(wall, 1),
        "success": agg.overall_mission_success if agg else False,
        "arrived": f"{arrived}/{total_v}",
        "collisions": collisions, "near_misses": near_misses,
        "inter_convoy_col": agg.inter_convoy_collisions if agg else 0,
        "merges": agg.merge_count if agg else 0,
        "splits": agg.split_count if agg else 0,
        "yields": agg.right_of_way_yields if agg else 0,
        "critical": by_sev.get("CRITICAL", 0),
        "warnings": by_sev.get("WARNING", 0),
    }


def main():
    results = []
    total = len(SCENARIOS) * len(SEEDS) + len(MULTI_SCENARIOS) * len(SEEDS)
    done = 0

    for scenario in SCENARIOS:
        for seed in SEEDS:
            done += 1
            print(f"[{done}/{total}] {scenario} s={seed} ", end="", flush=True)
            r = run_single(scenario, seed)
            s = "OK" if r["success"] else "FAIL"
            print(f"{s} arr={r['arrived']} col={r['collisions']} nm={r['near_misses']} "
                  f"eta={r['eta']} coh={r['cohesion']} safe={r['safe_mode']} "
                  f"bv={r['boundary_violations']} ({r['wall']}s)")
            results.append(r)

    for scenario in MULTI_SCENARIOS:
        for seed in SEEDS:
            done += 1
            print(f"[{done}/{total}] {scenario} s={seed} ", end="", flush=True)
            r = run_multi(scenario, seed)
            s = "OK" if r["success"] else "FAIL"
            print(f"{s} arr={r['arrived']} col={r['collisions']} nm={r['near_misses']} "
                  f"({r['wall']}s)")
            results.append(r)

    with open("quick_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    success = sum(1 for r in results if r["success"])
    print(f"Success: {success}/{len(results)}")
    print(f"Total collisions: {sum(r['collisions'] for r in results)}")
    print(f"Total near misses: {sum(r['near_misses'] for r in results)}")


if __name__ == "__main__":
    main()
