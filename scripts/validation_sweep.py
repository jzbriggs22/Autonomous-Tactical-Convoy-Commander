#!/usr/bin/env python3
"""Validation sweep: confirm improvements across all scenarios."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.multi_runner import MultiConvoyRunner
from convoy_commander.sim.scenarios import get_scenario, get_multi_convoy_scenario

SCENARIOS = [
    "baseline", "gps_denied", "comms_degraded", "leader_failure",
    "platooning", "mesh_relay", "heavy_rain", "winter_storm",
    "jammed_corridor", "multi_threat", "gps_spoofed", "silent_running",
    "comms_blackout", "sensor_drift_spike",
]
MULTI = ["two_convoy_crossing", "convoy_merge", "convoy_split_reroute", "multi_convoy_contested"]
SEEDS = [42, 7, 123]
DUR = 180.0
NV = 8

results = []
total = len(SCENARIOS) * len(SEEDS) + len(MULTI) * len(SEEDS)
done = 0

print(f"=== Validation Sweep: {total} runs, {DUR}s each ===\n")

for sc in SCENARIOS:
    for seed in SEEDS:
        done += 1
        cfg = get_scenario(sc, seed=seed, vehicles=NV, duration=DUR)
        runner = SimRunner(cfg)
        t0 = time.time()
        res = runner.run()
        wall = time.time() - t0
        m = res.collector.compute_final(
            res.vehicles, res.comms.total_sent,
            res.comms.total_delivered, res.comms.total_dropped,
            cfg.duration, comms_by_type=res.comms.get_stats_by_type())
        res.collector.populate_ew_metrics(m, res.event_log)
        s = "OK" if m.mission_success else "FAIL"
        r = {
            "scenario": sc, "seed": seed, "type": "single",
            "success": m.mission_success, "arrived": m.vehicles_arrived,
            "total": m.vehicles_total, "collisions": m.collision_count,
            "near_misses": m.near_miss_count, "eta": round(m.avg_time_to_destination, 1),
            "cohesion": round(m.convoy_cohesion_score, 1),
            "comms_pct": round(m.comms_delivery_ratio * 100, 1),
            "pos_err": round(m.avg_position_error, 2),
            "safe_mode": m.num_safe_mode_activations,
            "wall": round(wall, 1),
        }
        results.append(r)
        print(f"[{done:2d}/{total}] {sc:22s} s={seed:3d} {s:4s} "
              f"arr={m.vehicles_arrived}/{m.vehicles_total} "
              f"col={m.collision_count:4d} nm={m.near_miss_count:4d} "
              f"eta={m.avg_time_to_destination:5.1f}s ({wall:.0f}s)")

for sc in MULTI:
    for seed in SEEDS:
        done += 1
        mc = get_multi_convoy_scenario(sc, seed=seed, duration=DUR)
        runner = MultiConvoyRunner(mc)
        t0 = time.time()
        res = runner.run()
        wall = time.time() - t0
        agg = res.aggregate_metrics
        arr = sum(1 for v in res.all_vehicles if v.has_reached_destination())
        tv = len(res.all_vehicles)
        col = sum(v.collision_count for v in res.all_vehicles)
        nm = sum(v.near_miss_count for v in res.all_vehicles)
        ok = agg.overall_mission_success if agg else False
        s = "OK" if ok else "FAIL"
        r = {
            "scenario": sc, "seed": seed, "type": "multi",
            "success": ok, "arrived": arr, "total": tv,
            "collisions": col, "near_misses": nm,
            "wall": round(wall, 1),
        }
        results.append(r)
        print(f"[{done:2d}/{total}] {sc:22s} s={seed:3d} {s:4s} "
              f"arr={arr}/{tv} col={col:4d} nm={nm:4d} ({wall:.0f}s)")

# Summary
print(f"\n{'='*60}")
print(f"VALIDATION COMPLETE: {len(results)} runs")
print(f"{'='*60}")
success_count = sum(1 for r in results if r["success"])
print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
print(f"Total collisions: {sum(r['collisions'] for r in results)}")
print(f"Total near misses: {sum(r['near_misses'] for r in results)}")

# Per-scenario
print(f"\nPer-scenario breakdown:")
from collections import defaultdict
by_sc = defaultdict(list)
for r in results:
    by_sc[r["scenario"]].append(r)
for sc, runs in sorted(by_sc.items()):
    sr = sum(1 for r in runs if r["success"]) / len(runs) * 100
    ac = sum(r["collisions"] for r in runs) / len(runs)
    anm = sum(r["near_misses"] for r in runs) / len(runs)
    aa = sum(r["arrived"] for r in runs) / len(runs)
    at = runs[0]["total"]
    print(f"  {sc:25s}: {sr:5.0f}% success, avg arr={aa:.1f}/{at}, "
          f"avg col={ac:.0f}, avg nm={anm:.0f}")

with open("validation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to validation_results.json")
