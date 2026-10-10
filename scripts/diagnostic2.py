#!/usr/bin/env python3
"""Quick diagnostic: 180s with 8 vehicles."""
import sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.scenarios import get_scenario

for scenario in ["baseline", "gps_denied", "comms_degraded", "platooning"]:
    for seed in [42, 7]:
        config = get_scenario(scenario, seed=seed, vehicles=8, duration=180)
        runner = SimRunner(config)
        t0 = time.time()
        result = runner.run()
        wall = time.time() - t0
        m = result.collector.compute_final(
            result.vehicles, result.comms.total_sent,
            result.comms.total_delivered, result.comms.total_dropped,
            config.duration)
        arrived = m.vehicles_arrived
        total = m.vehicles_total
        print(f"{scenario:20s} s={seed:3d}: "
              f"arr={arrived}/{total} col={m.collision_count} nm={m.near_miss_count:4d} "
              f"eta={m.avg_time_to_destination:5.1f}s "
              f"coh={m.convoy_cohesion_score:5.1f}m "
              f"safe={m.num_safe_mode_activations} "
              f"posErr={m.avg_position_error:.2f}m ({wall:.1f}s)")
