#!/usr/bin/env python3
"""Quick diagnostic: trace vehicle positions and speeds to identify bottlenecks."""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.scenarios import get_scenario

config = get_scenario("baseline", seed=42, vehicles=4, duration=120)
runner = SimRunner(config)

# Track vehicle states every 10s
def trace(step, total):
    t = step * config.dt
    if abs(t % 10.0) < config.dt:
        print(f"\nt={t:.0f}s:")
        for v in runner.vehicles:
            dest = v.assigned_destination
            if dest:
                d2dest = math.hypot(v.state.x - dest[0], v.state.y - dest[1])
            else:
                d2dest = -1
            print(f"  V{v.id}: pos=({v.state.x:.0f},{v.state.y:.0f}) "
                  f"spd={v.state.speed:.1f} hdg={math.degrees(v.state.heading):.0f}° "
                  f"wp={v.current_waypoint_idx}/{len(v.waypoints)} "
                  f"d2dest={d2dest:.0f}m status={v.status.name}")

result = runner.run(progress_callback=trace)

print(f"\n=== FINAL (t={config.duration}s) ===")
for v in result.vehicles:
    dest = v.assigned_destination
    d2dest = math.hypot(v.state.x - dest[0], v.state.y - dest[1]) if dest else -1
    print(f"  V{v.id}: pos=({v.state.x:.0f},{v.state.y:.0f}) "
          f"spd={v.state.speed:.1f} d2dest={d2dest:.0f}m "
          f"wp={v.current_waypoint_idx}/{len(v.waypoints)} "
          f"dist_traveled={v.total_distance:.0f}m "
          f"status={v.status.name}")

print(f"\nDestination: ({config.world.width - 80:.0f}, {config.world.height - 80:.0f})")
arrived = sum(1 for v in result.vehicles if v.has_reached_destination())
print(f"Arrived: {arrived}/{len(result.vehicles)}")

# Show route for vehicle 0
v0 = result.vehicles[0]
if v0.waypoints:
    print(f"\nV0 waypoints ({len(v0.waypoints)}):")
    for i, wp in enumerate(v0.waypoints[:10]):
        print(f"  [{i}] ({wp[0]:.0f}, {wp[1]:.0f})")
    if len(v0.waypoints) > 10:
        print(f"  ... {len(v0.waypoints) - 10} more")
    print(f"  DEST: {v0.assigned_destination}")
