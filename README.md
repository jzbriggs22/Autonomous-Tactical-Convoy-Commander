# Autonomous Tactical Convoy Commander

Simulates a fleet of unmanned trucks coordinating in GPS-denied environments with
degraded communications, decentralized decision-making, and safe fallback modes.

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Run baseline scenario
python -m convoy_commander run --scenario baseline --seed 42

# Run GPS-denied scenario with 8 vehicles
python -m convoy_commander run --scenario gps_denied --seed 42 --vehicles 8

# Run unit tests
python -m pytest -q
```

## CLI Reference

```
python -m convoy_commander run --scenario <name> [options]

Options:
  --scenario  Scenario name (baseline|gps_denied|comms_degraded|leader_failure|obstacle_pop)
  --seed      Random seed (default: 42)
  --vehicles  Number of vehicles (default: 8)
  --loss      Packet loss rate 0-1 (overrides scenario default)
  --latency   Mean latency in ms (overrides scenario default)
  --duration  Simulation duration in seconds (overrides scenario default)
  --output    Output directory (default: runs/<scenario>_<timestamp>/)
```

Each run produces:
- `report.md` — markdown report with metrics tables and embedded plot references
- `plots/` — trajectory maps, position error, speed/fuel/uncertainty timelines, comms graph
- `metrics.json` — machine-readable metrics
- `time_series.jsonl` — per-vehicle per-timestep data

## Scenarios

| Scenario | Description |
|----------|-------------|
| `baseline` | GPS available, low packet loss (2%), low latency (30ms) |
| `gps_denied` | No GPS, elevated IMU drift, landmark fixes only |
| `comms_degraded` | 30% packet loss, 200ms latency, reduced comms range |
| `leader_failure` | Leader vehicle breaks down at t=120s, triggers re-election |
| `obstacle_pop` | Large obstacle appears at t=90s, forces global replanning |

## Models and Assumptions

### Vehicle Dynamics
- 2D kinematic model (bicycle-like): position, heading, speed
- Constrained max speed (15 m/s), acceleration (3 m/s²), deceleration (5 m/s²), turn rate (0.8 rad/s)
- Fuel model: idle consumption + speed-proportional consumption

### Position Estimation
- Dead-reckoning propagation with additive drift noise and a slow bias random walk
- Occasional absolute fixes reduce uncertainty via a complementary filter (Kalman-like gain)
- GPS fix (std 0.5m) available in baseline; landmark fixes (std 1.0m, range 50m) always available
- When uncertainty exceeds a threshold (default 20m), vehicle enters **safe mode**

### Communications
- Ad-hoc network: adjacency determined by Euclidean distance vs max range (200m default)
- Packet loss: base rate + distance-squared degradation + blackout region multiplier
- Latency: Gaussian-distributed, floor at 1ms
- Message types: state broadcast, intent, hazard alert, leader election, heartbeat, waypoint bid

### Coordination
- **Formation control**: consensus-based — each vehicle maintains spacing behind the leader with heading alignment from neighbor averaging
- **Leader election**: Bully algorithm — priority = fuel fraction; heartbeat timeout triggers election; highest priority wins with ID tie-breaking
- **Task allocation**: distributed greedy — all vehicles share the convoy destination; formation offsets handled by the formation controller
- **Safe mode**: triggered by high position uncertainty or comms timeout; reduces speed to 40%, doubles formation spacing, switches to local avoidance only

### Planning
- **Global planner**: A* on a road graph (weighted grid with some edges randomly removed and some diagonal shortcuts)
- **Local planner**: potential field — attractive force toward waypoint, repulsive forces from obstacles, no-go zones, and other vehicles; speed modulated by heading error, approach distance, and obstacle proximity

### World
- 2D continuous space (default 1000m x 1000m) with:
  - Road graph (grid nodes with weighted edges)
  - Circular obstacles (blocks paths)
  - No-go zones (larger, avoidance penalty)
  - Landmarks (enable position fixes within detection range)

## Metrics

| Metric | Description |
|--------|-------------|
| Mission success | At least half the vehicles reached the destination |
| Vehicles arrived | Count of vehicles that reached within 15m of the destination |
| Avg time to destination | Mean arrival time across vehicles that arrived |
| Total fuel used | Sum of fuel consumed across all vehicles |
| Convoy cohesion | Average pairwise distance between operational vehicles over time |
| Near misses | Pair-steps where separation < min_separation (8m) but > collision radius |
| Collisions | Pair-steps where separation < collision radius (4m) |
| Comms delivery ratio | Messages delivered / messages sent |
| Avg/max position error | Euclidean distance between true and estimated positions |
| Leader elections | Number of election rounds triggered |
| Safe mode activations | Number of times any vehicle entered safe mode |

## How to Add Scenarios

Create a new function in `convoy_commander/sim/scenarios.py`:

```python
def _my_scenario(**overrides: object) -> SimConfig:
    config = SimConfig(
        gps_available=False,
        duration=400.0,
    )
    config.comms.packet_loss = 0.5
    return _apply_overrides(config, **overrides)
```

Then register it in the `builders` dict inside `get_scenario()`.

## Architecture

```
convoy_commander/
  core/           World model, physics, config (pydantic)
  vehicles/       Vehicle dynamics, fuel, position estimator
  comms/          Network model (range, loss, latency), message types
  planning/       Global planner (A* on road graph), local planner (potential field)
  coordination/   Formation control, leader election, waypoint allocation
  sim/            Simulation runner loop, scenario definitions
  metrics/        Per-step collection, final aggregation, JSON export
  viz/            Matplotlib plots, markdown report generation
  cli.py          CLI entry point
tests/            Unit tests (pytest)
runs/             Output directory for simulation results
```

## Limitations and Next Steps

**Current limitations:**
- 2D only — no terrain elevation or 3D dynamics
- Potential field local planner can get stuck in local minima in dense obstacle fields
- Comms model is distance-based with no frequency / bandwidth modeling
- No GPS spoofing model (only denial)
- Leader election assumes all non-failed vehicles eventually hear each other
- No persistent vehicle-to-vehicle state sharing (each vehicle only uses the latest broadcast)

**Planned extensions:**
- Multi-objective optimization (time vs fuel vs risk) with Pareto frontier
- GPS spoofing model with false fixes and innovation gating in the estimator
- Behavior switching: "silent running" (minimal comms) vs "chatty" mode
- Optional centralized supervisor agent for comparison with decentralized approach
- Terrain and elevation modeling
- More sophisticated local planner (DWA or RRT)
