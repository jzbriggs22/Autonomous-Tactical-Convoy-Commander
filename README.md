# Autonomous Tactical Convoy Commander

Simulates a fleet of unmanned trucks coordinating in GPS-denied environments with
degraded communications, decentralized decision-making, and safe fallback modes.

**Safety-critical design**: conservative defaults, explicit assumptions documented
at every model boundary, structured audit trail for every safety-relevant event.

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Run baseline scenario
python -m convoy_commander run --scenario baseline --seed 42

# Run GPS-denied scenario with 8 vehicles
python -m convoy_commander run --scenario gps_denied --seed 42 --vehicles 8

# Run unit tests (62 tests including 30 safety-specific)
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
- `report.md` — markdown report with metrics, safety audit, and model assumptions
- `plots/` — trajectory maps, position error, speed/fuel/uncertainty timelines, comms graph
- `metrics.json` — machine-readable metrics
- `time_series.jsonl` — per-vehicle per-timestep data
- `event_log.jsonl` — structured safety audit trail (every event with timestamp, severity, vehicle ID)

## Safety Design

### Conservative Defaults
All defaults are chosen conservatively. When in doubt the simulator errs on
the side of caution:
- Max speed: 12 m/s (~43 km/h), not the vehicle's physical limit
- Min separation: 10 m (hard constraint); formation spacing: 25 m
- Safe mode triggers at uncertainty > 15 m OR comms lost > 8 s
- Safe mode reduces speed to 30% and increases spacing by 2.5x
- Deceleration must always exceed acceleration (validated at config time)

### Config Validation
Unsafe parameter combinations are rejected **before simulation starts**:
- `max_decel >= max_accel` (vehicle can always stop as fast as it accelerates)
- `collision_radius < min_separation < formation_spacing` (proper safety hierarchy)
- `max_speed * dt < 0.5 * smallest_obstacle_radius` (no obstacle skipping)
- `collision_radius >= 0.5 * vehicle_body_diagonal` (physically meaningful detection)

### Structured Audit Trail
Every safety-relevant event is recorded with:
- Monotonic simulation timestamp
- Severity level (DEBUG / INFO / WARNING / CRITICAL)
- Event kind (collision, near_miss, safe_mode_enter, leader_elected, etc.)
- Vehicle ID (where applicable)
- Human-readable message and structured details

Events are written to `event_log.jsonl` and summarised in the report's
**Safety Audit** section. CRITICAL events (collisions, invariant violations)
are always shown in full.

### Pre/Post-Condition Checks
- `KinematicState.step()`: rejects NaN commands, negative dt, non-positive limits
- `FuelState.consume()`: caps at remaining fuel, never goes negative
- `PositionEstimator`: caps uncertainty to prevent float overflow
- Sim runner: post-step boundary clamping, speed limit enforcement, obstacle
  intrusion detection (emergency stop)

### Explicit Assumptions
Every module documents its assumptions in docstrings (prefixed `SAFETY-CRITICAL
ASSUMPTIONS:`). The generated report includes a full **Model Assumptions &
Limitations** section. Key assumptions:
- First-order Euler integration (acceptable at dt <= 0.1 s)
- IMU drift model is Gaussian (underestimates real worst-case drift)
- Scalar uncertainty proxy (optimistic in cross-track direction)
- Landmark detection cannot fail or be spoofed (not yet modelled)

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
- Constrained max speed (12 m/s), acceleration (2 m/s²), deceleration (4 m/s²), turn rate (0.6 rad/s)
- Fuel model: idle consumption + speed-proportional consumption
- Speed is non-negative; no reverse motion

### Position Estimation
- Dead-reckoning propagation with additive drift noise and a slow bias random walk
- Occasional absolute fixes reduce uncertainty via a complementary filter (Kalman-like gain)
- GPS fix (std 0.5m) available in baseline; landmark fixes (std 1.0m, range 50m) always available
- When uncertainty exceeds threshold (default 15m), vehicle enters **safe mode**
- Fix functions return innovation magnitude for anomaly detection

### Communications
- Ad-hoc network: adjacency determined by Euclidean distance vs max range (200m default)
- Packet loss: base rate + distance-squared degradation + blackout region multiplier
- Latency: Gaussian-distributed, floor at 1ms
- Message types: state broadcast, intent, hazard alert, leader election, heartbeat, waypoint bid

### Coordination
- **Formation control**: consensus-based — each vehicle maintains spacing behind the leader with heading alignment from neighbor averaging
- **Leader election**: Bully algorithm — priority = fuel fraction; heartbeat timeout triggers election; highest priority wins with ID tie-breaking
- **Task allocation**: distributed greedy — all vehicles share the convoy destination; formation offsets handled by the formation controller
- **Safe mode policy**: conservative — enters on ANY single trigger (high uncertainty OR comms timeout), exits only when ALL conditions clear

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
| Near misses | Pair-steps where separation < min_separation (10m) but > collision radius |
| Collisions | Pair-steps where separation < collision radius (5m) |
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
  core/           World model, physics, config (pydantic), event log
  vehicles/       Vehicle dynamics, fuel, position estimator
  comms/          Network model (range, loss, latency), message types
  planning/       Global planner (A* on road graph), local planner (potential field)
  coordination/   Formation control, leader election, waypoint allocation
  sim/            Simulation runner loop, scenario definitions
  metrics/        Per-step collection, final aggregation, JSON export
  viz/            Matplotlib plots, markdown report generation
  cli.py          CLI entry point
tests/            62 tests (physics, estimator, comms, planning, election, sim, safety)
runs/             Output directory for simulation results
```

## Limitations and Next Steps

**Current limitations:**
- 2D only — no terrain elevation or 3D dynamics
- Potential field local planner can get stuck in local minima in dense obstacle fields
- Comms model is distance-based with no frequency / bandwidth modeling
- No GPS spoofing model (only denial)
- Leader election assumes all non-failed vehicles eventually hear each other
- Scalar uncertainty proxy is optimistic in cross-track direction
- IMU drift model is Gaussian (real drift is heavier-tailed)
- No persistent vehicle-to-vehicle state sharing (each vehicle only uses the latest broadcast)

**Planned extensions:**
- Multi-objective optimization (time vs fuel vs risk) with Pareto frontier
- GPS spoofing model with false fixes and innovation gating in the estimator
- Behavior switching: "silent running" (minimal comms) vs "chatty" mode
- Optional centralized supervisor agent for comparison with decentralized approach
- Terrain and elevation modeling
- More sophisticated local planner (DWA or RRT)
- Full covariance tracking in the estimator (replacing scalar uncertainty)
