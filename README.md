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

# Run unit tests (133 tests: 30 safety-specific + 38 Phase 2 + 33 Phase 3)
python -m pytest -q
```

## Running the Simulator

### Local Installation

#### Minimum System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11 | 3.12+ |
| CPU | 2 cores, 2 GHz | 4+ cores, 3 GHz |
| RAM | 2 GB available | 4 GB available |
| Disk | 500 MB free | 2 GB free (for run archives) |
| OS | Linux, macOS, Windows (WSL2) | Linux or macOS |

A single 300-second simulation with 8 vehicles completes in approximately 30–60 seconds on a modern
laptop.  Batch runs of all 8 scenarios take 4–8 minutes.

#### Install Steps

```bash
# 1. Clone
git clone https://github.com/jzbriggs22/Autonomous-Tactical-Convoy-Commander.git
cd Autonomous-Tactical-Convoy-Commander

# 2. Create virtual environment (Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell

# 3. Install package + dev dependencies
pip install -e ".[dev]"

# 4. Verify installation
python -m convoy_commander --help
python -m pytest -q --tb=short     # 133 tests should pass
```

#### Run All Scenarios Locally

```bash
# Baseline
python -m convoy_commander run --scenario baseline --seed 42 --vehicles 8

# GPS-denied with aggressive drift
python -m convoy_commander run --scenario gps_denied --seed 42 --vehicles 8

# Degraded comms (30% loss, 200ms latency)
python -m convoy_commander run --scenario comms_degraded --seed 42 --vehicles 8

# Leader failure at t=120s
python -m convoy_commander run --scenario leader_failure --seed 42 --vehicles 8

# Dynamic obstacle insertion at t=90s
python -m convoy_commander run --scenario obstacle_pop --seed 42 --vehicles 8

# GPS spoofing attack (Phase 2)
python -m convoy_commander run --scenario gps_spoofed --seed 42 --vehicles 8

# Silent running / reduced RF emissions (Phase 2)
python -m convoy_commander run --scenario silent_running --seed 42 --vehicles 8

# Communications blackout zone (Phase 2)
python -m convoy_commander run --scenario comms_blackout --seed 42 --vehicles 8

# IMU drift spike at t=60s, GPS-denied (Phase 3)
python -m convoy_commander run --scenario sensor_drift_spike --seed 42 --vehicles 8
```

Each run writes output to `runs/<scenario>_<timestamp>/`.

---

### AWS Deployment

AWS is recommended if you need:
- **Batch parameter sweeps** across many seeds / vehicle counts simultaneously.
- **Long duration simulations** (> 30 minutes wall-clock) that would tie up a laptop.
- **Reproducible cloud archive** of run artifacts (S3 or EBS).

The simulator itself is CPU-bound and single-threaded (one Python process per run).
No GPU is required.

#### Recommended Instance Types

| Use Case | Instance | vCPU | RAM | On-Demand Cost (us-east-1) |
|----------|----------|------|-----|---------------------------|
| Single scenario testing | `t3.medium` | 2 | 4 GB | ~$0.04/hr |
| Single full run (300 s sim) | `m5.large` | 2 | 8 GB | ~$0.10/hr |
| Parallel batch (8 scenarios) | `c5.2xlarge` | 8 | 16 GB | ~$0.34/hr |
| Large-fleet sweeps (N=32+) | `c5.4xlarge` | 16 | 32 GB | ~$0.68/hr |

For quick exploratory runs a `t3.medium` Spot instance costs < $0.01.

#### AWS Setup (Ubuntu 22.04 / Amazon Linux 2023)

```bash
# 1. Launch instance (e.g., Amazon Linux 2023 AMI, t3.medium, 20 GB EBS)
# 2. SSH in and install Python 3.11
sudo dnf install -y python3.11 python3.11-pip git   # Amazon Linux 2023
# or: sudo apt install -y python3.11 python3.11-venv git  # Ubuntu 22.04

# 3. Clone repo
git clone https://github.com/jzbriggs22/Autonomous-Tactical-Convoy-Commander.git
cd Autonomous-Tactical-Convoy-Commander

# 4. Install
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 5. Run (inside a tmux or screen session for long runs)
tmux new -s convoy
python -m convoy_commander run --scenario baseline --seed 42 --vehicles 8 \
    --output /tmp/runs/baseline_42
# Detach: Ctrl+B, D

# 6. Copy results to S3
aws s3 cp /tmp/runs/ s3://my-bucket/convoy-runs/ --recursive
```

#### Parallel Batch on c5.2xlarge (8 scenarios × 1 seed)

```bash
# Install GNU parallel
sudo dnf install -y parallel          # Amazon Linux
# or: sudo apt install -y parallel    # Ubuntu

scenarios=(baseline gps_denied comms_degraded leader_failure obstacle_pop \
           gps_spoofed silent_running comms_blackout)

parallel -j 8 python -m convoy_commander run \
    --scenario {} --seed 42 --vehicles 8 \
    --output /tmp/runs/{}_42 ::: "${scenarios[@]}"
```

Expected wall-clock time on `c5.2xlarge`: ~2–4 minutes for all 8 scenarios in parallel.

#### Storage Estimates

| Artefact | Per-run Size |
|----------|-------------|
| `report.md` | ~15 KB |
| `metrics.json` | ~2 KB |
| `time_series.jsonl` | ~5–20 MB (300 s, 8 vehicles) |
| `event_log.jsonl` | ~100–500 KB |
| `plots/` (4 PNGs) | ~2–4 MB |
| **Total per run** | **~8–25 MB** |

---

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
| `gps_spoofed` | 3 GPS spoofing zones with up to 60m offset; innovation gate defends |
| `silent_running` | 5× longer broadcast interval; extended comms timeout |
| `comms_blackout` | 120m-radius blackout zone (20× loss) on convoy path |
| `sensor_drift_spike` | Sudden IMU bias spike at t=60s (GPS denied); tests estimator recovery |

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
  - Axis-aligned rectangle obstacles (exact clearance computation via slab method)
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
| Comms by message type | Per-type sent/delivered/dropped breakdown (in report and metrics.json) |
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
  core/           World model (obstacles, road graph, spoof regions), physics,
                  config (pydantic with safety validators), event log
  vehicles/       Vehicle dynamics, fuel, position estimator (with innovation gate),
                  CommsMode enum
  comms/          Network model (range, loss, latency, blackout regions), messages
  planning/       Global planner (multi-objective A* on road graph),
                  local planner (DWA-lite with potential-field fallback)
  coordination/   Formation control, Bully leader election, waypoint allocation
  supervisor/     Centralised supervisor agent (fleet-level anomaly detection)
  sim/            Simulation runner loop, 8 scenario definitions
  metrics/        Per-step collection, final aggregation, JSON export
  viz/            Matplotlib plots, markdown report with safety audit section
  cli.py          CLI entry point
tests/            133 tests: physics, estimator, comms, planning, election,
                  sim, safety (30), Phase 2 features (38), Phase 3 features (33)
runs/             Output directory for simulation results
```

## Limitations and Next Steps

**Phase 2 — Implemented:**
- GPS spoofing model with false position fixes and innovation gating (5σ default)
- DWA-lite local planner (replaces pure potential field; falls back to potential field)
- Multi-objective route planning — weighted cost: time + fuel + per-edge risk
- Behaviour switching: `CommsMode.SILENT` suppresses broadcasts (emissions control)
- Communications blackout zone scenario (20× loss multiplier region)
- Centralised supervisor agent: detects stuck vehicles, convoy splits, isolated nodes
- Formation degraded logging when no operational leader exists

**Phase 3 — Implemented:**
- **Axis-aligned rectangle obstacles** (`PolyObstacle`): exact clearance via slab method; integrated in local planner, global planner risk annotation, and collision detection
- **Drift spike injection**: `apply_drift_spike(magnitude)` on `PositionEstimator`; `sensor_drift_spike` scenario injects 8m IMU bias jump at t=60s with safe-mode escalation
- **Per-message-type bandwidth statistics**: `CommsNetwork.get_stats_by_type()` returns per-`MessageType` sent/delivered/dropped counts; exposed in `metrics.json` and the markdown report's new "Communications Bandwidth by Message Type" table
- `DRIFT_SPIKE` event kind added to the structured audit trail

**Remaining limitations:**
- 2D only — no terrain elevation or 3D dynamics
- DWA forward simulation uses point model (no swept volume)
- Comms model is distance-based with no frequency / bandwidth modeling
- Leader election assumes all non-failed vehicles eventually hear each other (multi-hop not modelled)
- Scalar uncertainty proxy is optimistic in cross-track direction (2×2 covariance / EKF not yet implemented)
- IMU drift model is Gaussian (real drift is heavier-tailed)
- No persistent vehicle-to-vehicle state sharing (each vehicle only uses the latest broadcast)
- Task allocation uses distributed greedy (CBBA-lite auction not yet implemented)

**Future extensions:**
- Terrain and elevation modeling
- Full covariance tracking (EKF replacing scalar uncertainty)
- Multi-hop mesh comms model
- CBBA-lite auction for task allocation
- Pareto frontier visualisation for multi-objective trade-offs
