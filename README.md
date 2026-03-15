# Autonomous Tactical Convoy Commander

[![CI](https://github.com/jzbriggs22/Autonomous-Tactical-Convoy-Commander/actions/workflows/ci.yml/badge.svg)](https://github.com/jzbriggs22/Autonomous-Tactical-Convoy-Commander/actions/workflows/ci.yml)

Simulates a fleet of unmanned trucks coordinating in GPS-denied environments with
degraded communications, decentralized decision-making, and safe fallback modes.

**Safety-critical design**: conservative defaults, explicit assumptions documented
at every model boundary, structured audit trail for every safety-relevant event.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module layout, trust boundaries, and data flow.

## Quick Evaluation (3 minutes)

```bash
make install       # pip install -e ".[dev]"
make test          # 278 tests
make evaluate      # 7 scenarios x 3 seeds -> eval_results/sweep_*/summary.md
```

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Run baseline scenario
python -m convoy_commander run --scenario baseline --seed 42

# Run GPS-denied scenario with 8 vehicles
python -m convoy_commander run --scenario gps_denied --seed 42 --vehicles 8

# Run evaluation harness (5 scenarios x 3 seeds, produces summary.md)
python -m convoy_commander evaluate

# View last run results
python -m convoy_commander report --last

# Run unit tests (278 tests)
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
laptop.  Batch runs of all 10 scenarios take 4–8 minutes.

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
python -m pytest -q --tb=short     # 278 tests should pass
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

### Docker (Recommended for Local Testing)

```bash
# Clone and checkout
git clone https://github.com/jzbriggs22/Autonomous-Tactical-Convoy-Commander.git
cd Autonomous-Tactical-Convoy-Commander

# Build the image
docker build -t convoy-commander .

# Run all tests
docker run --rm convoy-commander

# Run tests with coverage
docker run --rm convoy-commander pytest --cov=convoy_commander --cov-report=term-missing -q

# Run a simulation (mount volume to get output)
# Linux / macOS:
docker run --rm -v $(pwd)/runs:/app/runs \
    convoy-commander convoy_commander run --scenario platooning --seed 42 --duration 60
# Windows CMD:
docker run --rm -v %cd%\runs:/app/runs convoy-commander convoy_commander run --scenario platooning --seed 42 --duration 60
# Windows PowerShell:
docker run --rm -v ${PWD}/runs:/app/runs convoy-commander convoy_commander run --scenario platooning --seed 42 --duration 60

# Run evaluation harness
docker run --rm -v $(pwd)/eval_results:/app/eval_results \
    convoy-commander convoy_commander evaluate

# Type check
docker run --rm convoy-commander mypy convoy_commander/ --ignore-missing-imports
```

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
| Parallel batch (10 scenarios) | `c5.2xlarge` | 8 | 16 GB | ~$0.34/hr |
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

#### Parallel Batch on c5.2xlarge (10 scenarios × 1 seed)

```bash
# Install GNU parallel
sudo dnf install -y parallel          # Amazon Linux
# or: sudo apt install -y parallel    # Ubuntu

scenarios=(baseline gps_denied comms_degraded leader_failure obstacle_pop \
           gps_spoofed silent_running comms_blackout sensor_drift_spike platooning mesh_relay)

parallel -j 8 python -m convoy_commander run \
    --scenario {} --seed 42 --vehicles 8 \
    --output /tmp/runs/{}_42 ::: "${scenarios[@]}"
```

Expected wall-clock time on `c5.2xlarge`: ~2–4 minutes for all 11 scenarios in parallel.

#### Storage Estimates

| Artefact | Per-run Size |
|----------|-------------|
| `report.md` | ~15 KB |
| `metrics.json` | ~2 KB |
| `time_series.jsonl` | ~5–20 MB (300 s, 8 vehicles) |
| `event_log.jsonl` | ~100–500 KB |
| `plots/` (9 PNGs) | ~5–10 MB |
| **Total per run** | **~8–25 MB** |

---

## CLI Reference

```
python -m convoy_commander run --scenario <name> [options]

Options:
  --scenario  Scenario name (baseline|gps_denied|comms_degraded|leader_failure|
              obstacle_pop|gps_spoofed|silent_running|comms_blackout|sensor_drift_spike|
              platooning|mesh_relay)
  --seed      Random seed (default: 42)
  --vehicles  Number of vehicles (default: 8)
  --loss      Packet loss rate 0-1 (overrides scenario default)
  --latency   Mean latency in ms (overrides scenario default)
  --duration  Simulation duration in seconds (overrides scenario default)
  --output    Output directory (default: runs/<scenario>_<timestamp>/)
```

```
python -m convoy_commander evaluate [options]

Options:
  --scenarios  Comma-separated scenario names (default: baseline,gps_denied,
               comms_degraded,leader_failure,comms_blackout,platooning,mesh_relay)
  --seeds      Comma-separated seeds (default: 42,123,7)
  --duration   Duration per run in seconds (default: 60)
  --vehicles   Number of vehicles (default: 8)
  --output     Output base directory (default: eval_results/)
```

```
python -m convoy_commander report --last      # display metrics from most recent run
python -m convoy_commander report --dir <path> # display metrics from specific run
```

Each run produces:
- `report.md` — markdown report with metrics, safety audit, performance notes, and model assumptions
- `config.json` — reproducibility stamp (git hash, python version, platform, full config)
- `plots/` — trajectory maps, position error, speed/fuel/uncertainty/headway timelines, comms graph, headway gaps, string stability, corridor adherence, error ellipses, network topology
- `metrics.json` — machine-readable metrics
- `time_series.jsonl` — per-vehicle per-timestep data
- `event_log.jsonl` — structured safety audit trail (every event with timestamp, severity, vehicle ID)

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Install package with dev dependencies |
| `make test` | Run all 278 tests |
| `make demo` | Run baseline scenario (60s) |
| `make evaluate` | Run evaluation harness (7 scenarios x 3 seeds) |
| `make sweep` | Run all 11 scenarios sequentially (60s each) |
| `make typecheck` | Run mypy type checker |
| `make docker` | Build and run tests in Docker |
| `make clean` | Remove caches |

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
| `platooning` | Actuator lag (0.2s), time headway (1.5s), corridor (25m), elevated IMU noise; leader brakes at t=40s for string stability |
| `mesh_relay` | Halved comms range (100m), 2-hop multi-hop relay enabled (10% per-hop loss); tests mesh networking |

## Models and Assumptions

### Vehicle Dynamics
- 2D kinematic model (bicycle-like): position, heading, speed
- Constrained max speed (12 m/s), acceleration (2 m/s²), deceleration (4 m/s²), turn rate (0.6 rad/s)
- Fuel model: idle consumption + speed-proportional consumption
- Speed is non-negative; no reverse motion

### Position Estimation (2×2 Covariance EKF)
- Dead-reckoning propagation with additive drift noise and a slow bias random walk
- **2×2 position covariance matrix** tracks full error ellipse (along-track vs cross-track uncertainty)
- Kalman-style measurement updates for GPS and landmark fixes (Joseph-form covariance update)
- Scalar `uncertainty` retained as sqrt(trace(P)/2) for backward-compatible safe-mode logic
- GPS fix (std 0.5m) available in baseline; landmark fixes (std 1.0m, range 50m) always available
- When uncertainty exceeds threshold (default 15m), vehicle enters **safe mode**
- Fix functions return innovation magnitude for anomaly detection
- `cov_eigenvalues` property provides major/minor axis variances for error-ellipse visualization

### Communications
- Ad-hoc network: adjacency determined by Euclidean distance vs max range (200m default)
- Packet loss: base rate + distance-squared degradation + blackout region multiplier
- Latency: Gaussian-distributed, floor at 1ms
- Message types: state broadcast, intent, hazard alert, leader election, heartbeat, waypoint bid
- **Multi-hop relay** (Phase 8): vehicles forward received broadcasts to out-of-range peers; configurable max hops (0–4) with per-hop loss penalty; neighbor state table caches latest broadcast per peer

### Coordination
- **Formation control**: consensus-based — each vehicle maintains spacing behind the leader with heading alignment from neighbor averaging
- **Leader election**: Bully algorithm — priority = fuel fraction; heartbeat timeout triggers election; highest priority wins with ID tie-breaking
- **Task allocation**: CBBA-lite auction (default) — consensus-based bundle algorithm assigns formation slot positions; re-auctioned every 10s; falls back to distributed greedy when `use_cbba=False`
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

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full diagram with trust boundaries and data flow.

```
convoy_commander/
  core/           World model (obstacles, road graph, spoof regions), physics,
                  config (pydantic with safety validators), event log
  vehicles/       Vehicle dynamics, fuel, position estimator (with innovation gate),
                  CommsMode enum
  comms/          Network model (range, loss, latency, blackout regions), messages
  planning/       Global planner (multi-objective A* on road graph),
                  local planner (DWA-lite with potential-field fallback)
  coordination/   Formation control, Bully leader election, CBBA-lite auction,
                  waypoint allocation
  supervisor/     Centralised supervisor agent (fleet-level anomaly detection)
  sim/            Simulation runner loop, 11 scenario definitions
  metrics/        Per-step collection, final aggregation, JSON export
  viz/            Matplotlib plots (9 types), markdown report with safety audit section
  evaluate.py     Batch evaluation harness (multi-scenario x multi-seed)
  stamp.py        Reproducibility metadata (git hash, python, platform)
  cli.py          CLI entry point (run, evaluate, report, test)
tests/            278 tests: physics, estimator, comms, planning, election,
                  sim, safety (30), Phase 2 (38), Phase 3 (33), Phase 4 (33),
                  Phase 5 (20), Phase 6 (30), Phase 7 (26), Phase 8 (36)
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

**Phase 4 — Implemented:**
- **2×2 Position Covariance (EKF upgrade)**: `EstimatorState.cov` is a 2×2 numpy array; propagation applies heading-dependent process noise Q; measurement updates use Joseph-form Kalman update preserving positive semi-definiteness; scalar `uncertainty` retained as sqrt(trace(P)/2) for backward compat; `cov_eigenvalues` property for error-ellipse visualisation
- **CBBA-lite Auction**: `cbba_allocate()` in `coordination/cbba.py` runs a consensus-based bundle algorithm where vehicles bid on formation slot positions based on proximity + fuel bonus; winner-takes-all with ID tie-breaking; re-auctioned every 10s in the sim loop; enabled by default (`use_cbba=True`); falls back to greedy index-based allocation when disabled
- **Enhanced Visualization**: `plot_world()` now renders rectangular obstacles (dimgray rectangles) and GPS spoof regions (translucent magenta circles) on all trajectory and comms graph plots

**Phase 5 — Implemented:**
- **Evaluation harness**: `evaluate` CLI command runs 6 scenarios x 3 seeds = 18 runs, produces `summary.md` with aggregate metrics table and per-scenario averages, plus per-run reports with full details
- **Reproducibility stamp**: every run records git commit hash, Python version, platform, and full resolved config in `config.json`; displayed in report's "Reproducibility" section
- **`report --last`**: displays key metrics from the most recent run directory (reads saved `metrics.json`)
- **Performance notes**: each report includes complexity analysis (O(N^2) collision detection, O(E log V) planning, O(N*S) CBBA)
- **GitHub Actions CI**: tests on Python 3.11 and 3.12 with pytest + mypy
- **Makefile**: `make install`, `make test`, `make demo`, `make evaluate`, `make sweep`, `make typecheck`, `make clean`
- **Architecture documentation**: [ARCHITECTURE.md](ARCHITECTURE.md) with module diagram, trust boundaries, data flow, and determinism guarantees

**Phase 6 — Implemented:**
- **Spatial hashing**: `SpatialHash` grid replaces O(N²) pairwise collision and comms checks with O(N*k); two grids (collision + comms) rebuilt each step
- **Road-corridor adherence**: DWA scores 4th component (corridor penalty, weight 0.25); measured against planned route polyline with ±3 segment window (O(1) per query); hard reject at 1.5× corridor width
- **Constant time headway**: `gap = standoff + time_headway × follower_speed`, capped at formation_spacing; replaces fixed-distance formation model
- **String stability metric**: RMS-based ratio computed during disturbance window; `platooning` scenario injects leader brake at t=40s
- **Actuator lag**: dead-time buffer with hold-last policy (not coast-to-zero); configurable 0–1s
- **Gauss-Markov + ARW/RRW IMU noise**: first-order Gauss-Markov position bias (exact discrete: σ_drive = σ_ss√(1−decay²)), heading bias random walk, angle random walk; process noise Q includes heading-induced position uncertainty
- **Platooning scenario**: combines actuator lag (0.2s), time headway (1.5s), corridor (25m), elevated IMU noise; leader speed perturbation for string stability test

**Phase 7 — Implemented:**
- **Platooning in default evaluation**: `platooning` added to `DEFAULT_SCENARIOS` (6 scenarios × 3 seeds = 18 runs)
- **Headway gap visualization**: `plot_headway_gaps` shows actual vs desired inter-vehicle gaps over time
- **String stability visualization**: `plot_string_stability` shows spacing errors and RMS per vehicle during disturbance window
- **Corridor adherence visualization**: `plot_corridor_adherence` shows trajectory + corridor distance over time
- **Metrics summary 4-subplot**: headway gap added as 4th subplot in `plot_metrics_summary`
- **Updated assumptions**: Phase 6 realism features (Gauss-Markov, CTH, actuator lag, corridor) reflected in model assumptions section
- **CI coverage reporting**: `pytest-cov` with term-missing and XML output in GitHub Actions
- **Per-step data collection**: `MetricsCollector` records headway_samples, corridor_samples, and spacing_error_samples per step

**Phase 8 — Implemented:**
- **Multi-hop message relay**: `CommsNetwork.relay_broadcast()` forwards messages through intermediate vehicles; configurable max hops (0–4) with per-hop loss penalty; `mesh_relay` scenario halves range to 100m and enables 2-hop relay
- **Neighbor state tables**: each `Vehicle` caches the latest `STATE_BROADCAST` per peer in `neighbor_states` dict; stale entries pruned automatically (3× broadcast interval)
- **Error-ellipse visualization**: `plot_error_ellipses` overlays 2×2 covariance ellipses (95% confidence) on trajectory plots; eigenvalues and rotation angle recorded every 100 steps
- **Network topology monitoring**: `plot_network_topology` shows avg/min degree and partition count over time; `get_network_stats()` computes connected components and multi-hop reachability
- **`mesh_relay` scenario**: reduced comms range (100m), 2-hop relay, 10% per-hop loss — tests mesh networking resilience
- **Evaluate updated**: 7 default scenarios (added mesh_relay); 9 plot types per run

**Remaining limitations:**
- 2D only — no terrain elevation or 3D dynamics
- DWA forward simulation uses point model (no swept volume)
- Comms model is distance-based with no frequency / bandwidth modeling
- Multi-hop relay is store-and-forward with no routing protocol (flood-based)
- IMU drift model is Gaussian (real drift is heavier-tailed)
- CBBA consensus is simulated centrally (not decentralised over the comms channel)

**Future extensions:**
- Terrain and elevation modeling
- Pareto frontier visualisation for multi-objective trade-offs
- Decentralised CBBA consensus via actual message passing
- Routing protocol for multi-hop (AODV or similar instead of flood)
