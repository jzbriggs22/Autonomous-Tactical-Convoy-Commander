# Architecture & Trust Boundaries

## System Overview

```
                          CLI / Evaluate
                              |
                         SimRunner (main loop)
                              |
          +-------------------+-------------------+
          |                   |                   |
     Vehicles            Coordination          World
     (N agents)          & Comms              (environment)
          |                   |                   |
    +-----+-----+     +------+------+      +-----+-----+
    |     |     |     |      |      |      |     |     |
  State  Est.  Fuel  Leader CBBA  Comms   Road  Obs  Land-
  (kin.) (EKF) (mgr) Elect. Lite  Net    Graph  tacles marks
    |           |                   |
  Local       Safety            Messages
  Planner     Envelope          (typed)
    |
  Global
  Planner
  (A* / Dijkstra)
```

## Module Responsibilities

| Module | Path | Role |
|--------|------|------|
| **CLI** | `convoy_commander/cli.py` | Argument parsing, command dispatch |
| **Evaluate** | `convoy_commander/evaluate.py` | Batch sweep: N scenarios x M seeds |
| **SimRunner** | `convoy_commander/sim/runner.py` | Main loop orchestration, safety envelope |
| **Scenarios** | `convoy_commander/sim/scenarios.py` | Pre-configured operational contexts |
| **Config** | `convoy_commander/core/config.py` | Pydantic models with safety validators |
| **World** | `convoy_commander/core/world.py` | Environment: roads, obstacles, landmarks, spoof zones |
| **Physics** | `convoy_commander/core/physics.py` | Kinematics, Euler integration |
| **Vehicle** | `convoy_commander/vehicles/vehicle.py` | Per-agent state, fuel, status |
| **Estimator** | `convoy_commander/vehicles/estimator.py` | 2x2 covariance EKF, innovation gating |
| **CommsNetwork** | `convoy_commander/comms/network.py` | LOS propagation, loss, latency, blackout zones |
| **Messages** | `convoy_commander/comms/messages.py` | Typed messages: heartbeat, election, hazard, state |
| **LeaderElection** | `convoy_commander/coordination/leader_election.py` | Bully algorithm |
| **CBBA** | `convoy_commander/coordination/cbba.py` | Consensus-based formation slot auction |
| **Formation** | `convoy_commander/coordination/formation.py` | Spacing correction, heading consensus |
| **GlobalPlanner** | `convoy_commander/planning/global_planner.py` | Multi-objective A* on road graph |
| **LocalPlanner** | `convoy_commander/planning/local_planner.py` | DWA + potential fields |
| **Supervisor** | `convoy_commander/supervisor/supervisor.py` | Centralised anomaly detector (optional) |
| **Metrics** | `convoy_commander/metrics/collector.py` | Per-step data, aggregate computation |
| **Report** | `convoy_commander/viz/report.py` | Markdown + plots + JSON artifacts |
| **Stamp** | `convoy_commander/stamp.py` | Reproducibility metadata (git, python, platform) |
| **EventLog** | `convoy_commander/core/event_log.py` | Structured safety audit trail |

## Trust Boundaries

### Safety-Critical (must be correct for safe operation)

These modules enforce hard invariants. A bug here could cause the simulated
convoy to behave unsafely:

1. **Safety Envelope** (`runner.py:_enforce_safety_envelope`) — Post-step
   invariant checks: boundary clamping, speed caps, obstacle collision
   detection. These run *after* every control step and override planner
   commands.

2. **Config Validators** (`config.py`) — Pydantic validators that reject
   configurations violating safety constraints (e.g., collision_radius <
   min_separation < formation_spacing).

3. **Innovation Gate** (`estimator.py:apply_gps_fix`, `apply_landmark_fix`)
   — Rejects measurement updates whose innovation exceeds a threshold.
   Primary defence against GPS spoofing.

4. **Safe Mode Policy** (`runner.py:_check_safe_mode`) — Conservative
   entry (any single trigger), exit only when all conditions clear. Reduces
   speed and increases spacing.

5. **Collision Detection** (`runner.py:_detect_collisions`) — Pairwise
   distance checks every step. Logged at CRITICAL severity.

### Advisory (influence behaviour but don't override safety)

These modules suggest actions but can be overridden by the safety envelope:

- **Local Planner** — DWA commands are subject to speed caps and obstacle checks
- **CBBA Auction** — Formation slots are suggestions; safety mode takes priority
- **Supervisor** — Replan advisories are non-binding
- **Leader Election** — Affects formation but not individual vehicle safety

### Untrusted Inputs (validated before use)

- **GPS measurements** — Potentially spoofed; gated by innovation filter
- **Landmark measurements** — Noisy; gated by innovation filter
- **Comms messages** — Subject to loss and latency; timeouts trigger safe mode

## Data Flow

```
Sensor Layer           Estimation           Decision           Actuation
                                            Layer              Layer
+-----------+       +-------------+      +-----------+      +---------+
| GPS/IMU   | ----> | EKF         | ---> | Global    | ---> | Speed   |
| Landmarks |       | (estimator) |      | Planner   |      | Heading |
+-----------+       +-----+-------+      +-----+-----+      +----+----+
                          |                    |                   |
                          v                    v                   v
                    +-----------+        +-----------+      +-----------+
                    | Covariance|        | Local     |      | Physics   |
                    | (2x2 P)  |        | Planner   |      | (Euler)   |
                    +-----------+        | (DWA)     |      +-----------+
                          |              +-----------+            |
                          v                    |                  v
                    +-----------+              |           +-----------+
                    | Safe Mode | <------------+           | Safety    |
                    | Decision  |                          | Envelope  |
                    +-----------+                          | (post-    |
                                                          |  step)    |
                                                          +-----------+
```

## Event Logging Architecture

Every safety-relevant state transition is logged to the EventLog with:
- **Timestamp** (monotonic within run)
- **Kind** (COLLISION, NEAR_MISS, SAFE_MODE_ENTER, GPS_SPOOFED, etc.)
- **Severity** (DEBUG, INFO, WARNING, CRITICAL)
- **Vehicle ID** (if applicable)
- **Message** (human-readable)
- **Details** (structured kwargs for machine processing)

The log is written to `event_log.jsonl` and surfaced in the safety audit
section of `report.md`.

## Determinism

Given the same seed, the simulation produces bit-identical results:
- `np.random.default_rng(seed)` for all stochastic components
- Per-vehicle RNG seeded as `seed + vehicle_id + 1`
- Fixed-timestep integration (no adaptive stepping)
- Deterministic leader election tie-breaking (highest ID wins)
