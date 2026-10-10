# Phase 2 Implementation Plan

## Overview

Phase 2 builds on the safety-hardened MVP to add the four extensions from the
original spec, plus targeted hardening of the thinnest MVP subsystems.  Every
addition follows the same safety-critical discipline: conservative defaults,
explicit assumptions, event-log coverage, and test-first development.

---

## 1. GPS Spoofing Model + Robust Filtering  (estimator, comms, sim)

### Problem
The estimator currently trusts every GPS/landmark fix.  A spoofed fix can
silently drag the position estimate off, violating the "fail-safe" property.

### Design
- **`SpoofRegion`** in `core/world.py` — circular zone where GPS fixes are
  offset by a configurable bias vector (direction + magnitude).
- **Estimator innovation gate** in `vehicles/estimator.py`:
  - Both `apply_gps_fix` and `apply_landmark_fix` already return the
    innovation (pre-update residual).  Add a configurable
    `innovation_threshold` (chi-squared-like gate).
  - If innovation > threshold × uncertainty → **reject the fix** and log
    `ESTIMATOR_FIX_REJECTED` at WARNING.
  - Track consecutive rejections; if > N → log CRITICAL, enter safe mode.
- **New config fields**: `EstimatorConfig.innovation_gate_sigma` (default 3.0),
  `SimConfig.spoof_regions` list.
- **New scenario**: `gps_spoofed` — GPS available but with a spoof region on
  the convoy's likely path.
- **Tests**: spoofed fix is rejected; non-spoofed fix is accepted; consecutive
  rejections trigger safe mode; deterministic replay.

### Files touched
`core/config.py`, `core/world.py`, `vehicles/estimator.py`, `sim/runner.py`,
`sim/scenarios.py`, `core/event_log.py` (new event kinds), `tests/test_estimator.py`,
new `tests/test_spoofing.py`.

---

## 2. Multi-Objective Optimization  (planning, coordination, metrics)

### Problem
The global planner uses A* with a single cost (weighted distance).  There is
no way to trade off time vs fuel vs risk.

### Design
- **Weighted-sum objective** in `planning/global_planner.py`:
  - Edge cost = `w_time * travel_time + w_fuel * fuel_cost + w_risk * risk_cost`
  - `travel_time` = edge_weight / expected_speed
  - `fuel_cost` = edge_weight * fuel_rate_per_speed
  - `risk_cost` = proximity to obstacles / no-go zones along the edge
    (precomputed during world generation).
- **`PlanningObjective`** dataclass with weights `(w_time, w_fuel, w_risk)`
  added to `SimConfig`.  Defaults: `(0.5, 0.3, 0.2)` — balanced, slightly
  time-preferring.
- **Per-edge risk annotation** on `World.road_graph`: during world generation,
  annotate each edge with a `risk` attribute (min clearance to nearest
  obstacle or no-go zone, inverted and normalised).
- **Report extension**: show the chosen objective weights and final
  Pareto-metric breakdown (time, fuel, risk components separately).
- **Tests**: planner avoids high-risk edge when w_risk is large; planner
  takes shortest path when w_time dominates; weights at construction are
  validated (non-negative, sum > 0).

### Files touched
`core/config.py`, `core/world.py`, `planning/global_planner.py`,
`sim/runner.py`, `viz/report.py`, new `tests/test_multi_objective.py`.

---

## 3. Behaviour Switching: Silent Running vs Chatty  (comms, coordination, sim)

### Problem
All vehicles broadcast at a fixed interval regardless of tactical context.
In some environments, minimising RF emissions is critical.

### Design
- **`CommsMode` enum**: `NORMAL`, `SILENT`, `CHATTY`.
  - `SILENT`: broadcast interval × 10, only heartbeats and hazards sent,
    state broadcasts suppressed.  Safe mode spacing enforced.
  - `CHATTY`: broadcast interval × 0.5, all message types sent.
  - `NORMAL`: current behaviour.
- **Per-vehicle comms mode** stored on `Vehicle`, changeable by the sim
  runner or via a new `COMMS_MODE_CHANGE` message type.
- **Sim runner integration**: `_broadcast_states` checks each vehicle's
  comms mode.  In SILENT, skip state broadcasts but still send heartbeats
  at the normal interval (leader must stay discoverable).
- **Config**: `SimConfig.default_comms_mode` (default NORMAL).
  New scenario `silent_running` sets mode to SILENT.
- **Event logging**: `COMMS_MODE_CHANGE` event whenever a vehicle switches.
- **Tests**: SILENT mode reduces message count; heartbeats still sent;
  mode switch logged.

### Files touched
`vehicles/vehicle.py`, `comms/messages.py`, `comms/network.py`,
`core/config.py`, `sim/runner.py`, `sim/scenarios.py`,
`core/event_log.py`, new `tests/test_comms_modes.py`.

---

## 4. Centralised Supervisor Agent  (new module `supervisor/`)

### Problem
There is no baseline to compare decentralised performance against.

### Design
- **New module `convoy_commander/supervisor/`** with `supervisor.py`.
- **`CentralSupervisor`** class:
  - Has perfect knowledge of all vehicle positions (ground truth).
  - Runs a single A* for the convoy waypoint sequence.
  - Assigns formation offsets centrally (no per-vehicle consensus).
  - Sends direct commands to each vehicle (no comms loss model — the
    point is to show the ideal-comms upper bound).
- **Integration**: `SimConfig.use_supervisor: bool = False`.  When True
  the runner delegates planning and formation to the supervisor instead
  of the decentralised modules.  Comms model still runs (for metrics)
  but is not used for coordination.
- **Report comparison**: when supervisor mode, report includes a
  "Centralised vs Decentralised" note.  Running both modes with the
  same seed and diffing the metrics.json gives a direct comparison.
- **Tests**: supervisor run completes; supervisor achieves lower
  cohesion score (tighter formation); supervisor has zero safe-mode
  activations when GPS is available.

### Files touched
New `convoy_commander/supervisor/__init__.py`, `supervisor/supervisor.py`.
`core/config.py`, `sim/runner.py`, `viz/report.py`,
new `tests/test_supervisor.py`.

---

## 5. MVP Hardening (targeted improvements to thin subsystems)

### 5a. Local Planner Upgrade: DWA-Lite

The current potential field planner can get stuck in local minima.  Replace
with a DWA-lite (Dynamic Window Approach):

- Sample (accel, turn_rate) pairs within the dynamic window.
- For each sample, forward-simulate one step.
- Score: `heading_to_goal * w1 + clearance * w2 + speed * w3`.
- Pick the highest-scoring command.
- Keeps the potential-field as a fallback if DWA sampling produces no
  admissible candidate (all samples collide).
- **Event log**: `PLANNER_FALLBACK` event when DWA falls back to
  potential field.

### 5b. Formation Robustness

- Handle the case where leader is None (no leader elected yet): vehicles
  fall back to waypoint-following only, no formation correction.
- Handle the case where formation_index exceeds number of operational
  vehicles (late joiner after breakdowns).
- Log `FORMATION_DEGRADED` when fewer than half of vehicles are in
  formation (pairwise spacing within 2× target).

### 5c. Comms Blackout Region as a Scenario

- New scenario `comms_blackout`: a circular blackout region placed on
  the convoy path at a known location.  Vehicles entering it lose comms
  and must rely on safe mode.  Tests: vehicles enter safe mode inside
  blackout and resume normal operations after exiting.

---

## Implementation Order

| Step | Item | Depends On | Est. Files |
|------|------|------------|------------|
| 1 | GPS spoofing + innovation gating | — | 8 |
| 2 | DWA-lite local planner | — | 3 |
| 3 | Multi-objective planning | world risk annotation | 6 |
| 4 | Behaviour switching (silent/chatty) | — | 8 |
| 5 | Formation robustness fixes | — | 3 |
| 6 | Comms blackout scenario | — | 3 |
| 7 | Centralised supervisor | — | 6 |
| 8 | Tests + report updates for all | 1-7 | 7 |
| 9 | Full regression run (all scenarios) | 8 | — |

Steps 1, 2, 4, 5 are independent and can be developed in parallel.
Step 3 depends on world risk annotation (part of step 3 itself).
Step 7 is self-contained.
Step 8–9 are integration and validation.

---

## Safety Discipline (applies to every item above)

- **Config validation**: every new config field has bounds (`ge`, `gt`, `le`)
  and cross-field validators where applicable.
- **Event logging**: every new state transition, rejection, or fallback
  produces an event at the appropriate severity.
- **Assumption docstrings**: every new module/function documents its
  assumptions with the `SAFETY-CRITICAL ASSUMPTIONS:` prefix.
- **Pre/post-conditions**: new functions validate inputs and enforce
  output invariants.
- **Tests**: each feature adds ≥ 5 targeted tests including at least one
  determinism test and one edge-case / failure-mode test.
