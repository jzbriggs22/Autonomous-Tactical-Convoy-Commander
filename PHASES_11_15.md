# Autonomous Tactical Convoy Commander — Development Phases 6–10

## Current State Summary

The project has completed 10 implementation phases covering:
- Core simulation engine with vehicle physics, comms, and event logging
- GPS denial/spoofing with EKF estimation and innovation gating
- Multi-objective route planning with DWA local planner
- CBBA-lite task auction and formation coordination
- Evaluation harness with reproducible seeded runs
- Realism upgrades: actuator lag, platooning, structured IMU noise
- Visualization: trajectory plots, error ellipses, markdown reports
- Multi-hop mesh relay networking
- Geospatial terrain (OSM roads, DEM elevation, slope-aware routing)
- Weather API integration with friction, wind, and visibility effects

**Test suite**: 300+ tests across 16 test files
**Scenarios**: 15 built-in operational scenarios
**Dependencies**: numpy, networkx, matplotlib, pydantic, scipy + optional geo/weather extras

---

## Phase 11 — Threat Modeling & Electronic Warfare

**Goal**: Simulate adversarial RF environments — jamming, spoofing, and electronic countermeasures that force the convoy to adapt in real time.

### New Module: `convoy_commander/threats/`

| File | Purpose |
|------|---------|
| `__init__.py` | Export threat models |
| `jammer.py` | `RFJammer` class: position, radius, power, frequency band, duty cycle |
| `threat_map.py` | `ThreatMap`: spatial index of active threats; query by position |
| `countermeasures.py` | `ECM` class: frequency hopping, power control, route avoidance |

### Key Features

1. **RF Jammer Model**
   - Configurable jammers placed in the world (static or mobile)
   - Signal-to-noise ratio (SNR) calculation at each vehicle position
   - When SNR drops below threshold: comms degraded proportionally (packet loss scales with jamming power)
   - GPS jamming variant: jammer suppresses GPS fixes within radius

2. **Threat Detection**
   - Vehicles detect jammers via received signal strength indicator (RSSI) anomaly detection
   - Collaborative threat localization: vehicles share bearing estimates to triangulate jammer position
   - Event log entries: `JAMMER_DETECTED`, `JAMMER_LOCALIZED`, `ECM_ACTIVATED`

3. **Electronic Countermeasures**
   - Frequency hopping: reduce jammer effectiveness by 60-80% (configurable)
   - Adaptive power control: increase TX power near jammer (trades range for reliability)
   - Route avoidance: planner adds high-cost zones around detected jammers

4. **New Scenarios**
   - `jammed_corridor`: static jammer blocking a chokepoint; convoy must detect, localize, and route around
   - `mobile_jammer`: moving jammer that follows the convoy; requires continuous ECM adaptation
   - `multi_threat`: combined GPS spoofing + RF jamming + comms blackout zones

### Modified Files

| File | Changes |
|------|---------|
| `core/config.py` | Add `ThreatConfig` with jammer count, power, ECM settings |
| `core/event_log.py` | Add `JAMMER_DETECTED`, `JAMMER_LOCALIZED`, `ECM_ACTIVATED` event kinds |
| `core/world.py` | Integrate `ThreatMap` into world generation |
| `comms/network.py` | SNR-based packet loss modifier from jammer proximity |
| `sim/runner.py` | Threat tick in simulation loop; ECM state machine |
| `sim/scenarios.py` | Register 3 new threat scenarios |
| `planning/global_planner.py` | Threat-avoidance cost layer in A* |
| `metrics/collector.py` | Track jammer detections, ECM activations, threat exposure time |
| `viz/report.py` | Threat map overlay in report; ECM timeline |

### Tests: `tests/test_threats.py`
- Jammer SNR attenuation at distance
- Threat detection via RSSI anomaly
- Collaborative triangulation accuracy
- Frequency hopping effectiveness
- Route avoidance around detected jammer
- Integration: `jammed_corridor` scenario completes with ECM enabled

---

## Phase 12 — Multi-Convoy Operations & Inter-Convoy Coordination

**Goal**: Scale from a single convoy to multiple independent convoys that share a road network, negotiate right-of-way, and can merge/split dynamically.

### New Module: `convoy_commander/multi_convoy/`

| File | Purpose |
|------|---------|
| `__init__.py` | Export multi-convoy orchestrator |
| `orchestrator.py` | `ConvoyOrchestrator`: manages N independent convoys sharing one world |
| `negotiation.py` | `RightOfWay` protocol: priority-based intersection negotiation |
| `merge_split.py` | `MergeSplit` controller: dynamic convoy join/leave operations |

### Key Features

1. **Multi-Convoy World**
   - `ConvoyOrchestrator` runs N `SimRunner` instances sharing one `World` and `CommsNetwork`
   - Each convoy has its own leader election, formation, and route plan
   - Inter-convoy comms use a separate "long-range" channel with higher latency

2. **Intersection Negotiation**
   - When two convoy routes cross, a `RightOfWay` protocol engages
   - Priority rules: mission priority > convoy size > first-arrival
   - Yielding convoy slows/holds at a safe standoff distance
   - Deadlock detection: timeout triggers one convoy to reroute

3. **Dynamic Merge/Split**
   - `MERGE` command: trailing convoy accelerates to join lead convoy's formation tail
   - `SPLIT` command: designated vehicles peel off and form a new convoy with independent routing
   - Split vehicles elect a new leader and recompute their route
   - Merge handshake: formation slot reallocation via CBBA re-auction

4. **New Scenarios**
   - `two_convoy_crossing`: two convoys on intersecting routes; tests negotiation
   - `convoy_merge`: trailing convoy catches up and merges
   - `convoy_split_reroute`: convoy splits at a fork; each sub-convoy routes independently
   - `multi_convoy_contested`: three convoys + jamming + weather; stress test

### Modified Files

| File | Changes |
|------|---------|
| `core/config.py` | Add `MultiConvoyConfig` with convoy count, priority levels, merge/split rules |
| `core/event_log.py` | Add `CONVOY_MERGE`, `CONVOY_SPLIT`, `RIGHT_OF_WAY`, `DEADLOCK` events |
| `coordination/leader_election.py` | Support re-election after merge/split |
| `coordination/cbba.py` | Slot reallocation for merged formation |
| `cli.py` | `--convoys N` flag, `--convoy-priority` |
| `metrics/collector.py` | Per-convoy metrics + inter-convoy events |
| `viz/report.py` | Multi-convoy trajectory overlay; merge/split timeline |

### Tests: `tests/test_multi_convoy.py`
- Two convoys navigate without collision
- Right-of-way protocol resolves intersection
- Merge produces valid single-file formation
- Split produces two independently-routing convoys
- Deadlock detection triggers reroute within timeout
- Metrics track per-convoy and aggregate stats

---

## Phase 13 — Machine Learning Integration & Adaptive Control

**Goal**: Replace hand-tuned control parameters with learned policies — a neural network speed/spacing controller trained via reinforcement learning on simulation data.

### New Module: `convoy_commander/ml/`

| File | Purpose |
|------|---------|
| `__init__.py` | Export ML components |
| `compat.py` | Lazy import guard for `torch` (follows existing pattern) |
| `observation.py` | `ObservationBuilder`: converts vehicle state to fixed-size feature vector |
| `policy.py` | `ConvoyPolicy`: small MLP that outputs (target_speed, target_spacing) |
| `trainer.py` | `PolicyTrainer`: PPO training loop using sim environment |
| `gym_env.py` | `ConvoyEnv(gym.Env)`: OpenAI Gym-compatible wrapper around SimRunner |

### Key Features

1. **Observation Space** (per vehicle, ~20 features)
   - Own speed, heading, fuel level, estimator uncertainty
   - Relative distance/bearing/speed to leader and predecessor
   - Comms health: packets received in last 5s, relay hop count
   - Weather state: friction factor, visibility factor, wind speed
   - Threat exposure: nearest jammer distance, ECM active flag

2. **Action Space**
   - Continuous: `(target_speed_fraction [0,1], spacing_factor [0.5, 3.0])`
   - Mapped to vehicle commands via existing DWA planner

3. **Reward Function**
   - +1.0 per timestep vehicle is within formation tolerance
   - -5.0 per near miss, -50.0 per collision
   - -0.1 * |spacing_error| (proportional penalty for formation drift)
   - +2.0 bonus for reaching destination
   - -0.01 * fuel_consumed (efficiency incentive)

4. **Training Pipeline**
   - `ConvoyEnv` wraps SimRunner with configurable scenario randomization
   - PPO with GAE, trained for 1M steps across mixed scenarios
   - Checkpoint saving/loading; policy export to ONNX for deployment
   - Evaluation: compare learned policy vs hand-tuned PID on same seeds

5. **Hybrid Mode**
   - `--controller ml` CLI flag to use learned policy
   - `--controller pid` (default) for hand-tuned baseline
   - Safety override: if ML policy commands violate safe-mode constraints, PID takes over

### New Dependencies (optional)

```toml
ml = [
    "torch>=2.0",
    "gymnasium>=0.29",
]
```

### New Scenarios
- `ml_training`: fast 30s episodes, randomized weather/comms/threats
- `ml_eval`: deterministic 300s run comparing ML vs PID controller

### Tests: `tests/test_ml.py`
- Observation builder produces correct feature vector shape
- Policy forward pass returns valid action range
- Gym env reset/step/render cycle works
- Trained checkpoint loads and produces deterministic actions
- Safety override engages when ML policy exceeds safe-mode bounds

---

## Phase 14 — Real-Time 3D Visualization & Dashboard

**Goal**: Replace static matplotlib plots with a live 3D visualization and operational dashboard for monitoring convoy operations in real time.

### New Module: `convoy_commander/dashboard/`

| File | Purpose |
|------|---------|
| `__init__.py` | Export dashboard server |
| `compat.py` | Lazy import guard for `flask`, `socketio` |
| `server.py` | Flask + Socket.IO backend serving real-time sim state |
| `api.py` | REST endpoints: start/stop/pause sim, get metrics, get config |
| `static/index.html` | Single-page dashboard app |
| `static/js/app.js` | Three.js 3D scene + Chart.js metric panels |
| `static/js/convoy3d.js` | Vehicle models, trail rendering, threat overlays |
| `static/css/style.css` | Dashboard styling |

### Key Features

1. **3D Scene (Three.js)**
   - Top-down camera with zoom/pan/rotate; follow-vehicle mode
   - Vehicle models rendered as oriented boxes with color-coded status (normal/safe-mode/failed)
   - Road network as textured ground plane
   - Obstacles as semi-transparent red volumes
   - Weather overlay: rain particles, fog opacity, wind direction arrows
   - Threat overlay: jammer radiation circles with pulsing animation

2. **Metric Panels (Chart.js)**
   - Real-time line charts: convoy speed, spacing, fuel, uncertainty
   - Comms health bar: packet delivery ratio with color thresholds
   - Event timeline: scrolling log of safety events with severity coloring
   - Formation diagram: bird's-eye schematic showing slot assignments

3. **Control Panel**
   - Start / Pause / Resume / Stop buttons
   - Scenario selector dropdown
   - Speed slider: 0.25x to 10x simulation speed
   - Inject event buttons: trigger jammer, weather change, leader failure mid-run

4. **Architecture**
   - SimRunner emits state snapshots via callback every N steps
   - Server broadcasts snapshots via Socket.IO to connected browsers
   - Supports multiple simultaneous viewers
   - Replay mode: load a completed run's event log and replay as if live

### New Dependencies (optional)

```toml
dashboard = [
    "flask>=3.0",
    "flask-socketio>=5.3",
    "python-socketio>=5.10",
]
```

### CLI Integration
- `convoy-commander dashboard` — start dashboard server on localhost:5000
- `convoy-commander dashboard --replay <run-dir>` — replay a previous run
- `--headless` flag on `run` command streams to dashboard without blocking

### Tests: `tests/test_dashboard.py`
- Server starts and serves index.html
- REST API returns valid config/metrics JSON
- Socket.IO emits state snapshots on sim tick
- Replay mode loads event log and streams correctly
- Multiple concurrent viewers receive same data

---

## Phase 15 — Hardware-in-the-Loop (HIL) & ROS2 Bridge

**Goal**: Bridge the simulator to real robotic platforms via ROS2, enabling hardware-in-the-loop testing where real vehicles run the same control logic against simulated comms and world state.

### New Module: `convoy_commander/hil/`

| File | Purpose |
|------|---------|
| `__init__.py` | Export HIL bridge components |
| `compat.py` | Lazy import guard for `rclpy` |
| `ros_bridge.py` | `ROS2Bridge`: publishes sim state as ROS2 topics, subscribes to vehicle commands |
| `time_sync.py` | `TimeSynchronizer`: aligns sim clock with ROS2 wall clock |
| `vehicle_adapter.py` | `VehicleROSAdapter`: translates between sim `Vehicle` and ROS2 `nav_msgs/Odometry` |
| `launch/convoy_hil.launch.py` | ROS2 launch file for HIL setup |

### Key Features

1. **ROS2 Topic Interface**
   - Publish per-vehicle: `/convoy/vehicle_N/odom` (nav_msgs/Odometry), `/convoy/vehicle_N/status` (custom msg)
   - Publish global: `/convoy/world_state` (occupancy grid), `/convoy/threats` (marker array)
   - Subscribe per-vehicle: `/convoy/vehicle_N/cmd_vel` (geometry_msgs/Twist)
   - Subscribe global: `/convoy/mission_command` (custom srv)

2. **Mixed-Reality Mode**
   - Designate vehicles 0..K as "real" (commands come from ROS2 hardware)
   - Vehicles K+1..N remain simulated with internal control
   - Real and simulated vehicles interact through shared comms network and world

3. **Time Synchronization**
   - Sim publishes `/clock` topic for ROS2 time source
   - Configurable real-time factor: 1.0x for HIL, faster for regression
   - Graceful handling of hardware lag: sim pauses if real vehicle falls behind

4. **Gazebo Integration (Optional)**
   - World export to SDF format for Gazebo visualization
   - Obstacle and terrain mesh generation from sim world state
   - Sensor simulation: GPS, IMU, LiDAR point clouds via Gazebo plugins

5. **Message Definitions**

```
convoy_msgs/
  msg/
    VehicleStatus.msg    # id, mode, fuel, uncertainty, formation_slot
    ConvoyState.msg      # array of VehicleStatus + leader_id + formation_type
    ThreatAlert.msg      # type, position, radius, confidence
  srv/
    MissionCommand.srv   # command (START/STOP/REROUTE/MERGE/SPLIT), params
```

### New Dependencies (optional)

```toml
hil = [
    "rclpy",
    "geometry_msgs",
    "nav_msgs",
    "sensor_msgs",
    "std_msgs",
]
```

### CLI Integration
- `convoy-commander hil` — start HIL bridge with default ROS2 config
- `--real-vehicles 0,1` — designate vehicle IDs as hardware-controlled
- `--rtf 1.0` — real-time factor
- `--gazebo` — export world to Gazebo SDF and launch

### Tests: `tests/test_hil.py`
- ROS2 bridge publishes odometry at correct rate
- Command subscription drives simulated vehicle
- Mixed-reality: real + sim vehicles maintain formation
- Time sync pauses sim when hardware lags
- Mission command service triggers convoy operations

---

## Implementation Priority & Dependencies

```
Phase 11 (Threats/EW)
    │
    ├──► Phase 12 (Multi-Convoy)     ← depends on threat map for contested scenarios
    │        │
    │        └──► Phase 13 (ML)      ← multi-convoy env for training diversity
    │
    └──► Phase 14 (Dashboard)        ← can visualize threats from Phase 11
             │
             └──► Phase 15 (HIL)     ← dashboard serves as HIL monitoring UI
```

Phases 11 and 14 can be developed in parallel. Phase 13 benefits from 11+12 for training environment diversity. Phase 15 builds on the dashboard for monitoring.

## Estimated Test Counts

| Phase | New Tests | Cumulative |
|-------|-----------|------------|
| 11    | ~25       | ~325       |
| 12    | ~20       | ~345       |
| 13    | ~15       | ~360       |
| 14    | ~12       | ~372       |
| 15    | ~15       | ~387       |
