"""Main simulation runner.

SAFETY-CRITICAL DESIGN:
  - Every safety-relevant state transition is logged to the EventLog.
  - The sim loop enforces hard invariants *after* each step (boundary
    clamping, speed capping, collision detection).  If an invariant is
    violated the event is logged and corrective action is taken
    (e.g., emergency stop).
  - The runner never silently swallows errors: unexpected states are
    logged at CRITICAL severity so they surface in the audit report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from convoy_commander.comms.messages import (
    Message,
    MessageType,
    make_hazard,
    make_state_broadcast,
)
from convoy_commander.comms.network import CommsNetwork
from convoy_commander.coordination.allocation import allocate_waypoints_greedy
from convoy_commander.coordination.cbba import cbba_allocate, compute_formation_slots
from convoy_commander.coordination.formation import (
    compute_formation_correction,
    get_formation_index,
)
from convoy_commander.coordination.leader_election import LeaderElection
from convoy_commander.core.config import SimConfig
from convoy_commander.core.event_log import EventKind, EventLog, Severity
from convoy_commander.core.world import World
from convoy_commander.metrics.collector import MetricsCollector
from convoy_commander.planning.global_planner import plan_route
from convoy_commander.stamp import ReproStamp, collect_stamp
from convoy_commander.planning.local_planner import compute_command
from convoy_commander.vehicles.vehicle import CommsMode, Vehicle, VehicleStatus


@dataclass
class SimResult:
    """Result of a simulation run."""

    config: SimConfig
    vehicles: list[Vehicle]
    world: World
    collector: MetricsCollector
    comms: CommsNetwork
    event_log: EventLog
    stamp: ReproStamp | None = None


class SimRunner:
    """Orchestrates the simulation loop."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.world = World(config.world, self.rng)
        self.comms = CommsNetwork(config.comms, self.rng)
        self.collector = MetricsCollector()
        self.event_log = EventLog()

        # --- Log simulation start with full config ---
        self.event_log.log(
            0.0, EventKind.SIM_START, Severity.INFO,
            message=f"Simulation starting: scenario={config.scenario} seed={config.seed} "
                    f"vehicles={config.num_vehicles} duration={config.duration}s",
            scenario=config.scenario, seed=config.seed, num_vehicles=config.num_vehicles,
        )

        # --- Log config validation warnings ---
        for warning in config.safety_warnings():
            self.event_log.log(
                0.0, EventKind.CONFIG_WARNING, Severity.WARNING,
                message=warning,
            )

        # Create vehicles in a staggered formation near bottom-left
        self.vehicles: list[Vehicle] = []
        for i in range(config.num_vehicles):
            row = i // 2
            col = i % 2
            start_x = 50.0 + col * 25.0
            start_y = 50.0 + row * 25.0
            v = Vehicle(
                vehicle_id=i,
                config=config,
                rng=np.random.default_rng(config.seed + i + 1),
                start_x=start_x,
                start_y=start_y,
                start_heading=0.3,  # roughly northeast
            )
            self.vehicles.append(v)
            self.event_log.log(
                0.0, EventKind.VEHICLE_SPAWNED, Severity.INFO, vehicle_id=i,
                message=f"Vehicle {i} spawned at ({start_x:.1f}, {start_y:.1f})",
                x=start_x, y=start_y,
            )

        # Leader election per vehicle
        self.elections: dict[int, LeaderElection] = {}
        for v in self.vehicles:
            self.elections[v.id] = LeaderElection(
                v, config.coordination.leader_heartbeat_timeout
            )

        # Set initial leader (vehicle 0)
        self.vehicles[0].is_leader = True
        self.vehicles[0].leader_id = 0
        for v in self.vehicles:
            v.leader_id = 0
            self.elections[v.id].last_heartbeat_time = 0.0
        self.event_log.log(
            0.0, EventKind.LEADER_ELECTED, Severity.INFO, vehicle_id=0,
            message="Vehicle 0 designated as initial leader",
        )

        # Convoy destination: upper-right area
        self.destination = (
            config.world.width - 80.0,
            config.world.height - 80.0,
        )

        # Plan initial routes
        allocate_waypoints_greedy(self.vehicles, self.destination)
        self._plan_all_routes()

        # Scenario event flags
        self._leader_failed = False
        self._obstacle_popped = False
        self._drift_spike_applied = False

        # Per-vehicle comms-lost tracking for edge-detect logging
        self._comms_lost_flags: dict[int, bool] = {v.id: False for v in self.vehicles}
        # Per-vehicle fuel-low tracking
        self._fuel_low_logged: set[int] = set()

        # Comms blackout scenario: add blackout region to the comms network
        if config.scenario == "comms_blackout":
            mid_x = config.world.width * 0.5
            mid_y = config.world.height * 0.5
            self.comms.add_blackout_region(mid_x, mid_y, radius=120.0, loss_mult=20.0)
            self.event_log.log(
                0.0, EventKind.SCENARIO_EVENT, Severity.INFO,
                message=f"SCENARIO: Comms blackout zone at ({mid_x:.0f}, {mid_y:.0f}) radius=120m",
                x=mid_x, y=mid_y, radius=120.0,
            )

        # CBBA-based formation slot assignments  {vehicle_id: slot_index}
        self._cbba_slots: dict[int, int] = {}
        self._cbba_realloc_timer: float = 0.0
        _CBBA_REALLOC_INTERVAL = 10.0  # re-auction every 10s
        self._cbba_interval = _CBBA_REALLOC_INTERVAL
        if config.use_cbba:
            self._run_cbba_allocation()

        # Centralised supervisor (optional)
        self._supervisor = None
        if config.use_supervisor:
            from convoy_commander.supervisor.supervisor import CentralSupervisor
            self._supervisor = CentralSupervisor(config)

    def _run_cbba_allocation(self) -> None:
        """Run CBBA-lite auction to assign formation slot indices."""
        leader = self._get_leader()
        operational = [v for v in self.vehicles if v.is_operational]
        if not operational or leader is None:
            return

        slot_positions = compute_formation_slots(
            leader.estimator.state.x,
            leader.estimator.state.y,
            leader.estimator.state.heading,
            len(operational),
            self.config.coordination.formation_spacing,
        )

        self._cbba_slots = cbba_allocate(operational, slot_positions)

    def _plan_all_routes(self) -> None:
        """Plan global routes for all vehicles using the configured planning objective."""
        objective = self.config.planning
        for v in self.vehicles:
            if v.assigned_destination is not None and v.is_operational:
                route = plan_route(
                    self.world,
                    v.estimator.state.x,
                    v.estimator.state.y,
                    v.assigned_destination[0],
                    v.assigned_destination[1],
                    objective=objective,
                )
                v.waypoints = route
                v.current_waypoint_idx = 0

    def run(self, progress_callback: callable | None = None) -> SimResult:
        """Run the full simulation."""
        dt = self.config.dt
        total_steps = int(self.config.duration / dt)
        broadcast_timer = 0.0

        for step in range(total_steps):
            current_time = step * dt

            # Progress reporting
            if progress_callback and step % 100 == 0:
                progress_callback(step, total_steps)

            # === Scenario events ===
            self._handle_scenario_events(current_time)

            # === Position fixes (GPS / landmarks) ===
            self._apply_position_fixes(current_time)

            # === Comms: deliver pending messages ===
            self.comms.tick(current_time)

            # === Process received messages ===
            self._process_messages(current_time)

            # === Leader election ===
            self._run_elections(current_time)

            # === Broadcast state periodically ===
            broadcast_timer += dt
            if broadcast_timer >= self.config.comms.broadcast_interval:
                broadcast_timer = 0.0
                self._broadcast_states(current_time)

            # === Check safe mode conditions ===
            self._check_safe_mode(current_time)

            # === Centralised supervisor tick ===
            if self._supervisor is not None:
                self._run_supervisor(current_time)

            # === CBBA re-allocation (periodic) ===
            if self.config.use_cbba:
                self._cbba_realloc_timer += dt
                if self._cbba_realloc_timer >= self._cbba_interval:
                    self._cbba_realloc_timer = 0.0
                    self._run_cbba_allocation()

            # === Planning and control ===
            leader = self._get_leader()
            operational_ids = [v.id for v in self.vehicles if v.is_operational]

            # Log formation degraded if no leader
            if leader is None and len(operational_ids) > 0:
                if step % 100 == 0:  # Throttle: log every 10s
                    self.event_log.log(
                        current_time, EventKind.FORMATION_DEGRADED, Severity.WARNING,
                        message="No operational leader; convoy formation degraded",
                    )

            for v in self.vehicles:
                if not v.is_operational:
                    continue

                # Get current waypoint target
                target = self._get_current_target(v)
                if target is None:
                    continue

                # Formation correction — use CBBA slot if available
                if self.config.use_cbba and v.id in self._cbba_slots:
                    formation_idx = self._cbba_slots[v.id]
                else:
                    formation_idx = get_formation_index(v.id, leader.id if leader else None, operational_ids)
                if leader and not v.is_leader:
                    correction = compute_formation_correction(
                        v, leader,
                        [vv for vv in self.vehicles if vv.id != v.id and vv.is_operational],
                        formation_idx,
                        v.effective_spacing,
                    )
                    target = (target[0] + correction[0] * 5.0, target[1] + correction[1] * 5.0)

                # Compute and apply command
                neighbors = [vv for vv in self.vehicles if vv.id != v.id]
                cmd = compute_command(v, target, self.world, neighbors, dt)

                # Clamp speed in safe mode
                if v.status == VehicleStatus.SAFE_MODE:
                    max_safe = v.effective_max_speed
                    if v.state.speed > max_safe and cmd.accel > 0:
                        cmd.accel = -v.vcfg.max_decel * 0.3

                v.step(cmd, dt)

                # === Safety envelope enforcement (post-step) ===
                self._enforce_safety_envelope(v, current_time)

                # Check waypoint advance
                self._advance_waypoint(v)

                # Check arrival
                if v.has_reached_destination():
                    if v.status != VehicleStatus.ARRIVED:
                        v.status = VehicleStatus.ARRIVED
                        v.state.speed = 0.0
                        self.collector.record_arrival(v.id, current_time)
                        self.event_log.log(
                            current_time, EventKind.VEHICLE_ARRIVED, Severity.INFO,
                            vehicle_id=v.id,
                            message=f"Vehicle {v.id} arrived at destination",
                            x=v.state.x, y=v.state.y,
                        )

            # === Fuel monitoring ===
            self._check_fuel(current_time)

            # === Collision and near-miss detection ===
            self._detect_collisions(current_time)

            # === Record metrics ===
            self.collector.record_step(current_time, self.vehicles)

            # Record comms adjacency periodically
            if step % 50 == 0:
                positions = {v.id: (v.state.x, v.state.y) for v in self.vehicles}
                adj = self.comms.get_adjacency(positions)
                self.collector.record_comms_adjacency(current_time, adj)

            # Check if all arrived
            if all(
                v.has_reached_destination() or not v.is_operational
                for v in self.vehicles
            ):
                break

        # --- Log simulation end ---
        final_time = min(total_steps * dt, self.config.duration)
        arrived = sum(1 for v in self.vehicles if v.has_reached_destination())
        self.event_log.log(
            final_time, EventKind.SIM_END, Severity.INFO,
            message=f"Simulation ended: {arrived}/{len(self.vehicles)} arrived",
            arrived=arrived, total=len(self.vehicles),
        )

        stamp = collect_stamp(self.config)

        return SimResult(
            config=self.config,
            vehicles=self.vehicles,
            world=self.world,
            collector=self.collector,
            comms=self.comms,
            event_log=self.event_log,
            stamp=stamp,
        )

    # ------------------------------------------------------------------
    # Safety envelope enforcement
    # ------------------------------------------------------------------

    def _enforce_safety_envelope(self, v: Vehicle, t: float) -> None:
        """Post-step invariant checks and corrective actions."""
        # Boundary clamping
        clamped = False
        if v.state.x < 0:
            v.state.x = 0.0
            clamped = True
        elif v.state.x > self.world.width:
            v.state.x = self.world.width
            clamped = True
        if v.state.y < 0:
            v.state.y = 0.0
            clamped = True
        elif v.state.y > self.world.height:
            v.state.y = self.world.height
            clamped = True
        if clamped:
            v.state.speed = 0.0  # Emergency stop on boundary
            self.event_log.log(
                t, EventKind.BOUNDARY_VIOLATION, Severity.WARNING, vehicle_id=v.id,
                message=f"Vehicle {v.id} clamped to world boundary, emergency stop",
                x=v.state.x, y=v.state.y,
            )

        # Speed limit enforcement (belt-and-braces)
        hard_max = v.vcfg.max_speed * 1.01  # 1% tolerance for float rounding
        if v.state.speed > hard_max:
            self.event_log.log(
                t, EventKind.SPEED_LIMIT_EXCEEDED, Severity.WARNING, vehicle_id=v.id,
                message=f"Vehicle {v.id} speed {v.state.speed:.2f} exceeds max {v.vcfg.max_speed:.2f}",
                speed=v.state.speed, max_speed=v.vcfg.max_speed,
            )
            v.state.speed = v.vcfg.max_speed

        # Obstacle collision check (vehicle inside obstacle -> emergency stop)
        if self.world.is_blocked(v.state.x, v.state.y):
            v.state.speed = 0.0
            self.event_log.log(
                t, EventKind.INVARIANT_VIOLATION, Severity.CRITICAL, vehicle_id=v.id,
                message=f"Vehicle {v.id} inside obstacle at ({v.state.x:.1f}, {v.state.y:.1f}), "
                        "emergency stop",
                x=v.state.x, y=v.state.y,
            )

    def _check_fuel(self, t: float) -> None:
        """Monitor fuel levels and log warnings."""
        for v in self.vehicles:
            if not v.is_operational:
                continue
            if v.fuel.is_empty:
                v.status = VehicleStatus.BREAKDOWN
                v.state.speed = 0.0
                self.event_log.log(
                    t, EventKind.VEHICLE_FUEL_EMPTY, Severity.CRITICAL, vehicle_id=v.id,
                    message=f"Vehicle {v.id} fuel exhausted — forced breakdown",
                )
            elif v.fuel.is_low and v.id not in self._fuel_low_logged:
                self._fuel_low_logged.add(v.id)
                self.event_log.log(
                    t, EventKind.VEHICLE_FUEL_LOW, Severity.WARNING, vehicle_id=v.id,
                    message=f"Vehicle {v.id} fuel below 20% ({v.fuel.fraction:.0%})",
                    fuel_fraction=v.fuel.fraction,
                )

    # ------------------------------------------------------------------
    # Scenario events
    # ------------------------------------------------------------------

    def _handle_scenario_events(self, t: float) -> None:
        """Trigger scenario-specific events."""
        scenario = self.config.scenario

        if scenario == "leader_failure" and not self._leader_failed and t >= 120.0:
            self._leader_failed = True
            leader = self._get_leader()
            if leader:
                self.event_log.log(
                    t, EventKind.SCENARIO_EVENT, Severity.WARNING, vehicle_id=leader.id,
                    message=f"SCENARIO: Leader vehicle {leader.id} forced breakdown at t={t:.1f}s",
                )
                leader.set_breakdown()
                self.event_log.log(
                    t, EventKind.LEADER_LOST, Severity.WARNING,
                    message=f"Leader {leader.id} lost — triggering re-election",
                )
                # Force election restart on all
                for v in self.vehicles:
                    if v.is_operational:
                        self.elections[v.id].reset()
                self.collector.leader_elections += 1

        if scenario == "sensor_drift_spike" and not self._drift_spike_applied and t >= 60.0:
            self._drift_spike_applied = True
            spike_magnitude = 8.0
            self.event_log.log(
                t, EventKind.SCENARIO_EVENT, Severity.WARNING,
                message=f"SCENARIO: IMU drift spike injected (magnitude={spike_magnitude}m) "
                        f"to all {sum(1 for v in self.vehicles if v.is_operational)} operational vehicles",
                magnitude=spike_magnitude,
            )
            for v in self.vehicles:
                if v.is_operational:
                    v.estimator.apply_drift_spike(spike_magnitude)
                    self.event_log.log(
                        t, EventKind.DRIFT_SPIKE, Severity.WARNING, vehicle_id=v.id,
                        message=f"Vehicle {v.id} drift spike: bias jump ≈ {spike_magnitude}m, "
                                f"uncertainty now ≈ {v.estimator.state.uncertainty:.1f}m",
                        magnitude=spike_magnitude,
                        uncertainty_after=v.estimator.state.uncertainty,
                    )

        if scenario == "obstacle_pop" and not self._obstacle_popped and t >= 90.0:
            self._obstacle_popped = True
            mid_x = self.world.width * 0.5
            mid_y = self.world.height * 0.5
            self.event_log.log(
                t, EventKind.SCENARIO_EVENT, Severity.WARNING,
                message=f"SCENARIO: New obstacle at ({mid_x:.0f}, {mid_y:.0f}) radius 35m",
                x=mid_x, y=mid_y, radius=35.0,
            )
            self.world.add_obstacle(mid_x, mid_y, 35.0)
            self._plan_all_routes()
            self.event_log.log(
                t, EventKind.ROUTE_REPLAN, Severity.INFO,
                message="All routes replanned after obstacle insertion",
            )

    # ------------------------------------------------------------------
    # Position fixes
    # ------------------------------------------------------------------

    def _apply_position_fixes(self, t: float) -> None:
        """Apply GPS or landmark fixes to vehicles.

        GPS fixes are potentially spoofed if the vehicle is inside a SpoofRegion.
        The innovation gate in the estimator may reject anomalous fixes.
        """
        for v in self.vehicles:
            if not v.is_operational:
                continue

            # Check for GPS spoofing at vehicle's true position
            spoof_offset = self.world.get_spoof_offset(v.state.x, v.state.y)
            if spoof_offset is not None:
                self.event_log.log(
                    t, EventKind.GPS_SPOOFED, Severity.WARNING, vehicle_id=v.id,
                    message=f"Vehicle {v.id} in GPS spoof zone; offset=({spoof_offset[0]:.1f},"
                            f"{spoof_offset[1]:.1f})m",
                    offset_x=spoof_offset[0], offset_y=spoof_offset[1],
                )

            # GPS fix
            if self.config.gps_available:
                meas_x = v.state.x + (spoof_offset[0] if spoof_offset else 0.0)
                meas_y = v.state.y + (spoof_offset[1] if spoof_offset else 0.0)
                innovation, accepted = v.estimator.apply_gps_fix(meas_x, meas_y)
                if not accepted:
                    self.event_log.log(
                        t, EventKind.ESTIMATOR_FIX_REJECTED, Severity.WARNING, vehicle_id=v.id,
                        message=f"Vehicle {v.id} GPS fix rejected by innovation gate "
                                f"(innovation={innovation:.1f}m)",
                        innovation_m=innovation,
                    )
            elif self.config.gps_intermittent_prob > 0:
                if self.rng.random() < self.config.gps_intermittent_prob * self.config.dt:
                    meas_x = v.state.x + (spoof_offset[0] if spoof_offset else 0.0)
                    meas_y = v.state.y + (spoof_offset[1] if spoof_offset else 0.0)
                    innovation, accepted = v.estimator.apply_gps_fix(meas_x, meas_y)
                    if not accepted:
                        self.event_log.log(
                            t, EventKind.ESTIMATOR_FIX_REJECTED, Severity.WARNING, vehicle_id=v.id,
                            message=f"Vehicle {v.id} intermittent GPS fix rejected "
                                    f"(innovation={innovation:.1f}m)",
                            innovation_m=innovation,
                        )

            # Landmark fixes
            nearby = self.world.landmarks_in_range(v.state.x, v.state.y)
            for lm in nearby:
                innovation, accepted = v.estimator.apply_landmark_fix(v.state.x, v.state.y)
                if not accepted:
                    self.event_log.log(
                        t, EventKind.ESTIMATOR_FIX_REJECTED, Severity.WARNING, vehicle_id=v.id,
                        message=f"Vehicle {v.id} landmark fix rejected by innovation gate "
                                f"(innovation={innovation:.1f}m)",
                        innovation_m=innovation,
                    )

    # ------------------------------------------------------------------
    # Comms processing
    # ------------------------------------------------------------------

    def _process_messages(self, t: float) -> None:
        """Process incoming messages for all vehicles."""
        for v in self.vehicles:
            if not v.is_operational:
                continue
            inbox = self.comms.get_inbox(v.id)
            for msg in inbox:
                v.last_comms_time = t

                # Edge-detect comms restored
                if self._comms_lost_flags.get(v.id, False):
                    self._comms_lost_flags[v.id] = False
                    self.event_log.log(
                        t, EventKind.COMMS_RESTORED, Severity.INFO, vehicle_id=v.id,
                        message=f"Vehicle {v.id} comms restored",
                    )

                if msg.msg_type == MessageType.LEADER_HEARTBEAT:
                    self.elections[v.id].on_heartbeat(msg, t)
                elif msg.msg_type == MessageType.LEADER_ELECTION:
                    self.elections[v.id].on_election_message(msg, t)
                elif msg.msg_type == MessageType.HAZARD:
                    pass
                elif msg.msg_type == MessageType.STATE_BROADCAST:
                    pass

    def _run_elections(self, t: float) -> None:
        """Run leader election logic for all vehicles."""
        positions = {v.id: (v.state.x, v.state.y) for v in self.vehicles if v.is_operational}
        prev_leaders = {v.id for v in self.vehicles if v.is_leader}
        for v in self.vehicles:
            if not v.is_operational:
                continue
            msgs = self.elections[v.id].check_and_elect(t)
            sender_pos = (v.state.x, v.state.y)
            for msg in msgs:
                self.comms.send_broadcast(msg, sender_pos, positions, t)

        # Detect new leader
        for v in self.vehicles:
            if v.is_leader and v.id not in prev_leaders:
                self.event_log.log(
                    t, EventKind.LEADER_ELECTED, Severity.INFO, vehicle_id=v.id,
                    message=f"Vehicle {v.id} elected as new leader",
                )
                self.collector.leader_elections += 1

    def _broadcast_states(self, t: float) -> None:
        """Broadcast vehicle states, respecting each vehicle's CommsMode."""
        positions = {v.id: (v.state.x, v.state.y) for v in self.vehicles if v.is_operational}
        for v in self.vehicles:
            if not v.is_operational:
                continue
            # SILENT mode: suppress all state broadcasts (stealth / emissions control)
            if v.comms_mode == CommsMode.SILENT:
                continue
            msg = make_state_broadcast(
                sender_id=v.id,
                timestamp=t,
                x=v.estimator.state.x,
                y=v.estimator.state.y,
                heading=v.estimator.state.heading,
                speed=v.state.speed,
                uncertainty=v.estimator.state.uncertainty,
                status=v.status.name,
                fuel=v.fuel.fuel,
            )
            sender_pos = (v.state.x, v.state.y)
            self.comms.send_broadcast(msg, sender_pos, positions, t)

    # ------------------------------------------------------------------
    # Safe mode logic
    # ------------------------------------------------------------------

    def _check_safe_mode(self, t: float) -> None:
        """Enter/exit safe mode based on safety conditions.

        Conservative policy: enter on *any* trigger, exit only when
        *all* conditions clear.
        """
        comms_timeout = self.config.coordination.comms_lost_timeout

        for v in self.vehicles:
            if not v.is_operational:
                continue

            reasons: list[str] = []

            # High position uncertainty
            if v.estimator.is_uncertain:
                reasons.append(
                    f"uncertainty={v.estimator.state.uncertainty:.1f}m > "
                    f"threshold={self.config.estimator.uncertainty_safe_threshold:.1f}m"
                )

            # Comms lost for too long
            comms_gap = t - v.last_comms_time
            if comms_gap > comms_timeout and t > 5.0:
                reasons.append(f"comms_lost={comms_gap:.1f}s > timeout={comms_timeout:.1f}s")
                # Edge-detect comms lost
                if not self._comms_lost_flags.get(v.id, False):
                    self._comms_lost_flags[v.id] = True
                    self.event_log.log(
                        t, EventKind.COMMS_LOST, Severity.WARNING, vehicle_id=v.id,
                        message=f"Vehicle {v.id} comms lost for {comms_gap:.1f}s",
                    )

            if reasons and v.status == VehicleStatus.ACTIVE:
                v.enter_safe_mode()
                self.collector.safe_mode_activations += 1
                self.event_log.log(
                    t, EventKind.SAFE_MODE_ENTER, Severity.WARNING, vehicle_id=v.id,
                    message=f"Vehicle {v.id} entering safe mode: {'; '.join(reasons)}",
                )
            elif not reasons and v.status == VehicleStatus.SAFE_MODE:
                v.exit_safe_mode()
                self.event_log.log(
                    t, EventKind.SAFE_MODE_EXIT, Severity.INFO, vehicle_id=v.id,
                    message=f"Vehicle {v.id} exiting safe mode — all conditions clear",
                )

    def _run_supervisor(self, t: float) -> None:
        """Run the centralised supervisor and act on its advisories."""
        if self._supervisor is None:
            return
        actions = self._supervisor.observe(self.vehicles, self.world, t)
        for action in actions:
            self.event_log.log(
                t, EventKind.SUPERVISOR_ACTION, Severity.INFO,
                vehicle_id=action.target_vehicle_id if action.target_vehicle_id >= 0 else None,
                message=f"Supervisor {action.action_type}: {action.reason}",
                action_type=action.action_type,
                **action.details,
            )
            if action.action_type == "replan":
                vid = action.target_vehicle_id
                v_list = [v for v in self.vehicles if v.id == vid]
                if v_list and v_list[0].is_operational:
                    v = v_list[0]
                    if v.assigned_destination:
                        from convoy_commander.planning.global_planner import plan_route
                        route = plan_route(
                            self.world,
                            v.state.x, v.state.y,
                            v.assigned_destination[0], v.assigned_destination[1],
                            objective=self.config.planning,
                        )
                        v.waypoints = route
                        v.current_waypoint_idx = 0
                        self.event_log.log(
                            t, EventKind.ROUTE_REPLAN, Severity.INFO, vehicle_id=vid,
                            message=f"Supervisor-triggered replan for V{vid}",
                        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_leader(self) -> Vehicle | None:
        """Get current leader vehicle."""
        for v in self.vehicles:
            if v.is_leader and v.is_operational:
                return v
        return None

    def _get_current_target(self, v: Vehicle) -> tuple[float, float] | None:
        """Get current waypoint target for vehicle."""
        if not v.waypoints:
            if v.assigned_destination:
                return v.assigned_destination
            return None
        if v.current_waypoint_idx < len(v.waypoints):
            return v.waypoints[v.current_waypoint_idx]
        return v.assigned_destination

    def _advance_waypoint(self, v: Vehicle) -> None:
        """Advance to next waypoint if close enough."""
        if not v.waypoints or v.current_waypoint_idx >= len(v.waypoints):
            return
        wp = v.waypoints[v.current_waypoint_idx]
        dist = math.hypot(v.state.x - wp[0], v.state.y - wp[1])
        if dist < 15.0:
            v.current_waypoint_idx += 1

    def _detect_collisions(self, t: float) -> None:
        """Detect collisions and near misses between vehicles."""
        collision_r = self.config.coordination.collision_radius
        min_sep = self.config.coordination.min_separation
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                a = self.vehicles[i]
                b = self.vehicles[j]
                if not a.is_operational or not b.is_operational:
                    continue
                dist = a.state.distance_to(b.state)
                if dist < collision_r:
                    a.collision_count += 1
                    b.collision_count += 1
                    self.event_log.log(
                        t, EventKind.COLLISION, Severity.CRITICAL,
                        message=f"COLLISION between V{a.id} and V{b.id} "
                                f"(dist={dist:.2f}m < {collision_r:.1f}m)",
                        vehicle_a=a.id, vehicle_b=b.id, distance=dist,
                    )
                elif dist < min_sep:
                    a.near_miss_count += 1
                    b.near_miss_count += 1
                    self.event_log.log(
                        t, EventKind.NEAR_MISS, Severity.WARNING,
                        message=f"Near miss V{a.id}–V{b.id} "
                                f"(dist={dist:.2f}m < min_sep={min_sep:.1f}m)",
                        vehicle_a=a.id, vehicle_b=b.id, distance=dist,
                    )
