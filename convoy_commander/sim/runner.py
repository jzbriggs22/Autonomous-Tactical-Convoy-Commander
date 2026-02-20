"""Main simulation runner."""

from __future__ import annotations

import math
import time as wall_time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from convoy_commander.comms.messages import (
    Message,
    MessageType,
    make_hazard,
    make_state_broadcast,
)
from convoy_commander.comms.network import CommsNetwork
from convoy_commander.coordination.allocation import allocate_waypoints_greedy
from convoy_commander.coordination.formation import (
    compute_formation_correction,
    get_formation_index,
)
from convoy_commander.coordination.leader_election import LeaderElection
from convoy_commander.core.config import SimConfig
from convoy_commander.core.world import World
from convoy_commander.metrics.collector import MetricsCollector
from convoy_commander.planning.global_planner import plan_route
from convoy_commander.planning.local_planner import compute_command
from convoy_commander.vehicles.vehicle import Vehicle, VehicleStatus


@dataclass
class SimResult:
    """Result of a simulation run."""

    config: SimConfig
    vehicles: list[Vehicle]
    world: World
    collector: MetricsCollector
    comms: CommsNetwork


class SimRunner:
    """Orchestrates the simulation loop."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.world = World(config.world, self.rng)
        self.comms = CommsNetwork(config.comms, self.rng)
        self.collector = MetricsCollector()

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

    def _plan_all_routes(self) -> None:
        """Plan global routes for all vehicles."""
        for v in self.vehicles:
            if v.assigned_destination is not None and v.is_operational:
                route = plan_route(
                    self.world,
                    v.estimator.state.x,
                    v.estimator.state.y,
                    v.assigned_destination[0],
                    v.assigned_destination[1],
                )
                v.waypoints = route
                v.current_waypoint_idx = 0

    def run(self, progress_callback: callable | None = None) -> SimResult:
        """Run the full simulation."""
        dt = self.config.dt
        total_steps = int(self.config.duration / dt)
        current_time = 0.0
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

            # === Planning and control ===
            leader = self._get_leader()
            operational_ids = [v.id for v in self.vehicles if v.is_operational]

            for v in self.vehicles:
                if not v.is_operational:
                    continue

                # Get current waypoint target
                target = self._get_current_target(v)
                if target is None:
                    continue

                # Formation correction
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

                # Boundary enforcement
                v.state.x = max(0, min(self.world.width, v.state.x))
                v.state.y = max(0, min(self.world.height, v.state.y))

                # Check waypoint advance
                self._advance_waypoint(v)

                # Check arrival
                if v.has_reached_destination():
                    v.status = VehicleStatus.ARRIVED
                    v.state.speed = 0.0
                    self.collector.record_arrival(v.id, current_time)

            # === Collision and near-miss detection ===
            self._detect_collisions()

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

        return SimResult(
            config=self.config,
            vehicles=self.vehicles,
            world=self.world,
            collector=self.collector,
            comms=self.comms,
        )

    def _handle_scenario_events(self, t: float) -> None:
        """Trigger scenario-specific events."""
        scenario = self.config.scenario

        if scenario == "leader_failure" and not self._leader_failed and t >= 120.0:
            self._leader_failed = True
            leader = self._get_leader()
            if leader:
                leader.set_breakdown()
                # Force election restart on all
                for v in self.vehicles:
                    if v.is_operational:
                        self.elections[v.id].reset()
                self.collector.leader_elections += 1

        if scenario == "obstacle_pop" and not self._obstacle_popped and t >= 90.0:
            self._obstacle_popped = True
            # Add obstacle in the middle of the likely path
            mid_x = self.world.width * 0.5
            mid_y = self.world.height * 0.5
            self.world.add_obstacle(mid_x, mid_y, 35.0)
            # Replan routes
            self._plan_all_routes()

    def _apply_position_fixes(self, t: float) -> None:
        """Apply GPS or landmark fixes to vehicles."""
        for v in self.vehicles:
            if not v.is_operational:
                continue

            # GPS fix
            if self.config.gps_available:
                v.estimator.apply_gps_fix(v.state.x, v.state.y)
            elif self.config.gps_intermittent_prob > 0:
                if self.rng.random() < self.config.gps_intermittent_prob * self.config.dt:
                    v.estimator.apply_gps_fix(v.state.x, v.state.y)

            # Landmark fixes
            nearby = self.world.landmarks_in_range(v.state.x, v.state.y)
            for lm in nearby:
                v.estimator.apply_landmark_fix(v.state.x, v.state.y)

    def _process_messages(self, t: float) -> None:
        """Process incoming messages for all vehicles."""
        for v in self.vehicles:
            if not v.is_operational:
                continue
            inbox = self.comms.get_inbox(v.id)
            for msg in inbox:
                v.last_comms_time = t
                if msg.msg_type == MessageType.LEADER_HEARTBEAT:
                    self.elections[v.id].on_heartbeat(msg, t)
                elif msg.msg_type == MessageType.LEADER_ELECTION:
                    self.elections[v.id].on_election_message(msg, t)
                elif msg.msg_type == MessageType.HAZARD:
                    # Could trigger replanning here
                    pass
                elif msg.msg_type == MessageType.STATE_BROADCAST:
                    # Update known positions of other vehicles (implicit via comms)
                    pass

    def _run_elections(self, t: float) -> None:
        """Run leader election logic for all vehicles."""
        positions = {v.id: (v.state.x, v.state.y) for v in self.vehicles if v.is_operational}
        for v in self.vehicles:
            if not v.is_operational:
                continue
            msgs = self.elections[v.id].check_and_elect(t)
            sender_pos = (v.state.x, v.state.y)
            for msg in msgs:
                self.comms.send_broadcast(msg, sender_pos, positions, t)

    def _broadcast_states(self, t: float) -> None:
        """Broadcast vehicle states."""
        positions = {v.id: (v.state.x, v.state.y) for v in self.vehicles if v.is_operational}
        for v in self.vehicles:
            if not v.is_operational:
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

    def _check_safe_mode(self, t: float) -> None:
        """Enter/exit safe mode based on conditions."""
        for v in self.vehicles:
            if not v.is_operational:
                continue

            should_safe = False

            # High position uncertainty
            if v.estimator.is_uncertain:
                should_safe = True

            # Comms lost for too long
            if t - v.last_comms_time > self.config.coordination.comms_lost_timeout and t > 5.0:
                should_safe = True

            if should_safe and v.status == VehicleStatus.ACTIVE:
                v.enter_safe_mode()
                self.collector.safe_mode_activations += 1
            elif not should_safe and v.status == VehicleStatus.SAFE_MODE:
                v.exit_safe_mode()

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

    def _detect_collisions(self) -> None:
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
                elif dist < min_sep:
                    a.near_miss_count += 1
                    b.near_miss_count += 1
