"""Multi-convoy simulation runner (Phase 12).

Orchestrates N independent convoys sharing one World and CommsNetwork.
The existing SimRunner remains completely untouched for backward
compatibility.  This runner reuses the same building blocks (LeaderElection,
CBBA, CommsNetwork, etc.) but iterates over ConvoyGroups.

SAFETY-CRITICAL DESIGN:
  - All safety-relevant events logged to the shared EventLog.
  - Inter-convoy collision detection is global (all vehicle pairs).
  - Right-of-way negotiation prevents route-crossing collisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from convoy_commander.comms.messages import (
    Message,
    MessageType,
    make_state_broadcast,
)
from convoy_commander.comms.network import CommsNetwork
from convoy_commander.coordination.allocation import allocate_waypoints_greedy
from convoy_commander.coordination.cbba import cbba_allocate, compute_formation_slots
from convoy_commander.coordination.convoy_group import ConvoyGroup
from convoy_commander.coordination.formation import (
    compute_formation_correction,
    get_formation_index,
)
from convoy_commander.coordination.leader_election import LeaderElection
from convoy_commander.coordination.merge_split import can_merge, merge_convoys, split_convoy
from convoy_commander.coordination.right_of_way import (
    apply_yield,
    detect_crossing,
    resolve_right_of_way,
)
from convoy_commander.core.convoy_config import MultiConvoyConfig
from convoy_commander.core.event_log import EventKind, EventLog, Severity
from convoy_commander.core.spatial import SpatialHash
from convoy_commander.core.world import World
from convoy_commander.metrics.collector import MetricsCollector
from convoy_commander.metrics.multi_metrics import (
    MultiConvoyMetrics,
    compute_multi_convoy_metrics,
)
from convoy_commander.planning.global_planner import plan_route
from convoy_commander.planning.local_planner import compute_command
from convoy_commander.stamp import ReproStamp, collect_stamp
from convoy_commander.vehicles.vehicle import CommsMode, Vehicle, VehicleStatus


@dataclass
class MultiConvoyResult:
    """Result of a multi-convoy simulation."""

    config: MultiConvoyConfig
    groups: dict[int, ConvoyGroup]
    all_vehicles: list[Vehicle]
    world: World
    comms: CommsNetwork
    event_log: EventLog
    aggregate_metrics: MultiConvoyMetrics | None = None
    stamp: ReproStamp | None = None


class MultiConvoyRunner:
    """Orchestrates a multi-convoy simulation."""

    def __init__(self, config: MultiConvoyConfig) -> None:
        self.config = config
        base = config.base
        self.rng = np.random.default_rng(base.seed)
        self.world = World(base.world, self.rng)
        self.comms = CommsNetwork(base.comms, self.rng)
        self.event_log = EventLog()

        self.event_log.log(
            0.0, EventKind.SIM_START, Severity.INFO,
            message=f"Multi-convoy simulation starting: "
                    f"{len(config.convoys)} convoys, "
                    f"{config.total_vehicles} total vehicles, "
                    f"seed={base.seed} duration={base.duration}s",
            num_convoys=len(config.convoys),
            total_vehicles=config.total_vehicles,
        )

        # Build convoy groups with globally unique vehicle IDs
        self.groups: dict[int, ConvoyGroup] = {}
        self.all_vehicles: list[Vehicle] = []
        vid_offset = 0
        for spec in config.convoys:
            vehicles: list[Vehicle] = []
            for i in range(spec.num_vehicles):
                vid = vid_offset + i
                row, col = i // 2, i % 2
                v = Vehicle(
                    vehicle_id=vid,
                    config=base,
                    rng=np.random.default_rng(base.seed + vid + 1),
                    start_x=spec.start_x + col * 25.0,
                    start_y=spec.start_y + row * 25.0,
                    start_heading=spec.start_heading,
                )
                v.convoy_id = spec.convoy_id
                vehicles.append(v)
                self.all_vehicles.append(v)
                self.event_log.log(
                    0.0, EventKind.VEHICLE_SPAWNED, Severity.INFO, vehicle_id=vid,
                    message=f"Vehicle {vid} spawned in convoy {spec.convoy_id}",
                    convoy_id=spec.convoy_id,
                )

            group = ConvoyGroup(
                convoy_id=spec.convoy_id,
                priority=spec.priority,
                vehicles=vehicles,
                destination=(spec.dest_x, spec.dest_y),
            )
            self._init_convoy_group(group)
            self.groups[spec.convoy_id] = group
            vid_offset += spec.num_vehicles

        # Shared spatial grids
        self._collision_grid = SpatialHash(base.coordination.min_separation)
        self._comms_grid = SpatialHash(max(base.comms.max_range / 3.0, 10.0))

        # Per-vehicle comms-lost tracking
        self._comms_lost_flags: dict[int, bool] = {v.id: False for v in self.all_vehicles}
        self._fuel_low_logged: set[int] = set()

        # Multi-convoy event counters
        self._inter_convoy_collisions = 0
        self._merge_count = 0
        self._split_count = 0
        self._right_of_way_yields = 0

        # Near-miss cooldown: {(vid_a, vid_b): last_logged_time}
        self._near_miss_cooldown: dict[tuple[int, int], float] = {}
        self._near_miss_cooldown_s: float = 1.0

        # Per-pair collision cooldown: one overlap = one event
        self._collision_cooldown: dict[tuple[int, int], float] = {}
        self._collision_cooldown_s: float = 0.5
        # Stuck timers and collision-triggered replan tracking
        self._stuck_timers: dict[int, float] = {v.id: 0.0 for v in self.all_vehicles}
        self._stuck_threshold: float = 5.0
        self._collision_window: dict[int, list[float]] = {v.id: [] for v in self.all_vehicles}
        self._collision_replan_threshold: int = 4
        self._collision_window_duration: float = 3.0
        self._last_replan_time: dict[int, float] = {v.id: -999.0 for v in self.all_vehicles}
        self._replan_cooldown: float = 8.0
        self._collision_partners: dict[int, dict[int, int]] = {v.id: {} for v in self.all_vehicles}
        self._hold_position_until: dict[int, float] = {}
        self._hold_reverse_target: dict[int, tuple[float, float]] = {}

        # Scenario event flags
        self._split_triggered = False

    def _init_convoy_group(self, group: ConvoyGroup) -> None:
        """Set up elections, initial leader, and routes for one convoy."""
        base = self.config.base
        for v in group.vehicles:
            group.elections[v.id] = LeaderElection(
                v, base.coordination.leader_heartbeat_timeout,
            )

        # Designate first vehicle as initial leader
        if group.vehicles:
            group.vehicles[0].is_leader = True
            group.vehicles[0].leader_id = group.vehicles[0].id
            for v in group.vehicles:
                v.leader_id = group.vehicles[0].id
                group.elections[v.id].last_heartbeat_time = 0.0
            self.event_log.log(
                0.0, EventKind.LEADER_ELECTED, Severity.INFO,
                vehicle_id=group.vehicles[0].id,
                message=f"Vehicle {group.vehicles[0].id} designated as initial leader "
                        f"of convoy {group.convoy_id}",
            )

        # Assign destination and plan routes
        allocate_waypoints_greedy(group.vehicles, group.destination)
        self._plan_group_routes(group)

        # Initial CBBA allocation
        if base.use_cbba:
            self._run_cbba_for_group(group)

    def _plan_group_routes(self, group: ConvoyGroup) -> None:
        """Plan global routes for all vehicles in a convoy group."""
        objective = self.config.base.planning
        for v in group.vehicles:
            if v.assigned_destination is not None and v.is_operational:
                route = plan_route(
                    self.world,
                    v.estimator.state.x, v.estimator.state.y,
                    v.assigned_destination[0], v.assigned_destination[1],
                    objective=objective,
                )
                v.waypoints = route
                v.current_waypoint_idx = 0

    def _run_cbba_for_group(self, group: ConvoyGroup) -> None:
        """Run CBBA-lite auction for one convoy group."""
        leader = group.get_leader()
        operational = group.operational_vehicles
        if not operational or leader is None:
            return
        slot_positions = compute_formation_slots(
            leader.estimator.state.x,
            leader.estimator.state.y,
            leader.estimator.state.heading,
            len(operational),
            self.config.base.coordination.formation_spacing,
        )
        group.cbba_slots = cbba_allocate(operational, slot_positions)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self, progress_callback=None) -> MultiConvoyResult:
        """Run the full multi-convoy simulation."""
        base = self.config.base
        dt = base.dt
        total_steps = int(base.duration / dt)

        for step in range(total_steps):
            current_time = step * dt

            if progress_callback and step % 100 == 0:
                progress_callback(step, total_steps)

            # 1. Handle scenario events (merge/split triggers)
            self._handle_scenario_events(current_time)

            # 2. Rebuild shared spatial grids (all vehicles)
            self._rebuild_spatial_grids()

            # 3. Position fixes for all vehicles
            self._apply_position_fixes(current_time)

            # 4. Comms tick (shared network)
            self.comms.tick(current_time)

            # 5. Per-convoy coordination
            for gid, group in list(self.groups.items()):
                if not group.vehicles:
                    continue
                self._process_messages_for_group(group, current_time)
                self._run_elections_for_group(group, current_time)
                group.broadcast_timer += dt
                if group.broadcast_timer >= base.comms.broadcast_interval:
                    group.broadcast_timer = 0.0
                    self._broadcast_states_for_group(group, current_time)
                self._check_safe_mode_for_group(group, current_time)
                if base.use_cbba:
                    group.cbba_realloc_timer += dt
                    if group.cbba_realloc_timer >= 10.0:
                        group.cbba_realloc_timer = 0.0
                        self._run_cbba_for_group(group)

            # 6. Right-of-way negotiation
            self._check_right_of_way(current_time)

            # 7. Planning and control (per-convoy, with yield speed factor)
            for gid, group in self.groups.items():
                if not group.vehicles:
                    continue
                self._step_vehicles_for_group(group, current_time, dt)

            # 8. Fuel monitoring
            self._check_fuel(current_time)

            # 9. Global collision detection (all vehicles)
            self._detect_collisions_global(current_time)

            # 10. Record metrics per-convoy
            for group in self.groups.values():
                if group.vehicles:
                    group.collector.record_step(current_time, group.vehicles)

            # 11. Check if all convoys arrived
            if all(
                v.has_reached_destination() or not v.is_operational
                for v in self.all_vehicles
            ):
                break

        # --- Log simulation end ---
        final_time = min(total_steps * dt, base.duration)
        arrived = sum(1 for v in self.all_vehicles if v.has_reached_destination())
        self.event_log.log(
            final_time, EventKind.SIM_END, Severity.INFO,
            message=f"Multi-convoy simulation ended: {arrived}/{len(self.all_vehicles)} arrived",
            arrived=arrived, total=len(self.all_vehicles),
        )

        # Compute metrics
        per_convoy_metrics = {}
        for gid, group in self.groups.items():
            if group.vehicles:
                per_convoy_metrics[gid] = group.collector.compute_final(
                    group.vehicles,
                    self.comms.total_sent, self.comms.total_delivered,
                    self.comms.total_dropped, base.duration,
                )

        aggregate = compute_multi_convoy_metrics(
            per_convoy_metrics,
            inter_convoy_collisions=self._inter_convoy_collisions,
            merge_count=self._merge_count,
            split_count=self._split_count,
            right_of_way_yields=self._right_of_way_yields,
        )

        stamp = collect_stamp(base)

        return MultiConvoyResult(
            config=self.config,
            groups=self.groups,
            all_vehicles=self.all_vehicles,
            world=self.world,
            comms=self.comms,
            event_log=self.event_log,
            aggregate_metrics=aggregate,
            stamp=stamp,
        )

    # ------------------------------------------------------------------
    # Scenario events
    # ------------------------------------------------------------------

    def _handle_scenario_events(self, t: float) -> None:
        """Trigger scenario-specific events (merge, split)."""
        scenario = self.config.base.scenario

        # convoy_split_reroute: split at t=60s
        if scenario == "convoy_split_reroute" and not self._split_triggered and t >= 60.0:
            self._split_triggered = True
            # Split the first (and only) convoy
            if 0 in self.groups:
                source = self.groups[0]
                # Split second half of vehicles
                all_ids = sorted(v.id for v in source.vehicles)
                mid = len(all_ids) // 2
                split_ids = all_ids[mid:]

                new_dest = (80.0, self.config.base.world.height - 80.0)
                new_group = split_convoy(
                    source, split_ids,
                    new_convoy_id=1,
                    new_destination=new_dest,
                    new_priority=0,
                    heartbeat_timeout=self.config.base.coordination.leader_heartbeat_timeout,
                    min_vehicles=self.config.split_min_vehicles,
                )
                if new_group is not None:
                    self.groups[1] = new_group
                    # Plan routes for new group
                    allocate_waypoints_greedy(new_group.vehicles, new_group.destination)
                    self._plan_group_routes(new_group)
                    # Re-plan source routes (destination unchanged)
                    self._plan_group_routes(source)
                    self._split_count += 1
                    self.event_log.log(
                        t, EventKind.CONVOY_SPLIT, Severity.INFO,
                        message=f"Convoy 0 split: {len(source.vehicles)} stay, "
                                f"{len(new_group.vehicles)} form convoy 1 "
                                f"heading to ({new_dest[0]:.0f}, {new_dest[1]:.0f})",
                        source_convoy=0, new_convoy=1,
                    )

        # convoy_merge: check merge conditions every step
        if scenario == "convoy_merge":
            self._check_merge(t)

    def _check_merge(self, t: float) -> None:
        """Check all convoy pairs for merge conditions."""
        group_ids = list(self.groups.keys())
        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                ga = self.groups[group_ids[i]]
                gb = self.groups[group_ids[j]]
                if not ga.vehicles or not gb.vehicles:
                    continue
                if can_merge(ga, gb, self.config.merge_distance):
                    # Higher priority absorbs lower; tie: lower id absorbs
                    if ga.priority > gb.priority or (
                        ga.priority == gb.priority and ga.convoy_id < gb.convoy_id
                    ):
                        absorber, absorbed = ga, gb
                    else:
                        absorber, absorbed = gb, ga

                    merge_convoys(
                        absorber, absorbed,
                        self.config.base.coordination.leader_heartbeat_timeout,
                    )
                    # Update all_vehicles list references
                    allocate_waypoints_greedy(absorber.vehicles, absorber.destination)
                    self._plan_group_routes(absorber)
                    self._merge_count += 1
                    self.event_log.log(
                        t, EventKind.CONVOY_MERGE, Severity.INFO,
                        message=f"Convoy {absorbed.convoy_id} merged into "
                                f"convoy {absorber.convoy_id} "
                                f"({len(absorber.vehicles)} vehicles)",
                        absorber=absorber.convoy_id,
                        absorbed=absorbed.convoy_id,
                    )

    # ------------------------------------------------------------------
    # Right-of-way
    # ------------------------------------------------------------------

    def _check_right_of_way(self, t: float) -> None:
        """Check all convoy pairs for right-of-way crossing."""
        group_ids = [gid for gid, g in self.groups.items() if g.vehicles]
        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                ga = self.groups[group_ids[i]]
                gb = self.groups[group_ids[j]]
                if detect_crossing(ga, gb, self.config.right_of_way_radius):
                    yielder = resolve_right_of_way(ga, gb, t)
                    if yielder is not None:
                        self._right_of_way_yields += 1
                        self.event_log.log(
                            t, EventKind.RIGHT_OF_WAY_YIELD, Severity.INFO,
                            message=f"Convoy {yielder.convoy_id} yielding "
                                    f"(priority={yielder.priority}) "
                                    f"until t={yielder.yield_until:.1f}s",
                            convoy_id=yielder.convoy_id,
                        )

        # Check yield expiry and log clearance
        for group in self.groups.values():
            if group.yielding and t >= group.yield_until:
                group.yielding = False
                group.yield_until = 0.0
                self.event_log.log(
                    t, EventKind.RIGHT_OF_WAY_CLEAR, Severity.INFO,
                    message=f"Convoy {group.convoy_id} yield cleared",
                    convoy_id=group.convoy_id,
                )

    # ------------------------------------------------------------------
    # Per-convoy coordination
    # ------------------------------------------------------------------

    def _process_messages_for_group(self, group: ConvoyGroup, t: float) -> None:
        """Process incoming messages for vehicles in one convoy."""
        convoy_vids = group.vehicle_ids()
        for v in group.vehicles:
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

                # Convoy-scoped coordination: only process election/heartbeat
                # from same-convoy vehicles
                if msg.msg_type == MessageType.LEADER_HEARTBEAT:
                    if msg.sender_id in convoy_vids:
                        group.elections[v.id].on_heartbeat(msg, t)
                elif msg.msg_type == MessageType.LEADER_ELECTION:
                    if msg.sender_id in convoy_vids:
                        group.elections[v.id].on_election_message(msg, t)
                elif msg.msg_type == MessageType.STATE_BROADCAST:
                    # State broadcasts from any convoy — for collision avoidance
                    v.update_neighbor(msg.sender_id, msg.payload, msg.timestamp)

    def _run_elections_for_group(self, group: ConvoyGroup, t: float) -> None:
        """Run leader election logic for vehicles in one convoy."""
        positions = {v.id: (v.state.x, v.state.y)
                     for v in self.all_vehicles if v.is_operational}
        prev_leaders = {v.id for v in group.vehicles if v.is_leader}

        for v in group.vehicles:
            if not v.is_operational:
                continue
            msgs = group.elections[v.id].check_and_elect(t)
            sender_pos = (v.state.x, v.state.y)
            for msg in msgs:
                self.comms.send_broadcast(msg, sender_pos, positions, t, self._comms_grid)

        # Detect new leader
        for v in group.vehicles:
            if v.is_leader and v.id not in prev_leaders:
                self.event_log.log(
                    t, EventKind.LEADER_ELECTED, Severity.INFO, vehicle_id=v.id,
                    message=f"Vehicle {v.id} elected as leader of convoy {group.convoy_id}",
                )
                group.collector.leader_elections += 1

    def _broadcast_states_for_group(self, group: ConvoyGroup, t: float) -> None:
        """Broadcast vehicle states for one convoy."""
        positions = {v.id: (v.state.x, v.state.y)
                     for v in self.all_vehicles if v.is_operational}
        for v in group.vehicles:
            if not v.is_operational:
                continue
            if v.comms_mode == CommsMode.SILENT:
                continue
            msg = make_state_broadcast(
                sender_id=v.id, timestamp=t,
                x=v.estimator.state.x, y=v.estimator.state.y,
                heading=v.estimator.state.heading, speed=v.state.speed,
                uncertainty=v.estimator.state.uncertainty,
                status=v.status.name, fuel=v.fuel.fuel,
            )
            self.comms.send_broadcast(
                msg, (v.state.x, v.state.y), positions, t, self._comms_grid,
            )

    def _check_safe_mode_for_group(self, group: ConvoyGroup, t: float) -> None:
        """Enter/exit safe mode for vehicles in one convoy."""
        comms_timeout = self.config.base.coordination.comms_lost_timeout
        for v in group.vehicles:
            if not v.is_operational:
                continue
            reasons: list[str] = []
            if v.estimator.is_uncertain:
                reasons.append(
                    f"uncertainty={v.estimator.state.uncertainty:.1f}m"
                )
            comms_gap = t - v.last_comms_time
            if comms_gap > comms_timeout and t > 5.0:
                reasons.append(f"comms_lost={comms_gap:.1f}s")
                if not self._comms_lost_flags.get(v.id, False):
                    self._comms_lost_flags[v.id] = True
                    self.event_log.log(
                        t, EventKind.COMMS_LOST, Severity.WARNING, vehicle_id=v.id,
                        message=f"Vehicle {v.id} comms lost for {comms_gap:.1f}s",
                    )
            if reasons and v.status == VehicleStatus.ACTIVE:
                v.enter_safe_mode()
                group.collector.safe_mode_activations += 1
                self.event_log.log(
                    t, EventKind.SAFE_MODE_ENTER, Severity.WARNING, vehicle_id=v.id,
                    message=f"Vehicle {v.id} entering safe mode: {'; '.join(reasons)}",
                )
            elif not reasons and v.status == VehicleStatus.SAFE_MODE:
                v.exit_safe_mode()
                self.event_log.log(
                    t, EventKind.SAFE_MODE_EXIT, Severity.INFO, vehicle_id=v.id,
                    message=f"Vehicle {v.id} exiting safe mode",
                )

    # ------------------------------------------------------------------
    # Vehicle stepping (planning + control)
    # ------------------------------------------------------------------

    def _step_vehicles_for_group(
        self, group: ConvoyGroup, t: float, dt: float,
    ) -> None:
        """Plan and control vehicles for one convoy group."""
        base = self.config.base
        yield_factor = apply_yield(group, t)
        leader = group.get_leader()
        operational_ids = [v.id for v in group.vehicles if v.is_operational]

        if leader is None and len(operational_ids) > 0:
            pass  # no leader — vehicles navigate independently

        for v in group.vehicles:
            if not v.is_operational:
                continue

            # Hold/reverse: vehicle ordered to move away from collision partner
            if t < self._hold_position_until.get(v.id, 0.0):
                from convoy_commander.vehicles.vehicle import VehicleCommand as _VC
                rev_target = self._hold_reverse_target.get(v.id)
                if rev_target is not None:
                    neighbors = [vv for vv in self.all_vehicles if vv.id != v.id]
                    cmd = compute_command(v, rev_target, self.world, neighbors, dt,
                                          stuck_time=self._stuck_timers.get(v.id, 0.0))
                else:
                    cmd = _VC(accel=-v.vcfg.max_decel * 0.5, turn_rate=0.0)
                v.step(cmd, dt)
                self._enforce_safety_envelope(v, t)
                self._advance_waypoint(v)
                if v.has_reached_destination() and v.status != VehicleStatus.ARRIVED:
                    v.status = VehicleStatus.ARRIVED
                    v.state.speed = 0.0
                    group.collector.record_arrival(v.id, t)
                continue

            # Get current waypoint target
            target = self._get_current_target(v)
            if target is None:
                continue

            # Formation correction — suppress in scatter mode
            recent_collisions = len(self._collision_window.get(v.id, []))
            in_scatter_mode = recent_collisions >= 4

            if base.use_cbba and v.id in group.cbba_slots:
                formation_idx = group.cbba_slots[v.id]
            else:
                formation_idx = get_formation_index(
                    v.id, leader.id if leader else None, operational_ids,
                )
            if leader and not v.is_leader and not in_scatter_mode:
                correction = compute_formation_correction(
                    v, leader,
                    [vv for vv in group.vehicles if vv.id != v.id and vv.is_operational],
                    formation_idx, v.effective_spacing,
                )
                corr_scale = 1.5
                for vv in group.vehicles:
                    if vv.id != v.id and vv.is_operational:
                        d = v.state.distance_to(vv.state)
                        if d < base.coordination.min_separation * 1.5:
                            corr_scale = min(corr_scale, 0.3)
                            break
                target = (
                    target[0] + correction[0] * corr_scale,
                    target[1] + correction[1] * corr_scale,
                )

            # Compute command
            neighbors = [vv for vv in self.all_vehicles if vv.id != v.id]
            cmd = compute_command(v, target, self.world, neighbors, dt,
                                  stuck_time=self._stuck_timers.get(v.id, 0.0))

            # Apply yield speed factor
            if yield_factor < 1.0 and cmd.accel > 0:
                cmd.accel *= yield_factor

            # Safe mode speed clamping
            if v.status == VehicleStatus.SAFE_MODE:
                max_safe = v.effective_max_speed
                if v.state.speed > max_safe and cmd.accel > 0:
                    cmd.accel = -v.vcfg.max_decel * 0.3

            # Emergency braking: only when converging toward another vehicle.
            collision_r = base.coordination.collision_radius
            for vv in self.all_vehicles:
                if vv.id != v.id and vv.is_operational:
                    d = v.state.distance_to(vv.state)
                    dx = vv.state.x - v.state.x
                    dy = vv.state.y - v.state.y
                    if d > 0.1:
                        nx, ny = dx / d, dy / d
                        v_rel = (
                            (v.state.speed * math.cos(v.state.heading)
                             - vv.state.speed * math.cos(vv.state.heading)) * nx
                            + (v.state.speed * math.sin(v.state.heading)
                               - vv.state.speed * math.sin(vv.state.heading)) * ny
                        )
                    else:
                        v_rel = v.state.speed

                    if d < collision_r * 1.3:
                        cmd.accel = -v.vcfg.max_decel * 0.8
                        away_angle = math.atan2(-dy, -dx)
                        steer_err = away_angle - v.state.heading
                        steer_err = math.atan2(math.sin(steer_err), math.cos(steer_err))
                        cmd.turn_rate = max(-v.vcfg.max_turn_rate,
                                            min(v.vcfg.max_turn_rate, steer_err * 3.0))
                        break
                    elif d < collision_r * 2.0 and v_rel > 0.5:
                        cmd.accel = min(cmd.accel, -v.vcfg.max_decel * 0.4)
                    elif d < base.coordination.min_separation and v_rel > 1.0:
                        cmd.accel = min(cmd.accel, -v.vcfg.max_decel * 0.15)

            v.step(cmd, dt)

            # Safety envelope
            self._enforce_safety_envelope(v, t)

            # Waypoint advance
            self._advance_waypoint(v)

            # Stuck detection
            if v.state.speed < 0.5 and v.status == VehicleStatus.ACTIVE:
                self._stuck_timers[v.id] = self._stuck_timers.get(v.id, 0.0) + dt
                if self._stuck_timers[v.id] >= self._stuck_threshold:
                    if v.waypoints and v.current_waypoint_idx < len(v.waypoints) - 1:
                        skip_count = min(3, len(v.waypoints) - 1 - v.current_waypoint_idx)
                        v.current_waypoint_idx += skip_count
                        self._stuck_timers[v.id] = 0.0
            else:
                self._stuck_timers[v.id] = 0.0

            # Check arrival
            if v.has_reached_destination() and v.status != VehicleStatus.ARRIVED:
                v.status = VehicleStatus.ARRIVED
                v.state.speed = 0.0
                group.collector.record_arrival(v.id, t)
                self.event_log.log(
                    t, EventKind.VEHICLE_ARRIVED, Severity.INFO, vehicle_id=v.id,
                    message=f"Vehicle {v.id} (convoy {group.convoy_id}) arrived",
                )

    # ------------------------------------------------------------------
    # Position fixes
    # ------------------------------------------------------------------

    def _apply_position_fixes(self, t: float) -> None:
        """Apply GPS or landmark fixes to all vehicles."""
        for v in self.all_vehicles:
            if not v.is_operational:
                continue
            # GPS fix
            if self.config.base.gps_available:
                spoof_offset = self.world.get_spoof_offset(v.state.x, v.state.y)
                meas_x = v.state.x + (spoof_offset[0] if spoof_offset else 0.0)
                meas_y = v.state.y + (spoof_offset[1] if spoof_offset else 0.0)
                v.estimator.apply_gps_fix(meas_x, meas_y)
            # Landmark fixes
            nearby = self.world.landmarks_in_range(v.state.x, v.state.y)
            for lm in nearby:
                v.estimator.apply_landmark_fix(v.state.x, v.state.y)

    # ------------------------------------------------------------------
    # Collision detection (global, inter-convoy)
    # ------------------------------------------------------------------

    def _detect_collisions_global(self, t: float) -> None:
        """Detect collisions and near misses across all vehicles."""
        collision_r = self.config.base.coordination.collision_radius
        min_sep = self.config.base.coordination.min_separation
        positions = {v.id: (v.state.x, v.state.y) for v in self.all_vehicles}
        checked: set[tuple[int, int]] = set()

        for v in self.all_vehicles:
            if not v.is_operational:
                continue
            nearby = self._collision_grid.query_radius(
                v.state.x, v.state.y, min_sep, positions,
            )
            for nid in nearby:
                if nid == v.id:
                    continue
                pair = (min(v.id, nid), max(v.id, nid))
                if pair in checked:
                    continue
                checked.add(pair)

                other = next(vv for vv in self.all_vehicles if vv.id == nid)
                dist = v.state.distance_to(other.state)
                same_convoy = v.convoy_id == other.convoy_id

                if dist < collision_r:
                    last_col = self._collision_cooldown.get(pair, -999.0)
                    if t - last_col >= self._collision_cooldown_s:
                        v.collision_count += 1
                        other.collision_count += 1
                        self._collision_window.setdefault(v.id, []).append(t)
                        self._collision_window.setdefault(other.id, []).append(t)
                        vp = self._collision_partners.setdefault(v.id, {})
                        vp[other.id] = vp.get(other.id, 0) + 1
                        op = self._collision_partners.setdefault(other.id, {})
                        op[v.id] = op.get(v.id, 0) + 1
                        self._collision_cooldown[pair] = t
                        if same_convoy:
                            kind = EventKind.COLLISION
                        else:
                            kind = EventKind.INTER_CONVOY_COLLISION
                            self._inter_convoy_collisions += 1
                        self.event_log.log(
                            t, kind, Severity.CRITICAL,
                            message=f"{'INTER-CONVOY ' if not same_convoy else ''}"
                                    f"COLLISION V{v.id}(c{v.convoy_id})–"
                                    f"V{other.id}(c{other.convoy_id}) "
                                    f"dist={dist:.2f}m",
                            vehicle_a=v.id, vehicle_b=other.id, distance=dist,
                        )
                elif dist < min_sep:
                    # Cooldown: only count once per pair per cooldown window
                    last_logged = self._near_miss_cooldown.get(pair, -999.0)
                    if t - last_logged >= self._near_miss_cooldown_s:
                        v.near_miss_count += 1
                        other.near_miss_count += 1
                        self._near_miss_cooldown[pair] = t
                        if same_convoy:
                            kind = EventKind.NEAR_MISS
                        else:
                            kind = EventKind.INTER_CONVOY_NEAR_MISS
                        self.event_log.log(
                            t, kind, Severity.WARNING,
                            message=f"Near miss V{v.id}(c{v.convoy_id})–"
                                    f"V{other.id}(c{other.convoy_id}) "
                                    f"dist={dist:.2f}m",
                            vehicle_a=v.id, vehicle_b=other.id, distance=dist,
                        )

        # Check collision-triggered replanning
        self._check_collision_replan(t)

    def _check_collision_replan(self, t: float) -> None:
        """Replan route for vehicles with excessive recent collisions."""
        actions: list[tuple] = []  # (vehicle, action_type, partner_id)

        for v in self.all_vehicles:
            if not v.is_operational or v.assigned_destination is None:
                continue
            if t < self._hold_position_until.get(v.id, 0.0):
                continue
            window = self._collision_window.get(v.id, [])
            cutoff = t - self._collision_window_duration
            self._collision_window[v.id] = [ts for ts in window if ts > cutoff]
            window = self._collision_window[v.id]

            if (len(window) >= self._collision_replan_threshold
                    and t - self._last_replan_time.get(v.id, -999.0) > self._replan_cooldown):

                partners = self._collision_partners.get(v.id, {})
                top_partner_id = max(partners, key=partners.get) if partners else None
                top_partner_count = partners.get(top_partner_id, 0) if top_partner_id is not None else 0

                d_goal = float("inf")
                if v.assigned_destination is not None:
                    d_goal = math.hypot(
                        v.state.x - v.assigned_destination[0],
                        v.state.y - v.assigned_destination[1],
                    )

                if top_partner_count >= 3 and top_partner_id is not None:
                    if v.id > top_partner_id:
                        if d_goal < 150.0:
                            actions.append((v, "brake_hold", top_partner_id))
                        else:
                            actions.append((v, "hold", top_partner_id))
                    else:
                        actions.append((v, "pair_replan", top_partner_id))
                else:
                    actions.append((v, "normal_replan", None))

        for v, action, partner_id in actions:
            self._last_replan_time[v.id] = t
            window = self._collision_window[v.id]

            if action == "brake_hold":
                self._hold_position_until[v.id] = t + 3.0
                self._hold_reverse_target.pop(v.id, None)
                self._collision_window[v.id] = []
                self._collision_partners[v.id] = {}
                if partner_id is not None:
                    self._last_replan_time[partner_id] = -999.0
                continue

            if action == "hold":
                self._hold_position_until[v.id] = t + 8.0
                partner_v = next((vv for vv in self.all_vehicles if vv.id == partner_id), None)
                if partner_v is not None:
                    dx = v.state.x - partner_v.state.x
                    dy = v.state.y - partner_v.state.y
                    d = math.hypot(dx, dy)
                    if d > 0.1:
                        rev_x = v.state.x + (dx / d) * 60.0
                        rev_y = v.state.y + (dy / d) * 60.0
                    else:
                        rev_x = v.state.x + 60.0
                        rev_y = v.state.y
                    rev_x = max(20.0, min(self.world.width - 20.0, rev_x))
                    rev_y = max(20.0, min(self.world.height - 20.0, rev_y))
                    self._hold_reverse_target[v.id] = (rev_x, rev_y)
                    route = plan_route(
                        self.world, rev_x, rev_y,
                        v.assigned_destination[0], v.assigned_destination[1],
                        objective=self.config.base.planning,
                    )
                    v.waypoints = [(rev_x, rev_y)] + route
                    v.current_waypoint_idx = 0
                self._collision_window[v.id] = []
                self._collision_partners[v.id] = {}
                continue

            if action == "pair_replan":
                dx = v.assigned_destination[0] - v.state.x
                dy = v.assigned_destination[1] - v.state.y
                d = math.hypot(dx, dy)
                if d > 1.0:
                    lateral_sign = 1.0 if v.id % 2 == 0 else -1.0
                    offset_x = v.state.x + (-dy / d) * lateral_sign * 40.0
                    offset_y = v.state.y + (dx / d) * lateral_sign * 40.0
                else:
                    offset_x = v.state.x + 40.0
                    offset_y = v.state.y
                self._collision_partners[v.id] = {}
            else:
                dx = v.assigned_destination[0] - v.state.x
                dy = v.assigned_destination[1] - v.state.y
                d = math.hypot(dx, dy)
                if d > 1.0:
                    lateral_sign = 1.0 if v.id % 2 == 0 else -1.0
                    offset_x = v.state.x + (dx / d) * 5.0 + (-dy / d) * lateral_sign * 15.0
                    offset_y = v.state.y + (dy / d) * 5.0 + (dx / d) * lateral_sign * 15.0
                else:
                    offset_x, offset_y = v.state.x, v.state.y

            route = plan_route(
                self.world,
                offset_x, offset_y,
                v.assigned_destination[0], v.assigned_destination[1],
                objective=self.config.base.planning,
            )
            v.waypoints = route
            v.current_waypoint_idx = 0
            self._collision_window[v.id] = []

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    def _enforce_safety_envelope(self, v: Vehicle, t: float) -> None:
        """Post-step invariant checks."""
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
            v.state.speed = 0.0
            self.event_log.log(
                t, EventKind.BOUNDARY_VIOLATION, Severity.WARNING, vehicle_id=v.id,
                message=f"Vehicle {v.id} clamped to boundary",
            )
        # Speed limit
        hard_max = v.vcfg.max_speed * 1.01
        if v.state.speed > hard_max:
            v.state.speed = v.vcfg.max_speed
        # Obstacle check
        if self.world.is_blocked(v.state.x, v.state.y):
            v.state.speed = 0.0

    def _check_fuel(self, t: float) -> None:
        """Monitor fuel levels."""
        for v in self.all_vehicles:
            if not v.is_operational:
                continue
            if v.fuel.is_empty:
                v.status = VehicleStatus.BREAKDOWN
                v.state.speed = 0.0
                self.event_log.log(
                    t, EventKind.VEHICLE_FUEL_EMPTY, Severity.CRITICAL, vehicle_id=v.id,
                    message=f"Vehicle {v.id} fuel exhausted",
                )
            elif v.fuel.is_low and v.id not in self._fuel_low_logged:
                self._fuel_low_logged.add(v.id)
                self.event_log.log(
                    t, EventKind.VEHICLE_FUEL_LOW, Severity.WARNING, vehicle_id=v.id,
                    message=f"Vehicle {v.id} fuel below 20%",
                )

    def _rebuild_spatial_grids(self) -> None:
        """Rebuild spatial hash grids from all vehicle positions."""
        self._collision_grid.clear()
        self._comms_grid.clear()
        for v in self.all_vehicles:
            if v.is_operational:
                self._collision_grid.insert(v.id, v.state.x, v.state.y)
                self._comms_grid.insert(v.id, v.state.x, v.state.y)

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
        """Advance to next waypoint if close enough.

        Uses a speed-adaptive radius: faster vehicles can advance earlier
        to avoid overshooting and circling back.
        """
        if not v.waypoints or v.current_waypoint_idx >= len(v.waypoints):
            return
        wp = v.waypoints[v.current_waypoint_idx]
        dist = math.hypot(v.state.x - wp[0], v.state.y - wp[1])
        threshold = 15.0 + v.state.speed * 0.5
        if dist < threshold:
            v.current_waypoint_idx += 1
