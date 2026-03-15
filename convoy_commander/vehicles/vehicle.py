"""Vehicle agent with dynamics, fuel, and estimator.

SAFETY-CRITICAL ASSUMPTIONS:
  A1. A vehicle in BREAKDOWN status executes no commands and remains stationary.
      No further state transitions are possible from BREAKDOWN.
  A2. Fuel depletion is treated identically to mechanical breakdown
      (fail-safe: stop and remain stopped).
  A3. Safe mode is the *only* correct response to high uncertainty or
      prolonged comms loss.  The sim runner is responsible for triggering
      safe mode; the vehicle itself cannot override a safe-mode entry.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from convoy_commander.core.config import SimConfig
from convoy_commander.core.physics import FuelState, KinematicState
from convoy_commander.vehicles.estimator import PositionEstimator


def _point_to_segment_dist(
    px: float, py: float,
    a: tuple[float, float], b: tuple[float, float],
) -> float:
    """Distance from point (px, py) to line segment a-b."""
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


class VehicleStatus(Enum):
    ACTIVE = auto()
    SAFE_MODE = auto()
    BREAKDOWN = auto()
    ARRIVED = auto()


class CommsMode(Enum):
    """Communications emission mode for behaviour switching.

    NORMAL  — broadcast at the configured interval (default).
    SILENT  — suppress all state broadcasts (stealth / emissions control).
    CHATTY  — broadcast every step (maximum coordination fidelity).
    """

    NORMAL = "normal"
    SILENT = "silent"
    CHATTY = "chatty"


@dataclass
class VehicleCommand:
    """Control command for a vehicle."""

    accel: float = 0.0
    turn_rate: float = 0.0


class Vehicle:
    """Autonomous convoy vehicle."""

    def __init__(
        self,
        vehicle_id: int,
        config: SimConfig,
        rng: np.random.Generator,
        start_x: float = 0.0,
        start_y: float = 0.0,
        start_heading: float = 0.0,
    ) -> None:
        self.id = vehicle_id
        self.config = config
        self.rng = rng
        self.vcfg = config.vehicle

        # True kinematic state
        self.state = KinematicState(x=start_x, y=start_y, heading=start_heading, speed=0.0)
        # History of true positions
        self.true_history: list[tuple[float, float]] = [(start_x, start_y)]
        self.est_history: list[tuple[float, float]] = [(start_x, start_y)]

        # Fuel
        self.fuel = FuelState(
            fuel=config.vehicle.fuel_capacity,
            capacity=config.vehicle.fuel_capacity,
            rate_idle=config.vehicle.fuel_rate_idle,
            rate_per_speed=config.vehicle.fuel_rate_per_speed,
        )

        # Estimator
        self.estimator = PositionEstimator(config.estimator, rng)
        self.estimator.initialize(self.state)

        # Status
        self.status = VehicleStatus.ACTIVE
        self.comms_mode = CommsMode.NORMAL
        self.is_leader = False
        self.leader_id: int | None = None
        self.safe_mode_timer: float = 0.0
        self.last_comms_time: float = 0.0
        self._comms_was_lost: bool = False  # edge-detect for logging
        self._in_blackout: bool = False  # edge-detect for blackout zone logging

        # Planning
        self.waypoints: list[tuple[float, float]] = []
        self.current_waypoint_idx: int = 0
        self.assigned_destination: tuple[float, float] | None = None

        # Actuator lag (Phase 6): dead-time buffer + hold-last
        self._actuator_lag = config.vehicle.actuator_lag
        self._current_time: float = 0.0
        self._command_buffer: deque[tuple[float, VehicleCommand]] = deque()
        self._last_applied_cmd: VehicleCommand | None = None

        # Neighbor state table (Phase 8): cache latest STATE_BROADCAST per peer
        # Keys: neighbor vehicle_id; values: dict with x, y, heading, speed,
        # uncertainty, fuel, status, timestamp
        self.neighbor_states: dict[int, dict[str, object]] = {}

        # Weather modifiers (set per-step by runner)
        self.weather_speed_factor: float = 1.0
        self.weather_accel_factor: float = 1.0
        self.weather_fuel_factor: float = 1.0
        self.weather_spacing_factor: float = 1.0

        # Metrics
        self.total_distance: float = 0.0
        self.near_miss_count: int = 0
        self.collision_count: int = 0
        self.total_fuel_consumed: float = 0.0

    def step(self, command: VehicleCommand, dt: float) -> None:
        """Advance vehicle by one timestep.

        Preconditions: dt > 0, status != BREAKDOWN (no-op if it is).

        Actuator lag: commands are buffered and delayed by ``actuator_lag``
        seconds.  Until a command matures, the last applied command is held
        (hold-last policy, not coast-to-zero).
        """
        if self.status == VehicleStatus.BREAKDOWN:
            return
        if self.fuel.is_empty:
            self.status = VehicleStatus.BREAKDOWN
            self.state.speed = 0.0
            return

        self._current_time += dt

        # --- Actuator lag ---
        if self._actuator_lag > 0:
            # Push new command into buffer
            self._command_buffer.append((self._current_time, command))
            # Trim stale commands (older than lag + 0.5s safety margin)
            max_age = self._actuator_lag + 0.5
            while (self._command_buffer
                   and self._current_time - self._command_buffer[0][0] > max_age):
                self._command_buffer.popleft()
            # Pop matured commands (age >= actuator_lag)
            matured_cmd: VehicleCommand | None = None
            while (self._command_buffer
                   and self._current_time - self._command_buffer[0][0] >= self._actuator_lag):
                _, matured_cmd = self._command_buffer.popleft()
            if matured_cmd is not None:
                self._last_applied_cmd = matured_cmd
            # Use last applied command (hold-last); zero if nothing yet
            applied = self._last_applied_cmd if self._last_applied_cmd is not None else VehicleCommand()
        else:
            # No lag: apply immediately
            applied = command

        prev_pos = self.state.position().copy()

        # Apply kinematic step (weather + safe mode via effective_max_speed)
        self.state.step(
            accel=applied.accel,
            turn_rate=applied.turn_rate,
            dt=dt,
            max_speed=self.effective_max_speed,
            max_accel=self.vcfg.max_accel * self.weather_accel_factor,
            max_decel=self.vcfg.max_decel * self.weather_accel_factor,
            max_turn_rate=self.vcfg.max_turn_rate,
        )

        # Consume fuel (weather increases consumption in adverse conditions)
        consumed = self.fuel.consume(self.state.speed, dt) * self.weather_fuel_factor
        self.total_fuel_consumed += consumed

        # Track distance
        new_pos = self.state.position()
        self.total_distance += float(np.linalg.norm(new_pos - prev_pos))

        # Propagate estimator
        self.estimator.propagate(self.state.speed, self.state.heading, dt)

        # Store histories
        self.true_history.append((self.state.x, self.state.y))
        self.est_history.append((self.estimator.state.x, self.estimator.state.y))

    def set_breakdown(self) -> None:
        """Force vehicle into breakdown state.  Irreversible."""
        self.status = VehicleStatus.BREAKDOWN
        self.state.speed = 0.0
        self.is_leader = False

    def enter_safe_mode(self) -> None:
        """Enter safe mode (reduce speed, increase separation)."""
        if self.status == VehicleStatus.ACTIVE:
            self.status = VehicleStatus.SAFE_MODE

    def exit_safe_mode(self) -> None:
        """Exit safe mode back to active."""
        if self.status == VehicleStatus.SAFE_MODE:
            self.status = VehicleStatus.ACTIVE

    @property
    def effective_max_speed(self) -> float:
        """Max speed considering safe mode and weather."""
        base = self.vcfg.max_speed
        if self.status == VehicleStatus.SAFE_MODE:
            base *= self.config.coordination.safe_mode_speed_factor
        return base * self.weather_speed_factor

    @property
    def effective_spacing(self) -> float:
        """Formation spacing considering safe mode and weather."""
        base = self.config.coordination.formation_spacing
        if self.status == VehicleStatus.SAFE_MODE:
            base *= self.config.coordination.safe_mode_spacing_factor
        return base * self.weather_spacing_factor

    @property
    def is_operational(self) -> bool:
        return self.status in (VehicleStatus.ACTIVE, VehicleStatus.SAFE_MODE)

    def route_corridor_distance(self, x: float, y: float) -> float:
        """Min distance from (x, y) to the planned route polyline.

        Only checks a window of ±3 segments around ``current_waypoint_idx``
        so the query is O(1), not O(len(waypoints)).

        Returns 0.0 if no route is available.
        """
        if not self.waypoints:
            return 0.0
        lo = max(0, self.current_waypoint_idx - 3)
        hi = min(len(self.waypoints), self.current_waypoint_idx + 4)
        min_d = float("inf")
        for i in range(lo, hi - 1):
            d = _point_to_segment_dist(x, y, self.waypoints[i], self.waypoints[i + 1])
            if d < min_d:
                min_d = d
        # Also check distance to nearest waypoint directly
        for i in range(lo, hi):
            dx = x - self.waypoints[i][0]
            dy = y - self.waypoints[i][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_d:
                min_d = d
        return min_d if min_d < float("inf") else 0.0

    def has_reached_destination(self, threshold: float = 15.0) -> bool:
        """Check if vehicle has reached its assigned destination."""
        if self.assigned_destination is None:
            return False
        dx = self.state.x - self.assigned_destination[0]
        dy = self.state.y - self.assigned_destination[1]
        return (dx * dx + dy * dy) < threshold * threshold

    # ------------------------------------------------------------------
    # Neighbor state table (Phase 8)
    # ------------------------------------------------------------------

    def update_neighbor(self, sender_id: int, payload: dict[str, object], timestamp: float) -> None:
        """Cache a received STATE_BROADCAST in the neighbor table."""
        self.neighbor_states[sender_id] = {
            "x": payload.get("x", 0.0),
            "y": payload.get("y", 0.0),
            "heading": payload.get("heading", 0.0),
            "speed": payload.get("speed", 0.0),
            "uncertainty": payload.get("uncertainty", 0.0),
            "fuel": payload.get("fuel", 0.0),
            "status": payload.get("status", "ACTIVE"),
            "timestamp": timestamp,
        }

    def get_stale_neighbors(self, current_time: float, max_age: float) -> list[int]:
        """Return IDs of neighbors whose last update is older than *max_age*."""
        return [
            nid for nid, state in self.neighbor_states.items()
            if current_time - float(state["timestamp"]) > max_age
        ]

    def prune_stale_neighbors(self, current_time: float, max_age: float) -> int:
        """Remove stale entries from the neighbor table. Returns count removed."""
        stale = self.get_stale_neighbors(current_time, max_age)
        for nid in stale:
            del self.neighbor_states[nid]
        return len(stale)
