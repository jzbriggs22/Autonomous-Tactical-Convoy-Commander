"""Vehicle agent with dynamics, fuel, and estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from convoy_commander.core.config import SimConfig
from convoy_commander.core.physics import FuelState, KinematicState
from convoy_commander.vehicles.estimator import PositionEstimator


class VehicleStatus(Enum):
    ACTIVE = auto()
    SAFE_MODE = auto()
    BREAKDOWN = auto()
    ARRIVED = auto()


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
        self.is_leader = False
        self.leader_id: int | None = None
        self.safe_mode_timer: float = 0.0
        self.last_comms_time: float = 0.0

        # Planning
        self.waypoints: list[tuple[float, float]] = []
        self.current_waypoint_idx: int = 0
        self.assigned_destination: tuple[float, float] | None = None

        # Metrics
        self.total_distance: float = 0.0
        self.near_miss_count: int = 0
        self.collision_count: int = 0

    def step(self, command: VehicleCommand, dt: float) -> None:
        """Advance vehicle by one timestep."""
        if self.status == VehicleStatus.BREAKDOWN:
            return
        if self.fuel.is_empty:
            self.status = VehicleStatus.BREAKDOWN
            return

        prev_pos = self.state.position().copy()

        # Apply kinematic step
        self.state.step(
            accel=command.accel,
            turn_rate=command.turn_rate,
            dt=dt,
            max_speed=self.vcfg.max_speed,
            max_accel=self.vcfg.max_accel,
            max_decel=self.vcfg.max_decel,
            max_turn_rate=self.vcfg.max_turn_rate,
        )

        # Consume fuel
        self.fuel.consume(self.state.speed, dt)

        # Track distance
        new_pos = self.state.position()
        self.total_distance += float(np.linalg.norm(new_pos - prev_pos))

        # Propagate estimator
        self.estimator.propagate(self.state.speed, self.state.heading, dt)

        # Store histories
        self.true_history.append((self.state.x, self.state.y))
        self.est_history.append((self.estimator.state.x, self.estimator.state.y))

    def set_breakdown(self) -> None:
        """Force vehicle into breakdown state."""
        self.status = VehicleStatus.BREAKDOWN
        self.state.speed = 0.0

    def enter_safe_mode(self) -> None:
        """Enter safe mode (reduce speed, increase separation)."""
        self.status = VehicleStatus.SAFE_MODE

    def exit_safe_mode(self) -> None:
        """Exit safe mode back to active."""
        if self.status == VehicleStatus.SAFE_MODE:
            self.status = VehicleStatus.ACTIVE

    @property
    def effective_max_speed(self) -> float:
        """Max speed considering safe mode."""
        if self.status == VehicleStatus.SAFE_MODE:
            return self.vcfg.max_speed * self.config.coordination.safe_mode_speed_factor
        return self.vcfg.max_speed

    @property
    def effective_spacing(self) -> float:
        """Formation spacing considering safe mode."""
        base = self.config.coordination.formation_spacing
        if self.status == VehicleStatus.SAFE_MODE:
            return base * self.config.coordination.safe_mode_spacing_factor
        return base

    @property
    def is_operational(self) -> bool:
        return self.status in (VehicleStatus.ACTIVE, VehicleStatus.SAFE_MODE)

    def has_reached_destination(self, threshold: float = 15.0) -> bool:
        """Check if vehicle has reached its assigned destination."""
        if self.assigned_destination is None:
            return False
        dx = self.state.x - self.assigned_destination[0]
        dy = self.state.y - self.assigned_destination[1]
        return (dx * dx + dy * dy) < threshold * threshold
