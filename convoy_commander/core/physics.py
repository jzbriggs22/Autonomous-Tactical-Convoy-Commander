"""Kinematic state and physics utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class KinematicState:
    """2D kinematic state for a vehicle."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians, 0 = east
    speed: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def forward_vec(self) -> np.ndarray:
        return np.array([math.cos(self.heading), math.sin(self.heading)])

    def step(
        self,
        accel: float,
        turn_rate: float,
        dt: float,
        max_speed: float,
        max_accel: float,
        max_decel: float,
        max_turn_rate: float,
    ) -> None:
        """Integrate one timestep with kinematic constraints."""
        accel = clamp(accel, -max_decel, max_accel)
        turn_rate = clamp(turn_rate, -max_turn_rate, max_turn_rate)

        self.speed = clamp(self.speed + accel * dt, 0.0, max_speed)
        self.heading = normalize_angle(self.heading + turn_rate * dt)
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

    def distance_to(self, other: KinematicState) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def copy(self) -> KinematicState:
        return KinematicState(x=self.x, y=self.y, heading=self.heading, speed=self.speed)


@dataclass
class FuelState:
    """Fuel / battery model."""

    fuel: float = 100.0
    capacity: float = 100.0
    rate_idle: float = 0.01
    rate_per_speed: float = 0.005

    def consume(self, speed: float, dt: float) -> None:
        usage = (self.rate_idle + self.rate_per_speed * speed) * dt
        self.fuel = max(0.0, self.fuel - usage)

    @property
    def is_empty(self) -> bool:
        return self.fuel <= 0.0

    @property
    def fraction(self) -> float:
        return self.fuel / self.capacity if self.capacity > 0 else 0.0
