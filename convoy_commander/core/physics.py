"""Kinematic state and physics utilities.

SAFETY-CRITICAL ASSUMPTIONS:
  A1. All integration uses first-order Euler.  This is acceptable at dt <= 0.1s
      for vehicle speeds <= 15 m/s (max displacement per step < 1.5 m).
  A2. Speed is non-negative (no reverse motion).  Negative accel decelerates
      toward zero; it does not command reverse.
  A3. Heading is normalised to [-pi, pi] after every step to prevent
      floating-point drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi].

    Precondition: lo <= hi.
    """
    if lo > hi:
        raise ValueError(f"clamp: lo ({lo}) > hi ({hi})")
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
    """2D kinematic state for a vehicle.

    Invariants maintained after every ``step()``:
      - speed >= 0
      - heading in [-pi, pi]
    """

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
        """Integrate one timestep with kinematic constraints.

        Preconditions:
          dt > 0, max_speed > 0, max_accel > 0, max_decel > 0, max_turn_rate > 0
        Postconditions:
          0 <= speed <= max_speed, heading in [-pi, pi]
        """
        if dt <= 0:
            raise ValueError(f"KinematicState.step: dt must be > 0, got {dt}")
        if max_speed <= 0 or max_accel <= 0 or max_decel <= 0 or max_turn_rate <= 0:
            raise ValueError("KinematicState.step: all max_* parameters must be > 0")

        # Reject NaN inputs (indicates upstream bug)
        if math.isnan(accel) or math.isnan(turn_rate):
            raise ValueError(f"KinematicState.step: NaN command (accel={accel}, turn_rate={turn_rate})")

        accel = clamp(accel, -max_decel, max_accel)
        turn_rate = clamp(turn_rate, -max_turn_rate, max_turn_rate)

        self.speed = clamp(self.speed + accel * dt, 0.0, max_speed)
        self.heading = normalize_angle(self.heading + turn_rate * dt)
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

        # Post-condition enforcement (belt-and-braces)
        if self.speed < 0:
            self.speed = 0.0

    def distance_to(self, other: KinematicState) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def copy(self) -> KinematicState:
        return KinematicState(x=self.x, y=self.y, heading=self.heading, speed=self.speed)


@dataclass
class FuelState:
    """Fuel / battery model.

    Invariants:
      - 0 <= fuel <= capacity
      - consumption is monotonically non-increasing (fuel never increases)
    """

    fuel: float = 100.0
    capacity: float = 100.0
    rate_idle: float = 0.01
    rate_per_speed: float = 0.005

    def consume(self, speed: float, dt: float, factor: float = 1.0) -> float:
        """Consume fuel for one timestep.  Returns fuel consumed.

        ``factor`` scales consumption (e.g. adverse-weather penalty) and is
        applied before the tank is debited, so the returned value and the
        remaining fuel always agree.

        Preconditions: speed >= 0, dt > 0, factor >= 0.
        """
        if speed < 0:
            speed = 0.0  # Defensive: negative speed should not generate fuel
        if dt <= 0:
            return 0.0
        if factor < 0:
            factor = 0.0  # Defensive: a negative factor must not add fuel
        usage = (self.rate_idle + self.rate_per_speed * speed) * dt * factor
        actual = min(usage, self.fuel)  # Cannot consume more than remaining
        self.fuel = max(0.0, self.fuel - actual)
        return actual

    @property
    def is_empty(self) -> bool:
        return self.fuel <= 0.0

    @property
    def fraction(self) -> float:
        return self.fuel / self.capacity if self.capacity > 0 else 0.0

    @property
    def is_low(self) -> bool:
        """Below 20% — conservative threshold for operational warnings."""
        return self.fraction < 0.20
