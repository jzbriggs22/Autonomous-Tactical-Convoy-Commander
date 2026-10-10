"""RF jammer model with SNR-based comms degradation and GPS denial."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RFJammer:
    """RF jammer placed in the world.

    Circular effect region with inverse-square-law SNR degradation.
    Optionally jams GPS fixes within radius.  Can be mobile (position
    updated each tick via ``step()``).
    """

    x: float
    y: float
    radius: float
    power_dbm: float = 30.0
    jam_gps: bool = False
    mobile: bool = False
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    # World bounds for clamping mobile jammers (set by runner)
    _world_w: float = 1000.0
    _world_h: float = 1000.0

    def contains(self, px: float, py: float) -> bool:
        """True if point is inside jammer radius."""
        dx = px - self.x
        dy = py - self.y
        return dx * dx + dy * dy < self.radius * self.radius

    def snr_degradation(self, px: float, py: float) -> float:
        """Return loss multiplier [1.0, inf) based on inverse-square-law SNR.

        At distance *d* from jammer centre::

            degradation = 1.0 + power_linear / max(d², 1.0)

        Returns 1.0 (no effect) if outside radius.
        """
        if not self.contains(px, py):
            return 1.0
        dx = px - self.x
        dy = py - self.y
        d_sq = max(dx * dx + dy * dy, 1.0)
        power_linear = 10.0 ** (self.power_dbm / 10.0)
        return 1.0 + power_linear / d_sq

    def gps_denied_at(self, px: float, py: float) -> bool:
        """True if GPS is jammed at this position."""
        return self.jam_gps and self.contains(px, py)

    def step(self, dt: float) -> None:
        """Advance position if mobile.  Clamps to world bounds."""
        if not self.mobile:
            return
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
        # Clamp and bounce off walls
        if self.x < 0:
            self.x = 0.0
            self.velocity_x = abs(self.velocity_x)
        elif self.x > self._world_w:
            self.x = self._world_w
            self.velocity_x = -abs(self.velocity_x)
        if self.y < 0:
            self.y = 0.0
            self.velocity_y = abs(self.velocity_y)
        elif self.y > self._world_h:
            self.y = self._world_h
            self.velocity_y = -abs(self.velocity_y)
