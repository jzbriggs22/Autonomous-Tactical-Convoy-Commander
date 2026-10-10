"""Threat detection via RSSI anomaly and collaborative triangulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from convoy_commander.ew.jammer import RFJammer


@dataclass
class BearingEstimate:
    """A single jammer bearing observation from one vehicle."""

    vehicle_id: int
    vehicle_x: float
    vehicle_y: float
    bearing_rad: float
    rssi_anomaly: float
    timestamp: float


class ThreatDetector:
    """Per-vehicle RSSI anomaly detection and collaborative triangulation."""

    def __init__(
        self,
        rng: np.random.Generator,
        detection_threshold: float = 0.1,
        bearing_noise_std: float = 0.1,
        estimate_ttl: float = 10.0,
    ) -> None:
        self.rng = rng
        self.detection_threshold = detection_threshold
        self.bearing_noise_std = bearing_noise_std
        self.estimate_ttl = estimate_ttl
        self.bearing_estimates: list[BearingEstimate] = []
        self.detected_threats: list[tuple[float, float, float]] = []  # (x, y, confidence)

    def measure_rssi_anomaly(
        self,
        vehicle_x: float,
        vehicle_y: float,
        jammers: list[RFJammer],
    ) -> float | None:
        """Return RSSI anomaly level if jammer detected, else None.

        Anomaly = sum of power_linear / d² for all jammers in range.
        """
        anomaly = 0.0
        for j in jammers:
            if not j.contains(vehicle_x, vehicle_y):
                continue
            dx = vehicle_x - j.x
            dy = vehicle_y - j.y
            d_sq = max(dx * dx + dy * dy, 1.0)
            power_linear = 10.0 ** (j.power_dbm / 10.0)
            anomaly += power_linear / d_sq
        if anomaly >= self.detection_threshold:
            return anomaly
        return None

    def estimate_bearing(
        self,
        vehicle_x: float,
        vehicle_y: float,
        jammers: list[RFJammer],
    ) -> float | None:
        """Return bearing estimate (rad) toward the strongest jammer.

        True bearing plus Gaussian noise with ``bearing_noise_std``.
        Returns None if no jammer is in range.
        """
        best_power = 0.0
        best_jammer: RFJammer | None = None
        for j in jammers:
            if not j.contains(vehicle_x, vehicle_y):
                continue
            dx = vehicle_x - j.x
            dy = vehicle_y - j.y
            d_sq = max(dx * dx + dy * dy, 1.0)
            power_linear = 10.0 ** (j.power_dbm / 10.0)
            eff = power_linear / d_sq
            if eff > best_power:
                best_power = eff
                best_jammer = j
        if best_jammer is None:
            return None
        true_bearing = math.atan2(best_jammer.y - vehicle_y, best_jammer.x - vehicle_x)
        noise = self.rng.normal(0.0, self.bearing_noise_std)
        return true_bearing + noise

    def add_estimate(self, est: BearingEstimate) -> None:
        """Add a bearing estimate (from self or received from peer)."""
        self.bearing_estimates.append(est)

    def try_triangulate(self, current_time: float) -> tuple[float, float] | None:
        """Attempt triangulation with current estimates.

        Prunes estimates older than ``estimate_ttl`` seconds.
        Requires at least 2 estimates from distinct positions.
        """
        # Prune stale
        cutoff = current_time - self.estimate_ttl
        self.bearing_estimates = [e for e in self.bearing_estimates if e.timestamp >= cutoff]

        if len(self.bearing_estimates) < 2:
            return None

        return self.triangulate(self.bearing_estimates)

    @staticmethod
    def triangulate(estimates: list[BearingEstimate]) -> tuple[float, float] | None:
        """Least-squares intersection of 2+ bearing lines.

        Each bearing line: ``(x_i, y_i) + t * (cos(b_i), sin(b_i))``.
        The perpendicular form gives the constraint::

            -sin(b_i) * X + cos(b_i) * Y = -sin(b_i) * x_i + cos(b_i) * y_i

        Stacked into ``Ax = b`` and solved with ``np.linalg.lstsq``.
        """
        if len(estimates) < 2:
            return None

        n = len(estimates)
        a_mat = np.zeros((n, 2))
        b_vec = np.zeros(n)

        for i, est in enumerate(estimates):
            s = math.sin(est.bearing_rad)
            c = math.cos(est.bearing_rad)
            a_mat[i, 0] = -s
            a_mat[i, 1] = c
            b_vec[i] = -s * est.vehicle_x + c * est.vehicle_y

        result, residuals, rank, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        if rank < 2:
            return None

        return (float(result[0]), float(result[1]))
