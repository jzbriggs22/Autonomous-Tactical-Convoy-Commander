"""Position estimator with dead-reckoning, drift model, and occasional fixes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from convoy_commander.core.config import EstimatorConfig
from convoy_commander.core.physics import KinematicState


@dataclass
class EstimatorState:
    """Estimated position and uncertainty."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    uncertainty: float = 0.0  # 1-sigma position uncertainty (m)
    bias_x: float = 0.0  # accumulated drift bias
    bias_y: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


class PositionEstimator:
    """Dead-reckoning estimator with drift and optional absolute fixes.

    The estimator propagates position using odometry (speed + heading) but
    accumulates drift over time.  When a landmark or GPS fix is available,
    it snaps the estimate toward the true position with noise.
    """

    def __init__(self, config: EstimatorConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.state = EstimatorState()

    def initialize(self, true_state: KinematicState) -> None:
        """Initialize estimator at known position."""
        self.state.x = true_state.x
        self.state.y = true_state.y
        self.state.heading = true_state.heading
        self.state.uncertainty = 0.5  # small initial uncertainty
        self.state.bias_x = 0.0
        self.state.bias_y = 0.0

    def propagate(self, speed: float, heading: float, dt: float) -> None:
        """Dead-reckoning propagation with drift."""
        # Bias random walk
        self.state.bias_x += self.rng.normal(0, self.config.drift_bias_rate * dt)
        self.state.bias_y += self.rng.normal(0, self.config.drift_bias_rate * dt)

        # Noisy odometry
        noise_x = self.rng.normal(0, self.config.drift_rate * dt)
        noise_y = self.rng.normal(0, self.config.drift_rate * dt)

        dx = speed * math.cos(heading) * dt + self.state.bias_x * dt + noise_x
        dy = speed * math.sin(heading) * dt + self.state.bias_y * dt + noise_y

        self.state.x += dx
        self.state.y += dy
        self.state.heading = heading

        # Grow uncertainty
        self.state.uncertainty += self.config.drift_rate * math.sqrt(dt) + abs(speed) * 0.001 * dt

    def apply_landmark_fix(self, true_x: float, true_y: float) -> None:
        """Apply a landmark fix (noisy absolute measurement)."""
        noise_x = self.rng.normal(0, self.config.landmark_fix_std)
        noise_y = self.rng.normal(0, self.config.landmark_fix_std)
        measured_x = true_x + noise_x
        measured_y = true_y + noise_y

        # Simple complementary filter: blend toward measurement
        # Weight depends on current uncertainty vs measurement noise
        meas_var = self.config.landmark_fix_std ** 2
        est_var = self.state.uncertainty ** 2
        gain = est_var / (est_var + meas_var) if (est_var + meas_var) > 0 else 0.5

        self.state.x += gain * (measured_x - self.state.x)
        self.state.y += gain * (measured_y - self.state.y)
        self.state.uncertainty = math.sqrt((1 - gain) * est_var)

        # Partially correct bias
        self.state.bias_x *= (1 - gain * 0.5)
        self.state.bias_y *= (1 - gain * 0.5)

    def apply_gps_fix(self, true_x: float, true_y: float) -> None:
        """Apply a GPS fix (less noisy than landmark)."""
        noise_x = self.rng.normal(0, self.config.gps_fix_std)
        noise_y = self.rng.normal(0, self.config.gps_fix_std)
        measured_x = true_x + noise_x
        measured_y = true_y + noise_y

        meas_var = self.config.gps_fix_std ** 2
        est_var = self.state.uncertainty ** 2
        gain = est_var / (est_var + meas_var) if (est_var + meas_var) > 0 else 0.5

        self.state.x += gain * (measured_x - self.state.x)
        self.state.y += gain * (measured_y - self.state.y)
        self.state.uncertainty = math.sqrt((1 - gain) * est_var)

        # Reset bias on GPS
        self.state.bias_x *= (1 - gain * 0.8)
        self.state.bias_y *= (1 - gain * 0.8)

    @property
    def is_uncertain(self) -> bool:
        """Check if uncertainty exceeds safe threshold."""
        return self.state.uncertainty > self.config.uncertainty_safe_threshold
