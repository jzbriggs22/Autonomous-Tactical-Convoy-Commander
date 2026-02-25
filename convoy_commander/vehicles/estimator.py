"""Position estimator with dead-reckoning, drift model, and occasional fixes.

SAFETY-CRITICAL ASSUMPTIONS:
  A1. Drift is modelled as additive Gaussian noise; real IMU errors are
      non-Gaussian (heavy-tailed, correlated).  The simulator therefore
      *underestimates* true worst-case drift.
  A2. The complementary filter is a single-gain approximation to a Kalman
      filter.  It does not maintain a full covariance matrix.
  A3. ``uncertainty`` is a scalar 1-sigma proxy, not a full error ellipse.
      It is *optimistic* in the cross-track direction.
  A4. Fixes are applied with the true (ground-truth) position plus noise.
      In a real system the landmark detection itself can fail or be spoofed;
      this model does not capture that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from convoy_commander.core.config import EstimatorConfig
from convoy_commander.core.physics import KinematicState


# Maximum plausible uncertainty before we cap (prevents float overflow in long runs)
_MAX_UNCERTAINTY_CAP: float = 1e6


@dataclass
class EstimatorState:
    """Estimated position and uncertainty."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    uncertainty: float = 0.0  # 1-sigma position uncertainty (m)
    bias_x: float = 0.0  # accumulated drift bias
    bias_y: float = 0.0
    total_fixes_applied: int = 0  # audit counter

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
        self._prev_uncertainty_high = False  # for edge-detect logging

    def initialize(self, true_state: KinematicState) -> None:
        """Initialize estimator at known position."""
        self.state.x = true_state.x
        self.state.y = true_state.y
        self.state.heading = true_state.heading
        self.state.uncertainty = 0.5  # small initial uncertainty
        self.state.bias_x = 0.0
        self.state.bias_y = 0.0
        self.state.total_fixes_applied = 0

    def propagate(self, speed: float, heading: float, dt: float) -> None:
        """Dead-reckoning propagation with drift.

        Preconditions: dt > 0, speed >= 0.
        Postcondition: uncertainty >= 0 (capped at _MAX_UNCERTAINTY_CAP).
        """
        if dt <= 0:
            return
        if speed < 0:
            speed = 0.0  # Defensive

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
        # Cap to prevent overflow
        self.state.uncertainty = min(self.state.uncertainty, _MAX_UNCERTAINTY_CAP)

    def apply_landmark_fix(self, true_x: float, true_y: float) -> tuple[float, bool]:
        """Apply a landmark fix (noisy absolute measurement).

        Returns (innovation, accepted):
          - innovation: pre-update residual magnitude (m)
          - accepted: True if the fix passed the innovation gate

        Innovation gating (SAFETY):  if ``innovation_gate_sigma > 0``, fixes
        whose innovation exceeds gate_sigma * max(uncertainty, landmark_fix_std)
        are rejected without updating the estimate.  This prevents a spoofed
        or faulty landmark from pulling the estimate to a wrong location.
        """
        noise_x = self.rng.normal(0, self.config.landmark_fix_std)
        noise_y = self.rng.normal(0, self.config.landmark_fix_std)
        measured_x = true_x + noise_x
        measured_y = true_y + noise_y

        # Innovation (pre-update residual)
        innovation = math.hypot(measured_x - self.state.x, measured_y - self.state.y)

        # Innovation gate
        if self.config.innovation_gate_sigma > 0:
            gate = self.config.innovation_gate_sigma * max(
                self.state.uncertainty, self.config.landmark_fix_std
            )
            if innovation > gate:
                return innovation, False  # Rejected — anomalous fix

        # Complementary filter gain
        meas_var = self.config.landmark_fix_std ** 2
        est_var = self.state.uncertainty ** 2
        gain = est_var / (est_var + meas_var) if (est_var + meas_var) > 0 else 0.5

        self.state.x += gain * (measured_x - self.state.x)
        self.state.y += gain * (measured_y - self.state.y)
        self.state.uncertainty = math.sqrt(max(0.0, (1 - gain) * est_var))

        # Partially correct bias
        self.state.bias_x *= (1 - gain * 0.5)
        self.state.bias_y *= (1 - gain * 0.5)

        self.state.total_fixes_applied += 1
        return innovation, True

    def apply_gps_fix(self, true_x: float, true_y: float) -> tuple[float, bool]:
        """Apply a GPS fix (less noisy than landmark).

        Returns (innovation, accepted).

        Innovation gating (SAFETY): rejects fixes whose innovation exceeds
        gate_sigma * max(uncertainty, gps_fix_std).  This is the primary
        defence against GPS spoofing attacks.
        """
        noise_x = self.rng.normal(0, self.config.gps_fix_std)
        noise_y = self.rng.normal(0, self.config.gps_fix_std)
        measured_x = true_x + noise_x
        measured_y = true_y + noise_y

        innovation = math.hypot(measured_x - self.state.x, measured_y - self.state.y)

        # Innovation gate
        if self.config.innovation_gate_sigma > 0:
            gate = self.config.innovation_gate_sigma * max(
                self.state.uncertainty, self.config.gps_fix_std
            )
            if innovation > gate:
                return innovation, False  # Rejected — spoofed or faulty fix

        meas_var = self.config.gps_fix_std ** 2
        est_var = self.state.uncertainty ** 2
        gain = est_var / (est_var + meas_var) if (est_var + meas_var) > 0 else 0.5

        self.state.x += gain * (measured_x - self.state.x)
        self.state.y += gain * (measured_y - self.state.y)
        self.state.uncertainty = math.sqrt(max(0.0, (1 - gain) * est_var))

        # Reset bias on GPS
        self.state.bias_x *= (1 - gain * 0.8)
        self.state.bias_y *= (1 - gain * 0.8)

        self.state.total_fixes_applied += 1
        return innovation, True

    @property
    def is_uncertain(self) -> bool:
        """Check if uncertainty exceeds safe threshold."""
        return self.state.uncertainty > self.config.uncertainty_safe_threshold

    def uncertainty_just_exceeded(self) -> bool:
        """Edge-detect: returns True once when uncertainty first crosses threshold."""
        currently_high = self.is_uncertain
        result = currently_high and not self._prev_uncertainty_high
        self._prev_uncertainty_high = currently_high
        return result

    def uncertainty_just_recovered(self) -> bool:
        """Edge-detect: returns True once when uncertainty drops back below threshold."""
        currently_high = self.is_uncertain
        result = not currently_high and self._prev_uncertainty_high
        self._prev_uncertainty_high = currently_high
        return result
