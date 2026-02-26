"""Position estimator with dead-reckoning, drift model, and occasional fixes.

SAFETY-CRITICAL ASSUMPTIONS:
  A1. Drift is modelled as additive Gaussian noise; real IMU errors are
      non-Gaussian (heavy-tailed, correlated).  The simulator therefore
      *underestimates* true worst-case drift.
  A2. The estimator now maintains a 2×2 position covariance matrix *P* and
      performs Extended-Kalman-Filter-style updates.  The scalar
      ``uncertainty`` field is retained as sqrt(trace(P)/2) for backward
      compatibility but the *primary* state representation is the
      covariance.
  A3. Fixes are applied with the true (ground-truth) position plus noise.
      In a real system the landmark detection itself can fail or be spoofed;
      this model does not capture that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from convoy_commander.core.config import EstimatorConfig
from convoy_commander.core.physics import KinematicState


# Maximum plausible uncertainty before we cap (prevents float overflow in long runs)
_MAX_UNCERTAINTY_CAP: float = 1e6


@dataclass
class EstimatorState:
    """Estimated position, heading, and 2×2 position covariance.

    The ``cov`` field is a 2×2 numpy array [[σ_xx, σ_xy], [σ_xy, σ_yy]].
    ``uncertainty`` is kept synchronised as sqrt(trace(cov)/2), which
    is the root-mean-square of the x and y standard deviations — a
    single scalar proxy used by the safe-mode logic and innovation gate.
    """

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    uncertainty: float = 0.0  # 1-sigma proxy: sqrt(trace(cov)/2)
    bias_x: float = 0.0  # accumulated drift bias
    bias_y: float = 0.0
    total_fixes_applied: int = 0  # audit counter
    cov: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.25)  # 2×2 covariance

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


def _cov_to_scalar(cov: np.ndarray) -> float:
    """Derive the backward-compatible scalar uncertainty from P."""
    return math.sqrt(max(0.0, 0.5 * float(cov[0, 0] + cov[1, 1])))


class PositionEstimator:
    """Dead-reckoning estimator with 2×2 covariance and optional absolute fixes.

    The estimator propagates position using odometry (speed + heading) but
    accumulates drift over time.  When a landmark or GPS fix is available,
    it performs a Kalman-style measurement update on both the position
    estimate and the 2×2 covariance matrix.
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
        self.state.cov = np.eye(2) * 0.25  # small initial covariance (0.5m std)
        self.state.uncertainty = _cov_to_scalar(self.state.cov)
        self.state.bias_x = 0.0
        self.state.bias_y = 0.0
        self.state.total_fixes_applied = 0

    def propagate(self, speed: float, heading: float, dt: float) -> None:
        """Dead-reckoning propagation with drift.

        Preconditions: dt > 0, speed >= 0.
        Postcondition: cov is symmetric positive semi-definite,
                       uncertainty >= 0 (capped at _MAX_UNCERTAINTY_CAP).

        The covariance prediction step is  P_{k+1} = F·P_k·Fᵀ + Q
        where F = I (position-only state, no velocity in the state vector)
        and Q models process noise from IMU drift and speed-proportional error.
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

        # Process noise covariance Q
        #
        # To maintain backward compatibility with the legacy scalar model
        # (which grew σ linearly per step rather than as √t), Q is derived
        # from the legacy per-step σ increment so that:
        #   σ_new = σ_old + Δσ   →   σ²_new = σ²_old + 2·σ_old·Δσ + Δσ²
        # The variance increment is therefore  Δ(σ²) = 2·σ·Δσ + Δσ².
        # This is state-dependent (like an adaptive filter) and is a valid
        # engineering model: larger uncertainty begets larger process noise.
        delta_sigma = self.config.drift_rate * math.sqrt(dt) + abs(speed) * 0.001 * dt
        current_sigma = _cov_to_scalar(self.state.cov)
        delta_var = 2.0 * current_sigma * delta_sigma + delta_sigma ** 2
        # Add heading-dependent noise structure:
        # More noise along the direction of travel, less perpendicular
        c, s = math.cos(heading), math.sin(heading)
        along_var = delta_var * (1.0 + abs(speed) * 0.01 * dt)
        cross_var = delta_var
        # Q = R · diag(along_var, cross_var) · Rᵀ
        Q = np.array([
            [c * c * along_var + s * s * cross_var, c * s * (along_var - cross_var)],
            [c * s * (along_var - cross_var), s * s * along_var + c * c * cross_var],
        ])

        # P = F·P·Fᵀ + Q  (F = I for position-only state)
        self.state.cov = self.state.cov + Q

        # Cap covariance to prevent overflow
        max_var = _MAX_UNCERTAINTY_CAP ** 2
        self.state.cov = np.clip(self.state.cov, -max_var, max_var)

        # Sync scalar uncertainty
        self.state.uncertainty = min(
            _cov_to_scalar(self.state.cov), _MAX_UNCERTAINTY_CAP
        )

    def apply_landmark_fix(self, true_x: float, true_y: float) -> tuple[float, bool]:
        """Apply a landmark fix (noisy absolute measurement) via Kalman update.

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

        # Innovation vector and scalar magnitude
        innov_vec = np.array([measured_x - self.state.x, measured_y - self.state.y])
        innovation = float(np.linalg.norm(innov_vec))

        # Innovation gate (scalar, backward-compatible)
        if self.config.innovation_gate_sigma > 0:
            gate = self.config.innovation_gate_sigma * max(
                self.state.uncertainty, self.config.landmark_fix_std
            )
            if innovation > gate:
                return innovation, False  # Rejected — anomalous fix

        # Kalman update:  H = I,  R = diag(σ², σ²)
        R = np.eye(2) * (self.config.landmark_fix_std ** 2)
        P = self.state.cov
        S = P + R  # Innovation covariance
        K = P @ np.linalg.solve(S, np.eye(2))  # Kalman gain: P · S⁻¹

        # State update
        self.state.x += float(K[0] @ innov_vec)
        self.state.y += float(K[1] @ innov_vec)

        # Covariance update: P = (I - K·H)·P  (Joseph form for stability)
        IKH = np.eye(2) - K
        self.state.cov = IKH @ P @ IKH.T + K @ R @ K.T

        # Partially correct bias (proportional to avg gain)
        avg_gain = 0.5 * float(K[0, 0] + K[1, 1])
        self.state.bias_x *= (1 - avg_gain * 0.5)
        self.state.bias_y *= (1 - avg_gain * 0.5)

        # Sync scalar
        self.state.uncertainty = _cov_to_scalar(self.state.cov)
        self.state.total_fixes_applied += 1
        return innovation, True

    def apply_gps_fix(self, true_x: float, true_y: float) -> tuple[float, bool]:
        """Apply a GPS fix (less noisy than landmark) via Kalman update.

        Returns (innovation, accepted).

        Innovation gating (SAFETY): rejects fixes whose innovation exceeds
        gate_sigma * max(uncertainty, gps_fix_std).  This is the primary
        defence against GPS spoofing attacks.
        """
        noise_x = self.rng.normal(0, self.config.gps_fix_std)
        noise_y = self.rng.normal(0, self.config.gps_fix_std)
        measured_x = true_x + noise_x
        measured_y = true_y + noise_y

        innov_vec = np.array([measured_x - self.state.x, measured_y - self.state.y])
        innovation = float(np.linalg.norm(innov_vec))

        # Innovation gate
        if self.config.innovation_gate_sigma > 0:
            gate = self.config.innovation_gate_sigma * max(
                self.state.uncertainty, self.config.gps_fix_std
            )
            if innovation > gate:
                return innovation, False  # Rejected — spoofed or faulty fix

        # Kalman update:  H = I,  R = diag(σ², σ²)
        R = np.eye(2) * (self.config.gps_fix_std ** 2)
        P = self.state.cov
        S = P + R
        K = P @ np.linalg.solve(S, np.eye(2))

        self.state.x += float(K[0] @ innov_vec)
        self.state.y += float(K[1] @ innov_vec)

        # Joseph-form covariance update
        IKH = np.eye(2) - K
        self.state.cov = IKH @ P @ IKH.T + K @ R @ K.T

        # Reset bias proportional to gain
        avg_gain = 0.5 * float(K[0, 0] + K[1, 1])
        self.state.bias_x *= (1 - avg_gain * 0.8)
        self.state.bias_y *= (1 - avg_gain * 0.8)

        self.state.uncertainty = _cov_to_scalar(self.state.cov)
        self.state.total_fixes_applied += 1
        return innovation, True

    def apply_drift_spike(self, magnitude: float) -> None:
        """Inject a sudden IMU bias spike (models vibration, shock, or sensor corruption).

        Adds a random-direction bias of ``magnitude`` m/s to the accumulated
        drift bias and increases both the covariance and scalar uncertainty.

        SAFETY NOTE: This is a worst-case instantaneous jump; real sensor
        faults may be gradual or oscillatory.
        """
        angle = self.rng.uniform(0, 2 * math.pi)
        self.state.bias_x += magnitude * math.cos(angle)
        self.state.bias_y += magnitude * math.sin(angle)
        # Add spike to covariance (isotropic perturbation)
        spike_var = (magnitude * 2.0) ** 2
        self.state.cov = self.state.cov + np.eye(2) * spike_var
        # Cap
        max_var = _MAX_UNCERTAINTY_CAP ** 2
        self.state.cov = np.clip(self.state.cov, -max_var, max_var)
        # Sync scalar
        self.state.uncertainty = min(
            _cov_to_scalar(self.state.cov), _MAX_UNCERTAINTY_CAP
        )

    @property
    def cov_eigenvalues(self) -> tuple[float, float]:
        """Return eigenvalues of the 2×2 covariance (major, minor axis variances).

        Useful for error-ellipse visualisation.
        """
        eigs = np.linalg.eigvalsh(self.state.cov)
        return (float(max(eigs)), float(min(eigs)))

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
