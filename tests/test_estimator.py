"""Tests for position estimator."""

import numpy as np

from convoy_commander.core.config import EstimatorConfig
from convoy_commander.core.physics import KinematicState
from convoy_commander.vehicles.estimator import PositionEstimator


def test_estimator_drift_increases():
    """Without fixes, uncertainty should grow over time."""
    config = EstimatorConfig(drift_rate=0.1, drift_bias_rate=0.01)
    rng = np.random.default_rng(42)
    est = PositionEstimator(config, rng)
    est.initialize(KinematicState(x=100, y=100, heading=0, speed=5))

    initial_uncertainty = est.state.uncertainty

    # Propagate for many steps without fixes
    for _ in range(500):
        est.propagate(speed=5.0, heading=0.0, dt=0.1)

    assert est.state.uncertainty > initial_uncertainty
    assert est.state.uncertainty > 1.0  # Should have grown significantly


def test_estimator_fix_reduces_uncertainty():
    """Applying a fix should reduce uncertainty."""
    config = EstimatorConfig(drift_rate=0.1, landmark_fix_std=1.0)
    rng = np.random.default_rng(42)
    est = PositionEstimator(config, rng)
    est.initialize(KinematicState(x=100, y=100, heading=0, speed=5))

    # Let drift grow
    for _ in range(200):
        est.propagate(speed=5.0, heading=0.0, dt=0.1)

    before_fix = est.state.uncertainty
    est.apply_landmark_fix(true_x=est.state.x, true_y=est.state.y)
    after_fix = est.state.uncertainty

    assert after_fix < before_fix


def test_estimator_gps_fix():
    """GPS fix should also reduce uncertainty."""
    config = EstimatorConfig(drift_rate=0.1, gps_fix_std=0.5)
    rng = np.random.default_rng(42)
    est = PositionEstimator(config, rng)
    est.initialize(KinematicState(x=100, y=100, heading=0, speed=5))

    for _ in range(200):
        est.propagate(speed=5.0, heading=0.0, dt=0.1)

    before_fix = est.state.uncertainty
    est.apply_gps_fix(true_x=est.state.x, true_y=est.state.y)
    assert est.state.uncertainty < before_fix


def test_estimator_is_uncertain():
    """Estimator should flag high uncertainty."""
    config = EstimatorConfig(drift_rate=0.5, uncertainty_safe_threshold=5.0)
    rng = np.random.default_rng(42)
    est = PositionEstimator(config, rng)
    est.initialize(KinematicState(x=100, y=100, heading=0, speed=5))

    assert not est.is_uncertain

    for _ in range(500):
        est.propagate(speed=5.0, heading=0.0, dt=0.1)

    assert est.is_uncertain
