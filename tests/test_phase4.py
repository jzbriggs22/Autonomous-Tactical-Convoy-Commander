"""Phase 4 feature tests: 2×2 covariance EKF, CBBA-lite auction, viz updates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from convoy_commander.core.config import EstimatorConfig, SimConfig
from convoy_commander.core.physics import KinematicState
from convoy_commander.coordination.cbba import (
    cbba_allocate,
    compute_formation_slots,
    _compute_bid,
)
from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.scenarios import get_scenario
from convoy_commander.vehicles.estimator import (
    EstimatorState,
    PositionEstimator,
    _cov_to_scalar,
    _MAX_UNCERTAINTY_CAP,
)


# ===========================================================================
# 1. 2×2 Covariance EKF
# ===========================================================================


class TestCovarianceInit:
    """EstimatorState and PositionEstimator initialisation with covariance."""

    def test_state_default_cov_is_2x2(self):
        st = EstimatorState()
        assert st.cov.shape == (2, 2)
        assert np.allclose(st.cov, np.eye(2) * 0.25)

    def test_cov_to_scalar_identity(self):
        """Identity * σ² → scalar = σ."""
        cov = np.eye(2) * 4.0
        assert abs(_cov_to_scalar(cov) - 2.0) < 1e-9

    def test_cov_to_scalar_zero(self):
        cov = np.zeros((2, 2))
        assert _cov_to_scalar(cov) == 0.0

    def test_estimator_initialize_sets_cov(self):
        cfg = EstimatorConfig()
        est = PositionEstimator(cfg, np.random.default_rng(0))
        ks = KinematicState(x=10.0, y=20.0, heading=0.5, speed=0.0)
        est.initialize(ks)
        assert est.state.cov.shape == (2, 2)
        assert abs(est.state.uncertainty - _cov_to_scalar(np.eye(2) * 0.25)) < 1e-9


class TestCovariancePropagate:
    """Propagation grows the covariance matrix."""

    def _make(self, seed: int = 0) -> PositionEstimator:
        cfg = EstimatorConfig(drift_rate=0.05, drift_bias_rate=0.002)
        est = PositionEstimator(cfg, np.random.default_rng(seed))
        ks = KinematicState(x=0.0, y=0.0, heading=0.0, speed=0.0)
        est.initialize(ks)
        return est

    def test_propagate_grows_covariance(self):
        est = self._make()
        cov_before = est.state.cov.copy()
        est.propagate(5.0, 0.0, 0.1)
        # Covariance trace should increase (process noise added)
        assert np.trace(est.state.cov) > np.trace(cov_before)

    def test_propagate_cov_stays_symmetric(self):
        est = self._make()
        for _ in range(100):
            est.propagate(8.0, 0.5, 0.1)
        P = est.state.cov
        assert abs(P[0, 1] - P[1, 0]) < 1e-10

    def test_propagate_uncertainty_syncs_with_cov(self):
        est = self._make()
        for _ in range(50):
            est.propagate(10.0, 0.3, 0.1)
        expected_unc = _cov_to_scalar(est.state.cov)
        assert abs(est.state.uncertainty - expected_unc) < 1e-6

    def test_propagate_heading_dependent_noise(self):
        """Propagating along x (heading=0) should produce more variance in xx than yy."""
        est = self._make()
        for _ in range(100):
            est.propagate(10.0, 0.0, 0.1)
        P = est.state.cov
        # Along x, speed-dependent noise is larger → σ_xx > σ_yy
        assert P[0, 0] > P[1, 1]


class TestCovarianceKalmanUpdate:
    """Kalman measurement updates reduce covariance correctly."""

    def _make(self, seed: int = 0) -> PositionEstimator:
        cfg = EstimatorConfig(
            drift_rate=0.05,
            gps_fix_std=0.5,
            landmark_fix_std=1.0,
            innovation_gate_sigma=5.0,
        )
        est = PositionEstimator(cfg, np.random.default_rng(seed))
        ks = KinematicState(x=0.0, y=0.0, heading=0.0, speed=0.0)
        est.initialize(ks)
        return est

    def test_gps_fix_reduces_covariance(self):
        est = self._make()
        # Grow covariance first
        for _ in range(50):
            est.propagate(5.0, 0.0, 0.1)
        cov_before = est.state.cov.copy()
        est.apply_gps_fix(est.state.x, est.state.y)
        assert np.trace(est.state.cov) < np.trace(cov_before)

    def test_landmark_fix_reduces_covariance(self):
        est = self._make()
        for _ in range(50):
            est.propagate(5.0, 0.0, 0.1)
        cov_before = est.state.cov.copy()
        est.apply_landmark_fix(est.state.x, est.state.y)
        assert np.trace(est.state.cov) < np.trace(cov_before)

    def test_gps_fix_cov_stays_positive_semidefinite(self):
        est = self._make()
        for _ in range(50):
            est.propagate(8.0, 0.5, 0.1)
        est.apply_gps_fix(est.state.x + 1.0, est.state.y + 1.0)
        eigs = np.linalg.eigvalsh(est.state.cov)
        assert all(e >= -1e-10 for e in eigs)

    def test_repeated_fixes_converge(self):
        """After many fixes, uncertainty should approach the measurement noise floor."""
        est = self._make()
        for _ in range(20):
            est.propagate(5.0, 0.0, 0.1)
            est.apply_gps_fix(0.0, 0.0)
        # Should be close to GPS noise std (0.5m)
        assert est.state.uncertainty < 2.0

    def test_rejected_fix_does_not_change_cov(self):
        """Innovation gate rejection should preserve covariance."""
        est = self._make()
        est.state.cov = np.eye(2) * 1.0  # moderate uncertainty
        est.state.uncertainty = _cov_to_scalar(est.state.cov)
        cov_before = est.state.cov.copy()
        # 100m offset → innovation >> gate → rejected
        _, accepted = est.apply_gps_fix(100.0, 0.0)
        assert not accepted
        assert np.allclose(est.state.cov, cov_before)


class TestCovarianceDriftSpike:
    """Drift spike should inflate covariance."""

    def test_spike_inflates_cov(self):
        cfg = EstimatorConfig()
        est = PositionEstimator(cfg, np.random.default_rng(0))
        ks = KinematicState(x=0.0, y=0.0, heading=0.0, speed=0.0)
        est.initialize(ks)
        cov_before = est.state.cov.copy()
        est.apply_drift_spike(5.0)
        assert np.trace(est.state.cov) > np.trace(cov_before)

    def test_spike_cov_matches_uncertainty(self):
        cfg = EstimatorConfig()
        est = PositionEstimator(cfg, np.random.default_rng(0))
        est.initialize(KinematicState())
        est.apply_drift_spike(10.0)
        assert abs(est.state.uncertainty - _cov_to_scalar(est.state.cov)) < 1e-6


class TestCovEigenvalues:
    """cov_eigenvalues property for error-ellipse support."""

    def test_eigenvalues_isotropic(self):
        cfg = EstimatorConfig()
        est = PositionEstimator(cfg, np.random.default_rng(0))
        est.initialize(KinematicState())
        est.state.cov = np.eye(2) * 4.0
        major, minor = est.cov_eigenvalues
        assert abs(major - 4.0) < 1e-9
        assert abs(minor - 4.0) < 1e-9

    def test_eigenvalues_anisotropic(self):
        cfg = EstimatorConfig()
        est = PositionEstimator(cfg, np.random.default_rng(0))
        est.initialize(KinematicState())
        est.state.cov = np.array([[9.0, 0.0], [0.0, 1.0]])
        major, minor = est.cov_eigenvalues
        assert abs(major - 9.0) < 1e-9
        assert abs(minor - 1.0) < 1e-9


# ===========================================================================
# 2. CBBA-lite Auction
# ===========================================================================


class TestCBBAFormationSlots:
    """compute_formation_slots geometry tests."""

    def test_slot_count(self):
        slots = compute_formation_slots(0.0, 0.0, 0.0, 5, 25.0)
        assert len(slots) == 5

    def test_slot_0_is_leader(self):
        slots = compute_formation_slots(100.0, 200.0, 0.5, 3, 25.0)
        assert slots[0] == (100.0, 200.0)

    def test_slots_behind_leader(self):
        """Slots should be spaced behind the leader along the reverse heading."""
        slots = compute_formation_slots(100.0, 100.0, 0.0, 4, 25.0)
        # heading=0 → cos=1, sin=0 → slots at (100, 100), (75, 100), (50, 100), (25, 100)
        for i in range(1, 4):
            assert abs(slots[i][0] - (100.0 - 25.0 * i)) < 1e-9
            assert abs(slots[i][1] - 100.0) < 1e-9


class TestCBBAAllocate:
    """CBBA-lite allocation algorithm tests."""

    def _make_vehicles(self, positions: list[tuple[float, float]], seed: int = 0) -> list:
        from convoy_commander.vehicles.vehicle import Vehicle
        vehicles = []
        for i, (x, y) in enumerate(positions):
            cfg = SimConfig(seed=seed + i, num_vehicles=len(positions))
            v = Vehicle(i, cfg, np.random.default_rng(seed + i), start_x=x, start_y=y)
            vehicles.append(v)
        return vehicles

    def test_each_vehicle_gets_a_slot(self):
        vehicles = self._make_vehicles([(0, 0), (10, 0), (20, 0)])
        slots = compute_formation_slots(0.0, 0.0, 0.0, 3, 25.0)
        result = cbba_allocate(vehicles, slots)
        assert len(result) == 3
        # Each vehicle gets a unique slot
        assert len(set(result.values())) == 3

    def test_closest_vehicle_wins_nearest_slot(self):
        """Vehicle at (0, 0) should win slot 0 at (0, 0) (the lead slot)."""
        vehicles = self._make_vehicles([(0, 0), (50, 0)])
        slots = [(0.0, 0.0), (-25.0, 0.0)]  # slot 0 at origin, slot 1 behind
        result = cbba_allocate(vehicles, slots)
        # Vehicle 0 is closest to slot 0
        assert result[0] == 0

    def test_no_duplicate_slot_assignments(self):
        vehicles = self._make_vehicles([(0, 0), (5, 0), (10, 0), (15, 0)])
        slots = compute_formation_slots(0.0, 0.0, 0.0, 4, 25.0)
        result = cbba_allocate(vehicles, slots)
        slot_set = set(result.values())
        assert len(slot_set) == len(result)

    def test_max_iterations_prevents_infinite_loop(self):
        vehicles = self._make_vehicles([(0, 0), (10, 0)])
        slots = [(0.0, 0.0), (-25.0, 0.0)]
        result = cbba_allocate(vehicles, slots, max_iterations=1)
        assert len(result) == 2

    def test_more_vehicles_than_slots(self):
        """When there are more vehicles than slots, all slots should still be covered."""
        vehicles = self._make_vehicles([(0, 0), (5, 0), (10, 0)])
        slots = [(0.0, 0.0), (-25.0, 0.0)]  # only 2 slots
        result = cbba_allocate(vehicles, slots)
        # At least 2 vehicles get slots
        assert len(result) >= 2

    def test_fuel_bonus_favours_full_fuel(self):
        """Vehicle with more fuel should be preferred (higher bid) for same distance."""
        vehicles = self._make_vehicles([(0, 0), (0, 0)])
        # Drain fuel from vehicle 1
        vehicles[1].fuel.fuel = 10.0  # low fuel
        slots = [(0.0, 0.0)]
        # Both are at same position; vehicle 0 has full fuel → higher bid
        b0 = _compute_bid(vehicles[0], slots[0])
        b1 = _compute_bid(vehicles[1], slots[0])
        assert b0 > b1


# ===========================================================================
# 3. Integration: CBBA + covariance in full simulation
# ===========================================================================


class TestPhase4Integration:
    """End-to-end integration tests combining covariance EKF and CBBA-lite."""

    def test_baseline_with_cbba_runs(self):
        config = get_scenario("baseline", seed=42, vehicles=4, duration=10.0)
        assert config.use_cbba
        runner = SimRunner(config)
        result = runner.run()
        assert result is not None
        # Check covariance exists on every vehicle
        for v in result.vehicles:
            assert v.estimator.state.cov.shape == (2, 2)

    def test_gps_denied_covariance_grows(self):
        """In GPS-denied, covariance should grow larger than in baseline."""
        config_gps = get_scenario("baseline", seed=42, vehicles=2, duration=20.0)
        config_denied = get_scenario("gps_denied", seed=42, vehicles=2, duration=20.0)
        result_gps = SimRunner(config_gps).run()
        result_denied = SimRunner(config_denied).run()

        avg_unc_gps = sum(v.estimator.state.uncertainty for v in result_gps.vehicles) / 2
        avg_unc_denied = sum(v.estimator.state.uncertainty for v in result_denied.vehicles) / 2
        assert avg_unc_denied > avg_unc_gps

    def test_cbba_false_uses_greedy(self):
        config = get_scenario("baseline", seed=42, vehicles=4, duration=5.0)
        # Override use_cbba
        config_dict = config.model_dump()
        config_dict["use_cbba"] = False
        config_no_cbba = SimConfig(**config_dict)
        runner = SimRunner(config_no_cbba)
        result = runner.run()
        assert result is not None

    @pytest.mark.parametrize("scenario", ["baseline", "gps_denied", "sensor_drift_spike"])
    def test_cov_always_positive_semidefinite(self, scenario: str):
        """After a simulation, all vehicle covariances should be PSD."""
        config = get_scenario(scenario, seed=42, vehicles=3, duration=15.0)
        result = SimRunner(config).run()
        for v in result.vehicles:
            eigs = np.linalg.eigvalsh(v.estimator.state.cov)
            assert all(e >= -1e-6 for e in eigs), (
                f"Vehicle {v.id}: negative eigenvalue {min(eigs)}"
            )

    def test_determinism_with_cbba(self):
        """Same seed → same result, including CBBA slot assignments."""
        config1 = get_scenario("baseline", seed=99, vehicles=4, duration=10.0)
        config2 = get_scenario("baseline", seed=99, vehicles=4, duration=10.0)
        r1 = SimRunner(config1).run()
        r2 = SimRunner(config2).run()
        for v1, v2 in zip(r1.vehicles, r2.vehicles):
            assert abs(v1.state.x - v2.state.x) < 1e-9
            assert abs(v1.state.y - v2.state.y) < 1e-9
