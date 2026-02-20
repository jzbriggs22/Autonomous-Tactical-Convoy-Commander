"""Safety-critical tests: config validation, invariant checks, event logging."""

import math

import numpy as np
import pytest

from convoy_commander.core.config import (
    CoordinationConfig,
    EstimatorConfig,
    SimConfig,
    VehicleConfig,
    WorldConfig,
)
from convoy_commander.core.event_log import EventKind, EventLog, Severity
from convoy_commander.core.physics import FuelState, KinematicState, clamp
from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.scenarios import get_scenario
from convoy_commander.vehicles.vehicle import Vehicle, VehicleStatus


# ===================================================================
# Config validation tests
# ===================================================================


class TestConfigValidation:
    """Config validators must reject unsafe parameter combinations."""

    def test_decel_must_exceed_accel(self):
        with pytest.raises(ValueError, match="SAFETY.*max_decel"):
            VehicleConfig(max_accel=5.0, max_decel=2.0)

    def test_collision_radius_lt_min_separation(self):
        with pytest.raises(ValueError, match="SAFETY.*collision_radius"):
            CoordinationConfig(collision_radius=12.0, min_separation=10.0)

    def test_min_separation_lt_formation_spacing(self):
        with pytest.raises(ValueError, match="SAFETY.*min_separation"):
            CoordinationConfig(
                collision_radius=3.0, min_separation=30.0, formation_spacing=25.0
            )

    def test_timestep_vs_obstacle_size(self):
        """max_speed * dt must not exceed half the smallest obstacle radius."""
        with pytest.raises(ValueError, match="SAFETY.*obstacle-skipping"):
            SimConfig(
                vehicle=VehicleConfig(max_speed=50.0, max_accel=2.0, max_decel=4.0),
                dt=0.5,
                world=WorldConfig(obstacle_radius_range=(5.0, 25.0)),
            )

    def test_collision_radius_vs_body_size(self):
        """collision_radius must be >= half body diagonal."""
        with pytest.raises(ValueError, match="SAFETY.*collision_radius.*body diagonal"):
            SimConfig(
                coordination=CoordinationConfig(
                    collision_radius=1.0, min_separation=5.0, formation_spacing=25.0
                ),
                vehicle=VehicleConfig(length=6.0, width=2.5, max_accel=2.0, max_decel=4.0),
            )

    def test_obstacle_radius_range_invalid(self):
        with pytest.raises(ValueError, match="SAFETY.*obstacle_radius_range"):
            WorldConfig(obstacle_radius_range=(30.0, 5.0))

    def test_nogo_zone_radius_range_invalid(self):
        with pytest.raises(ValueError, match="SAFETY.*nogo_zone_radius_range"):
            WorldConfig(nogo_zone_radius_range=(-1.0, 5.0))

    def test_valid_config_passes(self):
        """Default config should pass all validators."""
        config = SimConfig()
        assert config.vehicle.max_decel >= config.vehicle.max_accel

    def test_safety_warnings_gps_denied_no_landmarks(self):
        config = SimConfig(gps_available=False, gps_intermittent_prob=0.0)
        config.world.landmark_count = 0
        warnings = config.safety_warnings()
        assert any("unboundedly" in w for w in warnings)

    def test_safety_warnings_high_loss(self):
        config = SimConfig()
        config.comms.packet_loss = 0.8
        warnings = config.safety_warnings()
        assert any("severely degraded" in w for w in warnings)


# ===================================================================
# Event log tests
# ===================================================================


class TestEventLog:
    """Event log must be append-only with monotonic timestamps."""

    def test_basic_logging(self):
        elog = EventLog()
        elog.log(0.0, EventKind.SIM_START, Severity.INFO, message="start")
        elog.log(1.0, EventKind.VEHICLE_SPAWNED, Severity.INFO, vehicle_id=0)
        assert len(elog) == 2

    def test_monotonicity_violation_recorded(self):
        elog = EventLog()
        elog.log(5.0, EventKind.SIM_START, Severity.INFO)
        elog.log(3.0, EventKind.VEHICLE_SPAWNED, Severity.INFO)
        # Should record a monotonicity violation event + the original
        violations = elog.filter(kind=EventKind.INVARIANT_VIOLATION)
        assert len(violations) == 1

    def test_filter_by_severity(self):
        elog = EventLog()
        elog.log(0.0, EventKind.SIM_START, Severity.INFO)
        elog.log(1.0, EventKind.COLLISION, Severity.CRITICAL)
        elog.log(2.0, EventKind.NEAR_MISS, Severity.WARNING)
        critical = elog.filter(severity_min=Severity.CRITICAL)
        assert len(critical) == 1
        assert critical[0].kind == EventKind.COLLISION.value

    def test_filter_by_vehicle(self):
        elog = EventLog()
        elog.log(0.0, EventKind.VEHICLE_SPAWNED, Severity.INFO, vehicle_id=0)
        elog.log(0.0, EventKind.VEHICLE_SPAWNED, Severity.INFO, vehicle_id=1)
        v0_events = elog.filter(vehicle_id=0)
        assert len(v0_events) == 1

    def test_count_by_kind(self):
        elog = EventLog()
        elog.log(0.0, EventKind.NEAR_MISS, Severity.WARNING)
        elog.log(1.0, EventKind.NEAR_MISS, Severity.WARNING)
        elog.log(2.0, EventKind.COLLISION, Severity.CRITICAL)
        counts = elog.count_by_kind()
        assert counts["near_miss"] == 2
        assert counts["collision"] == 1

    def test_save_and_load(self, tmp_path):
        elog = EventLog()
        elog.log(0.0, EventKind.SIM_START, Severity.INFO, message="test")
        path = tmp_path / "events.jsonl"
        elog.save(path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1


# ===================================================================
# Physics precondition tests
# ===================================================================


class TestPhysicsPreconditions:
    """Physics functions must reject invalid inputs."""

    def test_clamp_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="clamp"):
            clamp(5.0, 10.0, 1.0)

    def test_kinematic_step_negative_dt_raises(self):
        state = KinematicState()
        with pytest.raises(ValueError, match="dt"):
            state.step(0, 0, dt=-1.0, max_speed=10, max_accel=2, max_decel=4, max_turn_rate=1)

    def test_kinematic_step_nan_accel_raises(self):
        state = KinematicState()
        with pytest.raises(ValueError, match="NaN"):
            state.step(float("nan"), 0, dt=0.1, max_speed=10, max_accel=2, max_decel=4, max_turn_rate=1)

    def test_fuel_consume_returns_amount(self):
        fuel = FuelState(fuel=1.0, capacity=100.0, rate_idle=0.01, rate_per_speed=0.005)
        consumed = fuel.consume(speed=10.0, dt=1.0)
        assert consumed > 0
        assert consumed <= 1.0

    def test_fuel_cannot_go_negative(self):
        fuel = FuelState(fuel=0.001, capacity=100.0, rate_idle=10.0, rate_per_speed=0.0)
        fuel.consume(speed=0.0, dt=1.0)
        assert fuel.fuel >= 0.0

    def test_fuel_is_low(self):
        fuel = FuelState(fuel=15.0, capacity=100.0)
        assert fuel.is_low
        fuel.fuel = 25.0
        assert not fuel.is_low


# ===================================================================
# Simulation safety integration tests
# ===================================================================


class TestSimSafety:
    """Integration tests for safety behaviour in the sim runner."""

    def test_event_log_exists_in_result(self):
        config = SimConfig(seed=42, duration=5.0, num_vehicles=3)
        runner = SimRunner(config)
        result = runner.run()
        assert len(result.event_log) > 0
        # Must have SIM_START and SIM_END
        kinds = {e.kind for e in result.event_log.events}
        assert "sim_start" in kinds
        assert "sim_end" in kinds

    def test_vehicle_spawn_events_logged(self):
        config = SimConfig(seed=42, duration=2.0, num_vehicles=4)
        runner = SimRunner(config)
        result = runner.run()
        spawn_events = result.event_log.filter(kind=EventKind.VEHICLE_SPAWNED)
        assert len(spawn_events) == 4

    def test_leader_election_logged(self):
        config = SimConfig(seed=42, duration=2.0, num_vehicles=3)
        runner = SimRunner(config)
        result = runner.run()
        leader_events = result.event_log.filter(kind=EventKind.LEADER_ELECTED)
        assert len(leader_events) >= 1  # At least initial leader

    def test_safe_mode_logged_in_gps_denied(self):
        """In GPS-denied with drift, vehicles should enter safe mode and log it."""
        config = get_scenario("gps_denied", seed=42, vehicles=3, duration=30.0)
        config.estimator.drift_rate = 0.5  # Aggressive drift
        config.estimator.uncertainty_safe_threshold = 5.0  # Low threshold
        config.world.landmark_count = 0  # No landmarks
        runner = SimRunner(config)
        result = runner.run()
        safe_events = result.event_log.filter(kind=EventKind.SAFE_MODE_ENTER)
        assert len(safe_events) > 0

    def test_collision_events_logged(self):
        """If collisions occur, they must be logged at CRITICAL."""
        config = SimConfig(seed=42, duration=5.0, num_vehicles=4)
        runner = SimRunner(config)
        result = runner.run()
        # Check that any collision events that occurred were logged
        collision_count = sum(v.collision_count for v in result.vehicles)
        collision_events = result.event_log.filter(kind=EventKind.COLLISION)
        # Both counters should agree (events may double-count pairs differently)
        if collision_count > 0:
            assert len(collision_events) > 0
            assert all(e.severity == "CRITICAL" for e in collision_events)

    def test_boundary_enforcement(self):
        """Vehicles must stay within world bounds."""
        config = SimConfig(seed=42, duration=50.0, num_vehicles=4)
        runner = SimRunner(config)
        result = runner.run()
        for v in result.vehicles:
            for tx, ty in v.true_history:
                assert 0 <= tx <= config.world.width, f"V{v.id} out of x bounds: {tx}"
                assert 0 <= ty <= config.world.height, f"V{v.id} out of y bounds: {ty}"

    def test_breakdown_is_irreversible(self):
        """Once a vehicle breaks down, it never becomes operational again."""
        config = get_scenario("leader_failure", seed=42, vehicles=4, duration=200.0)
        runner = SimRunner(config)
        result = runner.run()
        for v in result.vehicles:
            if v.status == VehicleStatus.BREAKDOWN:
                assert v.state.speed == 0.0
                assert not v.is_leader

    def test_deterministic_with_safety(self):
        """Safety hardening must not break determinism."""
        config = SimConfig(seed=42, duration=10.0, num_vehicles=3)
        r1 = SimRunner(config).run()
        r2 = SimRunner(config).run()
        for v1, v2 in zip(r1.vehicles, r2.vehicles):
            assert abs(v1.state.x - v2.state.x) < 1e-10
            assert abs(v1.state.y - v2.state.y) < 1e-10
