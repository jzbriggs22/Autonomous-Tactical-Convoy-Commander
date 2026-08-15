"""Tests for simulation runner."""

import numpy as np

from convoy_commander.core.config import SimConfig
from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.scenarios import get_scenario
from convoy_commander.vehicles.vehicle import VehicleStatus


def test_deterministic_stepping():
    """Same seed should produce identical results."""
    config = SimConfig(seed=42, duration=10.0, num_vehicles=3)
    config.world.obstacle_count = 3
    config.world.nogo_zone_count = 1

    runner1 = SimRunner(config)
    result1 = runner1.run()

    runner2 = SimRunner(config)
    result2 = runner2.run()

    for v1, v2 in zip(result1.vehicles, result2.vehicles):
        assert abs(v1.state.x - v2.state.x) < 1e-10
        assert abs(v1.state.y - v2.state.y) < 1e-10
        assert abs(v1.fuel.fuel - v2.fuel.fuel) < 1e-10


def test_collision_avoidance_baseline():
    """In baseline scenario, minimum separation should be maintained most of the time."""
    config = get_scenario("baseline", seed=42, vehicles=4, duration=30.0)
    runner = SimRunner(config)
    result = runner.run()

    total_collisions = sum(v.collision_count for v in result.vehicles)
    # Allow some initial collisions during formation, but should be low
    assert total_collisions < 50, f"Too many collisions: {total_collisions}"


def test_metrics_computed():
    """Metrics should be computed after a run."""
    config = SimConfig(seed=42, duration=10.0, num_vehicles=3)
    config.world.obstacle_count = 3
    config.world.nogo_zone_count = 1
    runner = SimRunner(config)
    result = runner.run()

    metrics = result.collector.compute_final(
        result.vehicles,
        result.comms.total_sent,
        result.comms.total_delivered,
        result.comms.total_dropped,
        config.duration,
    )
    assert metrics.vehicles_total == 3
    assert metrics.total_fuel_used > 0
    assert metrics.comms_total_sent > 0
    assert metrics.total_distance_traveled > 0
    assert len(result.collector.time_series) > 0


def test_scenario_baseline():
    """Baseline scenario should load without errors."""
    config = get_scenario("baseline")
    assert config.gps_available is True
    assert config.scenario == "baseline"


def test_scenario_gps_denied():
    """GPS denied scenario should have GPS off."""
    config = get_scenario("gps_denied")
    assert config.gps_available is False


def test_scenario_comms_degraded():
    """Comms degraded scenario should have high loss."""
    config = get_scenario("comms_degraded")
    assert config.comms.packet_loss >= 0.2


def test_short_sim_runs():
    """A short simulation should complete without errors."""
    for scenario in ["baseline", "gps_denied", "comms_degraded"]:
        config = get_scenario(scenario, seed=42, vehicles=4, duration=5.0)
        runner = SimRunner(config)
        result = runner.run()
        assert len(result.vehicles) == 4
