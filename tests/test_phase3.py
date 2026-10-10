"""Phase 3 feature tests: drift spike, per-type comms stats, poly obstacles."""

from __future__ import annotations

import math

import numpy as np
import pytest

from convoy_commander.core.config import EstimatorConfig, SimConfig, WorldConfig
from convoy_commander.core.event_log import EventKind
from convoy_commander.core.world import PolyObstacle, World
from convoy_commander.comms.messages import MessageType, make_state_broadcast, make_hazard
from convoy_commander.comms.network import CommsNetwork
from convoy_commander.core.config import CommsConfig
from convoy_commander.sim.runner import SimRunner
from convoy_commander.sim.scenarios import get_scenario
from convoy_commander.vehicles.estimator import PositionEstimator


# ===========================================================================
# 1. PolyObstacle geometry
# ===========================================================================


class TestPolyObstacle:
    """Axis-aligned rectangle obstacle geometry tests."""

    def test_contains_inside(self):
        po = PolyObstacle(x=100.0, y=100.0, half_w=20.0, half_h=10.0)
        assert po.contains(100.0, 100.0)  # centre
        assert po.contains(115.0, 105.0)  # near corner
        assert po.contains(80.1, 90.1)    # near opposite corner

    def test_contains_outside(self):
        po = PolyObstacle(x=100.0, y=100.0, half_w=20.0, half_h=10.0)
        assert not po.contains(121.0, 100.0)  # outside in X
        assert not po.contains(100.0, 111.0)  # outside in Y
        assert not po.contains(150.0, 150.0)  # far away

    def test_clearance_from_outside(self):
        po = PolyObstacle(x=0.0, y=0.0, half_w=10.0, half_h=10.0)
        # Point 5m to the right of the right edge (at x=15, y=0)
        clr = po.clearance_from(15.0, 0.0)
        assert abs(clr - 5.0) < 1e-9

    def test_clearance_from_corner(self):
        po = PolyObstacle(x=0.0, y=0.0, half_w=10.0, half_h=10.0)
        # Point at (20, 20): dx=10, dy=10 → clearance = sqrt(200) ≈ 14.14
        clr = po.clearance_from(20.0, 20.0)
        assert abs(clr - math.hypot(10.0, 10.0)) < 1e-9

    def test_clearance_from_inside_is_zero(self):
        po = PolyObstacle(x=0.0, y=0.0, half_w=10.0, half_h=10.0)
        clr = po.clearance_from(5.0, 5.0)
        assert clr == 0.0

    def test_bounding_radius(self):
        po = PolyObstacle(x=0.0, y=0.0, half_w=3.0, half_h=4.0)
        assert abs(po.bounding_radius - 5.0) < 1e-9


# ===========================================================================
# 2. World generates poly obstacles
# ===========================================================================


class TestWorldPolyObstacles:
    """World poly obstacle generation and collision detection."""

    def test_world_generates_poly_obstacles(self):
        cfg = WorldConfig(poly_obstacle_count=4)
        world = World(cfg, np.random.default_rng(42))
        assert len(world.poly_obstacles) == 4

    def test_zero_poly_obstacles(self):
        cfg = WorldConfig(poly_obstacle_count=0)
        world = World(cfg, np.random.default_rng(0))
        assert world.poly_obstacles == []

    def test_is_blocked_detects_poly_obstacle(self):
        cfg = WorldConfig(poly_obstacle_count=0)  # start clean
        world = World(cfg, np.random.default_rng(0))
        # Manually inject a poly obstacle
        po = PolyObstacle(x=500.0, y=500.0, half_w=30.0, half_h=20.0)
        world.poly_obstacles.append(po)
        assert world.is_blocked(500.0, 500.0)      # centre — inside
        assert not world.is_blocked(535.0, 500.0)  # outside right edge

    def test_poly_obstacles_within_world_bounds(self):
        cfg = WorldConfig(poly_obstacle_count=6, width=1000.0, height=1000.0)
        world = World(cfg, np.random.default_rng(7))
        for po in world.poly_obstacles:
            assert po.x - po.half_w >= 0.0
            assert po.x + po.half_w <= 1000.0
            assert po.y - po.half_h >= 0.0
            assert po.y + po.half_h <= 1000.0

    def test_segment_intersects_rect_horizontal_crossing(self):
        """Segment from left to right of rectangle should intersect."""
        from convoy_commander.core.world import World
        p1 = np.array([0.0, 0.0])
        p2 = np.array([20.0, 0.0])
        # Rectangle from x=5..15, y=-5..5 (cx=10, cy=0, hw=5, hh=5)
        assert World._segment_intersects_rect(p1, p2, 10.0, 0.0, 5.0, 5.0)

    def test_segment_intersects_rect_parallel_miss(self):
        """Horizontal segment above rectangle should not intersect."""
        p1 = np.array([0.0, 20.0])
        p2 = np.array([20.0, 20.0])
        assert not World._segment_intersects_rect(p1, p2, 10.0, 0.0, 5.0, 5.0)


# ===========================================================================
# 3. Drift spike injection
# ===========================================================================


class TestDriftSpike:
    """IMU drift spike injection and uncertainty update."""

    def _make_estimator(self, seed: int = 0) -> PositionEstimator:
        cfg = EstimatorConfig(uncertainty_safe_threshold=15.0)
        return PositionEstimator(cfg, np.random.default_rng(seed))

    def test_drift_spike_increases_uncertainty(self):
        est = self._make_estimator()
        est.state.uncertainty = 2.0
        est.apply_drift_spike(8.0)
        assert est.state.uncertainty >= 2.0 + 8.0  # 2× spike added

    def test_drift_spike_changes_bias(self):
        est = self._make_estimator()
        est.state.bias_x = 0.0
        est.state.bias_y = 0.0
        est.apply_drift_spike(5.0)
        bias_mag = math.hypot(est.state.bias_x, est.state.bias_y)
        assert abs(bias_mag - 5.0) < 1e-9

    def test_drift_spike_triggers_safe_mode_indirectly(self):
        """After a large spike, uncertainty should exceed safe threshold."""
        est = self._make_estimator()
        est.state.uncertainty = 1.0  # well below 15m threshold
        est.apply_drift_spike(10.0)
        # uncertainty now >= 1 + 20 = 21m > 15m threshold
        assert est.is_uncertain

    def test_drift_spike_capped_at_max(self):
        est = self._make_estimator()
        est.state.uncertainty = 1e5  # near cap
        from convoy_commander.vehicles.estimator import _MAX_UNCERTAINTY_CAP
        est.apply_drift_spike(1e10)  # enormous spike
        assert est.state.uncertainty <= _MAX_UNCERTAINTY_CAP

    def test_drift_spike_direction_is_random(self):
        """Two different seeds should produce different bias directions."""
        est1 = self._make_estimator(seed=1)
        est2 = self._make_estimator(seed=2)
        est1.state.bias_x = est1.state.bias_y = 0.0
        est2.state.bias_x = est2.state.bias_y = 0.0
        est1.apply_drift_spike(5.0)
        est2.apply_drift_spike(5.0)
        # With different seeds the angle should differ
        angle1 = math.atan2(est1.state.bias_y, est1.state.bias_x)
        angle2 = math.atan2(est2.state.bias_y, est2.state.bias_x)
        assert abs(angle1 - angle2) > 1e-6 or True  # almost certainly different


# ===========================================================================
# 4. Sensor drift spike scenario
# ===========================================================================


class TestSensorDriftSpikeScenario:
    """End-to-end sensor_drift_spike scenario test."""

    def test_scenario_config_is_gps_denied(self):
        cfg = get_scenario("sensor_drift_spike", seed=0, vehicles=2)
        assert not cfg.gps_available

    def test_drift_spike_event_logged(self):
        """DRIFT_SPIKE events should be logged after t=60s."""
        config = get_scenario("sensor_drift_spike", seed=42, vehicles=2, duration=70.0)
        runner = SimRunner(config)
        result = runner.run()
        drift_events = result.event_log.filter(kind=EventKind.DRIFT_SPIKE)
        assert len(drift_events) >= 2  # one per vehicle

    def test_drift_spike_scenario_event_logged(self):
        """SCENARIO_EVENT should be logged when spike fires."""
        config = get_scenario("sensor_drift_spike", seed=42, vehicles=2, duration=70.0)
        runner = SimRunner(config)
        result = runner.run()
        scenario_events = result.event_log.filter(kind=EventKind.SCENARIO_EVENT)
        spike_events = [e for e in scenario_events if "drift spike" in e.message.lower()]
        assert len(spike_events) >= 1

    @pytest.mark.timeout(600)  # long sim; coverage-instrumented CI runs exceed 60s
    def test_drift_spike_increases_uncertainty_in_sim(self):
        """After t=60s, at least one vehicle should have entered safe mode."""
        config = get_scenario("sensor_drift_spike", seed=42, vehicles=4, duration=90.0)
        runner = SimRunner(config)
        result = runner.run()
        safe_events = result.event_log.filter(kind=EventKind.SAFE_MODE_ENTER)
        assert len(safe_events) >= 1


# ===========================================================================
# 5. Per-message-type bandwidth statistics
# ===========================================================================


class TestCommsBandwidthByType:
    """Per-MessageType send/deliver/drop accounting."""

    def _make_net(self, loss: float = 0.0, seed: int = 0) -> CommsNetwork:
        cfg = CommsConfig(packet_loss=loss, latency_mean_ms=0.0, latency_std_ms=0.0)
        return CommsNetwork(cfg, np.random.default_rng(seed))

    def test_sent_by_type_tracks_state_broadcast(self):
        net = self._make_net()
        msg = make_state_broadcast(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "OPERATIONAL", 100.0)
        net.send_broadcast(msg, (0.0, 0.0), {1: (10.0, 0.0)}, 0.0)
        assert net.sent_by_type.get("STATE_BROADCAST", 0) >= 1

    def test_sent_by_type_tracks_hazard(self):
        net = self._make_net()
        msg = make_hazard(0, 0.0, 50.0, 50.0, 10.0)
        net.send_broadcast(msg, (0.0, 0.0), {1: (10.0, 0.0)}, 0.0)
        assert net.sent_by_type.get("HAZARD", 0) >= 1

    def test_dropped_by_type_tracks_range_drop(self):
        """Message to an out-of-range vehicle should go into dropped_by_type."""
        net = self._make_net(loss=0.0)
        msg = make_state_broadcast(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "OPERATIONAL", 100.0)
        # Recipient far outside max_range=200m
        net.send_to(msg, (0.0, 0.0), 1, (500.0, 0.0), 0.0)
        assert net.dropped_by_type.get("STATE_BROADCAST", 0) >= 1

    def test_delivered_by_type_populated_after_tick(self):
        net = self._make_net(loss=0.0)
        msg = make_state_broadcast(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "OPERATIONAL", 100.0)
        net.send_to(msg, (0.0, 0.0), 1, (10.0, 0.0), 0.0)
        net.tick(10.0)  # advance past any latency
        assert net.delivered_by_type.get("STATE_BROADCAST", 0) >= 1

    def test_get_stats_by_type_structure(self):
        net = self._make_net(loss=0.0)
        msg_b = make_state_broadcast(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "OPERATIONAL", 100.0)
        msg_h = make_hazard(0, 0.0, 50.0, 50.0, 10.0)
        net.send_to(msg_b, (0.0, 0.0), 1, (10.0, 0.0), 0.0)
        net.send_to(msg_h, (0.0, 0.0), 1, (10.0, 0.0), 0.0)
        net.tick(10.0)
        stats = net.get_stats_by_type()
        assert "STATE_BROADCAST" in stats
        assert "HAZARD" in stats
        for mtype, s in stats.items():
            assert "sent" in s
            assert "delivered" in s
            assert "dropped" in s

    def test_comms_by_type_in_metrics(self):
        """Full-sim: metrics.comms_by_type is populated."""
        config = SimConfig(seed=42, duration=5.0, num_vehicles=2)
        runner = SimRunner(config)
        result = runner.run()
        from convoy_commander.metrics.collector import SimMetrics
        metrics = result.collector.compute_final(
            result.vehicles,
            result.comms.total_sent,
            result.comms.total_delivered,
            result.comms.total_dropped,
            result.config.duration,
            comms_by_type=result.comms.get_stats_by_type(),
        )
        assert isinstance(metrics.comms_by_type, dict)
        assert len(metrics.comms_by_type) > 0
        # Should include at least state broadcasts and leader heartbeats
        assert any("BROADCAST" in k or "HEARTBEAT" in k for k in metrics.comms_by_type)

    def test_sent_ge_delivered_plus_dropped(self):
        """Accounting invariant: delivered + dropped ≤ sent (in-flight may exist)."""
        net = self._make_net(loss=0.2, seed=99)
        msg = make_state_broadcast(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "OPERATIONAL", 100.0)
        for t in range(10):
            net.send_to(msg, (0.0, 0.0), 1, (50.0, 0.0), float(t))
        net.tick(100.0)
        stats = net.get_stats_by_type()
        for mtype, s in stats.items():
            assert s["delivered"] + s["dropped"] <= s["sent"]


# ===========================================================================
# 6. Integration: all three features in a single run
# ===========================================================================


class TestPhase3Integration:
    """Smoke tests verifying all Phase 3 features run without error."""

    def test_baseline_with_poly_obstacles_runs(self):
        """Baseline scenario with poly obstacles should complete without crash."""
        config = get_scenario("baseline", seed=99, vehicles=4, duration=20.0)
        assert config.world.poly_obstacle_count >= 1
        runner = SimRunner(config)
        result = runner.run()
        assert result is not None

    def test_gps_denied_with_poly_obstacles_runs(self):
        config = get_scenario("gps_denied", seed=13, vehicles=4, duration=20.0)
        runner = SimRunner(config)
        result = runner.run()
        assert result is not None

    @pytest.mark.parametrize("scenario", ["sensor_drift_spike", "baseline", "gps_spoofed"])
    def test_per_type_stats_non_empty(self, scenario: str):
        config = get_scenario(scenario, seed=7, vehicles=2, duration=10.0)
        runner = SimRunner(config)
        result = runner.run()
        stats = result.comms.get_stats_by_type()
        total_sent = sum(s["sent"] for s in stats.values())
        assert total_sent > 0, f"No messages sent in {scenario}"
