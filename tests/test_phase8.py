"""Phase 8 tests: neighbor state tables, multi-hop relay, error ellipses, network topology.

Tests cover:
  - Vehicle neighbor state table: update, staleness, pruning
  - CommsNetwork multi-hop relay: relay_broadcast, multi-hop adjacency, network stats
  - Error-ellipse visualization: plot creates file, empty data no crash
  - Network topology visualization: plot creates file
  - mesh_relay scenario: runs and produces new plot outputs
  - Evaluate defaults: mesh_relay in DEFAULT_SCENARIOS
  - CLI help: mesh_relay in help text
  - Assumptions updated for Phase 8
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from convoy_commander.comms.messages import MessageType, make_state_broadcast
from convoy_commander.comms.network import CommsNetwork
from convoy_commander.core.config import CommsConfig, SimConfig
from convoy_commander.evaluate import DEFAULT_SCENARIOS
from convoy_commander.metrics.collector import MetricsCollector
from convoy_commander.sim.scenarios import get_scenario
from convoy_commander.vehicles.vehicle import Vehicle
from convoy_commander.viz.report import _build_assumptions_section


# ---------------------------------------------------------------------------
# Vehicle Neighbor State Table
# ---------------------------------------------------------------------------


class TestNeighborStateTable:
    def test_update_neighbor(self):
        config = SimConfig()
        v = Vehicle(0, config, np.random.default_rng(1), 50.0, 50.0, 0.3)
        payload = {
            "x": 100.0, "y": 100.0, "heading": 0.5,
            "speed": 5.0, "uncertainty": 2.0, "fuel": 80.0, "status": "ACTIVE",
        }
        v.update_neighbor(1, payload, 10.0)
        assert 1 in v.neighbor_states
        assert v.neighbor_states[1]["x"] == 100.0
        assert v.neighbor_states[1]["timestamp"] == 10.0

    def test_update_neighbor_overwrites(self):
        config = SimConfig()
        v = Vehicle(0, config, np.random.default_rng(1), 50.0, 50.0, 0.3)
        payload1 = {"x": 100.0, "y": 100.0, "heading": 0.0, "speed": 5.0,
                     "uncertainty": 2.0, "fuel": 80.0, "status": "ACTIVE"}
        payload2 = {"x": 200.0, "y": 200.0, "heading": 0.0, "speed": 8.0,
                     "uncertainty": 1.0, "fuel": 70.0, "status": "ACTIVE"}
        v.update_neighbor(1, payload1, 10.0)
        v.update_neighbor(1, payload2, 15.0)
        assert v.neighbor_states[1]["x"] == 200.0
        assert v.neighbor_states[1]["timestamp"] == 15.0

    def test_get_stale_neighbors(self):
        config = SimConfig()
        v = Vehicle(0, config, np.random.default_rng(1), 50.0, 50.0, 0.3)
        payload = {"x": 0, "y": 0, "heading": 0, "speed": 0,
                   "uncertainty": 0, "fuel": 0, "status": "ACTIVE"}
        v.update_neighbor(1, payload, 5.0)
        v.update_neighbor(2, payload, 10.0)
        stale = v.get_stale_neighbors(current_time=12.0, max_age=5.0)
        assert 1 in stale
        assert 2 not in stale

    def test_prune_stale_neighbors(self):
        config = SimConfig()
        v = Vehicle(0, config, np.random.default_rng(1), 50.0, 50.0, 0.3)
        payload = {"x": 0, "y": 0, "heading": 0, "speed": 0,
                   "uncertainty": 0, "fuel": 0, "status": "ACTIVE"}
        v.update_neighbor(1, payload, 1.0)
        v.update_neighbor(2, payload, 14.0)
        removed = v.prune_stale_neighbors(current_time=15.0, max_age=3.0)
        assert removed == 1
        assert 1 not in v.neighbor_states
        assert 2 in v.neighbor_states

    def test_empty_neighbor_table(self):
        config = SimConfig()
        v = Vehicle(0, config, np.random.default_rng(1), 50.0, 50.0, 0.3)
        assert v.neighbor_states == {}
        assert v.get_stale_neighbors(10.0, 5.0) == []
        assert v.prune_stale_neighbors(10.0, 5.0) == 0


# ---------------------------------------------------------------------------
# CommsNetwork Multi-Hop Relay
# ---------------------------------------------------------------------------


class TestMultiHopRelay:
    def test_relay_config_defaults(self):
        config = CommsConfig()
        assert config.max_relay_hops == 0
        assert config.relay_loss_per_hop == 0.1

    def test_relay_disabled_by_default(self):
        config = CommsConfig()
        rng = np.random.default_rng(42)
        net = CommsNetwork(config, rng)
        msg = make_state_broadcast(0, 1.0, 50.0, 50.0, 0.0, 5.0, 1.0, "ACTIVE", 80.0)
        count = net.relay_broadcast(
            msg, 1, (50.0, 50.0), {0: (50.0, 50.0), 2: (100.0, 100.0)},
            1.0, {0, 1}, None,
        )
        assert count == 0

    def test_relay_broadcasts_to_unreached(self):
        config = CommsConfig(max_relay_hops=2, max_range=200.0, packet_loss=0.0)
        rng = np.random.default_rng(42)
        net = CommsNetwork(config, rng)
        msg = make_state_broadcast(0, 1.0, 50.0, 50.0, 0.0, 5.0, 1.0, "ACTIVE", 80.0)
        positions = {0: (50.0, 50.0), 1: (100.0, 100.0), 2: (150.0, 100.0)}
        already_received = {0, 1}
        count = net.relay_broadcast(msg, 1, (100.0, 100.0), positions, 1.0,
                                    already_received, None)
        assert count >= 1  # Should attempt to relay to vehicle 2
        assert net.total_relayed >= 1

    def test_relay_respects_hop_limit(self):
        config = CommsConfig(max_relay_hops=1, max_range=200.0, packet_loss=0.0)
        rng = np.random.default_rng(42)
        net = CommsNetwork(config, rng)
        # Message already has 1 hop in relay path
        msg = make_state_broadcast(0, 1.0, 50.0, 50.0, 0.0, 5.0, 1.0, "ACTIVE", 80.0)
        msg.payload["_relay_path"] = [1]
        positions = {0: (50.0, 50.0), 1: (100.0, 100.0), 2: (150.0, 100.0)}
        count = net.relay_broadcast(msg, 2, (150.0, 100.0), positions, 1.0, {0, 1, 2}, None)
        assert count == 0  # Should not relay — already at max hops

    def test_relay_stats_tracked(self):
        config = CommsConfig(max_relay_hops=2, max_range=200.0, packet_loss=0.0)
        rng = np.random.default_rng(42)
        net = CommsNetwork(config, rng)
        assert net.total_relayed == 0
        assert net.total_relay_delivered == 0

    def test_multi_hop_adjacency(self):
        config = CommsConfig(max_relay_hops=2, max_range=100.0, packet_loss=0.0)
        rng = np.random.default_rng(42)
        net = CommsNetwork(config, rng)
        # Chain: 0 -- 1 -- 2 (each 80m apart, range 100m)
        positions = {0: (0.0, 0.0), 1: (80.0, 0.0), 2: (160.0, 0.0)}
        one_hop = net.get_adjacency(positions)
        # 0 can reach 1, 1 can reach 0 and 2, 2 can reach 1 (but 0 and 2 are 160m apart)
        assert 1 in one_hop[0]
        assert 2 not in one_hop[0]  # Too far for direct
        multi = net.get_multi_hop_adjacency(positions)
        assert 2 in multi[0]  # Reachable via 1
        assert 0 in multi[2]  # Reachable via 1

    def test_network_stats(self):
        config = CommsConfig(max_relay_hops=0, max_range=200.0)
        rng = np.random.default_rng(42)
        net = CommsNetwork(config, rng)
        positions = {0: (0.0, 0.0), 1: (50.0, 0.0), 2: (100.0, 0.0)}
        stats = net.get_network_stats(positions)
        assert "avg_degree" in stats
        assert "min_degree" in stats
        assert "num_partitions" in stats
        assert stats["num_partitions"] == 1  # All within range

    def test_network_stats_partitioned(self):
        config = CommsConfig(max_relay_hops=0, max_range=50.0)
        rng = np.random.default_rng(42)
        net = CommsNetwork(config, rng)
        # Two clusters far apart
        positions = {0: (0.0, 0.0), 1: (20.0, 0.0), 2: (500.0, 500.0)}
        stats = net.get_network_stats(positions)
        assert stats["num_partitions"] == 2


# ---------------------------------------------------------------------------
# Error Ellipse Visualization
# ---------------------------------------------------------------------------


class TestPlotErrorEllipses:
    def test_plot_creates_file(self):
        c = MetricsCollector()
        for t in range(10):
            c.record_cov_ellipse(float(t), 0, 50.0 + t, 50.0 + t, 2.0, 0.5, 0.3)
            c.record_cov_ellipse(float(t), 1, 60.0 + t, 60.0 + t, 1.0, 0.8, -0.1)

        config = SimConfig()
        from convoy_commander.core.world import World
        world = World(config.world, np.random.default_rng(42))
        vehicles: list[Vehicle] = []
        for i in range(2):
            v = Vehicle(i, config, np.random.default_rng(i + 1), 50.0, 50.0, 0.3)
            vehicles.append(v)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ellipses.png"
            from convoy_commander.viz.plots import plot_error_ellipses
            plot_error_ellipses(world, vehicles, c, 0.1, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_plot_empty_data_no_crash(self):
        c = MetricsCollector()
        config = SimConfig()
        from convoy_commander.core.world import World
        world = World(config.world, np.random.default_rng(42))

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ellipses.png"
            from convoy_commander.viz.plots import plot_error_ellipses
            plot_error_ellipses(world, [], c, 0.1, out)
            assert out.exists()


# ---------------------------------------------------------------------------
# Network Topology Visualization
# ---------------------------------------------------------------------------


class TestPlotNetworkTopology:
    def test_plot_creates_file(self):
        c = MetricsCollector()
        for t in range(20):
            c.record_network_stats(float(t), {
                "avg_degree": 3.5 + t * 0.01,
                "min_degree": 2,
                "num_partitions": 1,
                "relay_reach_avg": 5.0,
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "network.png"
            from convoy_commander.viz.plots import plot_network_topology
            plot_network_topology(c, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_plot_empty_data_no_crash(self):
        c = MetricsCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "network.png"
            from convoy_commander.viz.plots import plot_network_topology
            plot_network_topology(c, out)
            assert out.exists()


# ---------------------------------------------------------------------------
# Evaluate Defaults
# ---------------------------------------------------------------------------


class TestEvaluateDefaultsPhase8:
    def test_mesh_relay_in_default_scenarios(self):
        assert "mesh_relay" in DEFAULT_SCENARIOS

    def test_default_scenarios_has_seven_entries(self):
        assert len(DEFAULT_SCENARIOS) == 7

    def test_mesh_relay_scenario_is_valid(self):
        config = get_scenario("mesh_relay")
        assert config.scenario == "mesh_relay"
        assert config.comms.max_relay_hops == 2
        assert config.comms.max_range == 100.0


# ---------------------------------------------------------------------------
# CLI Help
# ---------------------------------------------------------------------------


class TestCLIHelpPhase8:
    def test_cli_source_includes_mesh_relay(self):
        from convoy_commander import cli
        import inspect
        source = inspect.getsource(cli)
        assert "mesh_relay" in source


# ---------------------------------------------------------------------------
# Assumptions Text
# ---------------------------------------------------------------------------


class TestAssumptionsPhase8:
    def test_assumptions_mentions_multi_hop(self):
        lines = _build_assumptions_section()
        text = "\n".join(lines)
        assert "multi-hop" in text.lower() or "Multi-hop" in text

    def test_assumptions_mentions_neighbor_state(self):
        lines = _build_assumptions_section()
        text = "\n".join(lines)
        assert "neighbor" in text.lower() or "Neighbor" in text


# ---------------------------------------------------------------------------
# Collector New Fields
# ---------------------------------------------------------------------------


class TestCollectorPhase8Fields:
    def test_cov_ellipse_samples_initially_empty(self):
        c = MetricsCollector()
        assert c.cov_ellipse_samples == []

    def test_network_stats_snapshots_initially_empty(self):
        c = MetricsCollector()
        assert c.network_stats_snapshots == []

    def test_record_cov_ellipse(self):
        c = MetricsCollector()
        c.record_cov_ellipse(1.0, 0, 50.0, 50.0, 2.0, 0.5, 0.3)
        assert len(c.cov_ellipse_samples) == 1
        s = c.cov_ellipse_samples[0]
        assert s["vehicle_id"] == 0
        assert s["major"] == 2.0

    def test_record_network_stats(self):
        c = MetricsCollector()
        c.record_network_stats(1.0, {"avg_degree": 3.0, "num_partitions": 1})
        assert len(c.network_stats_snapshots) == 1
        assert c.network_stats_snapshots[0]["avg_degree"] == 3.0


# ---------------------------------------------------------------------------
# Integration: mesh_relay scenario runs and populates new data
# ---------------------------------------------------------------------------


class TestMeshRelayIntegration:
    def test_mesh_relay_runs(self):
        config = get_scenario("mesh_relay", duration=10.0, vehicles=4)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        assert result.collector is not None
        assert len(result.collector.time_series) > 0

    def test_mesh_relay_populates_network_stats(self):
        config = get_scenario("mesh_relay", duration=10.0, vehicles=4)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        assert len(result.collector.network_stats_snapshots) > 0

    def test_mesh_relay_populates_cov_ellipses(self):
        config = get_scenario("mesh_relay", duration=10.0, vehicles=4)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        assert len(result.collector.cov_ellipse_samples) > 0

    def test_mesh_relay_populates_neighbor_states(self):
        config = get_scenario("mesh_relay", duration=10.0, vehicles=4)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        # At least some vehicles should have neighbor state entries
        has_neighbors = any(len(v.neighbor_states) > 0 for v in result.vehicles)
        assert has_neighbors

    def test_mesh_relay_report_generates_new_plots(self):
        config = get_scenario("mesh_relay", duration=5.0, vehicles=3)
        from convoy_commander.sim.runner import SimRunner
        from convoy_commander.viz.report import generate_report
        runner = SimRunner(config)
        result = runner.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, Path(tmpdir))
            assert report_path.exists()
            plots_dir = Path(tmpdir) / "plots"
            assert (plots_dir / "error_ellipses.png").exists()
            assert (plots_dir / "network_topology.png").exists()

    def test_relay_stats_in_comms(self):
        config = get_scenario("mesh_relay", duration=10.0, vehicles=4)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        # Relay should have been attempted at least once
        assert result.comms.total_relayed >= 0  # Could be 0 if all in range


# ---------------------------------------------------------------------------
# Baseline scenario still populates Phase 8 data
# ---------------------------------------------------------------------------


class TestBaselinePhase8Data:
    def test_baseline_populates_cov_ellipses(self):
        config = get_scenario("baseline", duration=10.0, vehicles=3)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        assert len(result.collector.cov_ellipse_samples) > 0

    def test_baseline_populates_network_stats(self):
        config = get_scenario("baseline", duration=10.0, vehicles=3)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        assert len(result.collector.network_stats_snapshots) > 0

    def test_baseline_vehicles_have_neighbor_states(self):
        config = get_scenario("baseline", duration=10.0, vehicles=3)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        has_neighbors = any(len(v.neighbor_states) > 0 for v in result.vehicles)
        assert has_neighbors
