"""Phase 7 tests: visualization + polish.

Tests cover:
  - Evaluate defaults: platooning in DEFAULT_SCENARIOS, all defaults valid
  - MetricsCollector new fields: headway_samples, corridor_samples, spacing_error_samples
  - Plot functions: headway gaps, string stability, corridor adherence
  - Metrics summary 4-subplot layout
  - Assumptions text updated for Phase 6
  - CLI help includes platooning
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from convoy_commander.core.config import SimConfig
from convoy_commander.evaluate import DEFAULT_SCENARIOS
from convoy_commander.metrics.collector import MetricsCollector
from convoy_commander.sim.scenarios import get_scenario
from convoy_commander.vehicles.vehicle import Vehicle
from convoy_commander.viz.report import _build_assumptions_section


# ---------------------------------------------------------------------------
# Evaluate Defaults
# ---------------------------------------------------------------------------


class TestEvaluateDefaults:
    def test_platooning_in_default_scenarios(self):
        assert "platooning" in DEFAULT_SCENARIOS

    def test_all_default_scenarios_are_valid(self):
        for name in DEFAULT_SCENARIOS:
            config = get_scenario(name)
            assert config.scenario == name

    def test_default_scenarios_has_expected_entries(self):
        assert len(DEFAULT_SCENARIOS) >= 6


# ---------------------------------------------------------------------------
# MetricsCollector New Fields
# ---------------------------------------------------------------------------


class TestCollectorNewFields:
    def test_headway_samples_initially_empty(self):
        c = MetricsCollector()
        assert c.headway_samples == []

    def test_corridor_samples_initially_empty(self):
        c = MetricsCollector()
        assert c.corridor_samples == []

    def test_spacing_error_samples_initially_empty(self):
        c = MetricsCollector()
        assert c.spacing_error_samples == []

    def test_record_headway_appends(self):
        c = MetricsCollector()
        c.record_headway(1.0, 2, 15.0, 12.0)
        assert len(c.headway_samples) == 1
        s = c.headway_samples[0]
        assert s["time"] == 1.0
        assert s["vehicle_id"] == 2
        assert s["actual_gap"] == 15.0
        assert s["desired_gap"] == 12.0

    def test_record_corridor_distance_appends(self):
        c = MetricsCollector()
        c.record_corridor_distance(2.5, 3, 5.0)
        assert len(c.corridor_samples) == 1
        s = c.corridor_samples[0]
        assert s["time"] == 2.5
        assert s["vehicle_id"] == 3
        assert s["corridor_dist"] == 5.0

    def test_record_spacing_error_appends(self):
        c = MetricsCollector()
        c.record_spacing_error(3.0, 1, -2.5)
        assert len(c.spacing_error_samples) == 1
        s = c.spacing_error_samples[0]
        assert s["time"] == 3.0
        assert s["vehicle_id"] == 1
        assert s["error"] == -2.5

    def test_multiple_records(self):
        c = MetricsCollector()
        for i in range(10):
            c.record_headway(float(i), 0, float(i) * 2, 10.0)
        assert len(c.headway_samples) == 10


# ---------------------------------------------------------------------------
# Plot Functions
# ---------------------------------------------------------------------------


class TestPlotHeadwayGaps:
    def test_plot_creates_file(self):
        c = MetricsCollector()
        for t in range(100):
            c.record_headway(float(t) * 0.1, 1, 15.0 + t * 0.01, 12.0)
            c.record_headway(float(t) * 0.1, 2, 18.0 + t * 0.01, 12.0)

        config = SimConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "headway.png"
            from convoy_commander.viz.plots import plot_headway_gaps
            plot_headway_gaps(c, config, 0.1, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_plot_empty_data_no_crash(self):
        c = MetricsCollector()
        config = SimConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "headway.png"
            from convoy_commander.viz.plots import plot_headway_gaps
            plot_headway_gaps(c, config, 0.1, out)
            assert out.exists()


class TestPlotStringStability:
    def test_plot_creates_file(self):
        c = MetricsCollector()
        for t in range(100):
            c.record_spacing_error(40.0 + t * 0.1, 1, math.sin(t * 0.1))
            c.record_spacing_error(40.0 + t * 0.1, 2, math.sin(t * 0.1) * 0.8)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "stability.png"
            from convoy_commander.viz.plots import plot_string_stability
            plot_string_stability(c, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_plot_empty_data_no_crash(self):
        c = MetricsCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "stability.png"
            from convoy_commander.viz.plots import plot_string_stability
            plot_string_stability(c, out)
            assert out.exists()


class TestPlotCorridorAdherence:
    def test_plot_creates_file(self):
        c = MetricsCollector()
        for t in range(50):
            c.record_corridor_distance(float(t) * 0.1, 0, 3.0 + t * 0.05)

        config = SimConfig()
        from convoy_commander.core.world import World
        world = World(config.world, np.random.default_rng(42))
        vehicles: list[Vehicle] = []
        for i in range(2):
            v = Vehicle(i, config, np.random.default_rng(i + 1), 50.0, 50.0, 0.3)
            vehicles.append(v)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "corridor.png"
            from convoy_commander.viz.plots import plot_corridor_adherence
            plot_corridor_adherence(world, vehicles, c, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_plot_empty_data_no_crash(self):
        c = MetricsCollector()
        config = SimConfig()
        from convoy_commander.core.world import World
        world = World(config.world, np.random.default_rng(42))

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "corridor.png"
            from convoy_commander.viz.plots import plot_corridor_adherence
            plot_corridor_adherence(world, [], c, out)
            assert out.exists()


# ---------------------------------------------------------------------------
# Metrics Summary (4 subplots)
# ---------------------------------------------------------------------------


class TestMetricsSummaryFourSubplots:
    def test_metrics_summary_with_headway_data(self):
        c = MetricsCollector()
        config = SimConfig()
        # Create minimal vehicle + time series entries
        vehicles: list[Vehicle] = []
        for i in range(2):
            v = Vehicle(i, config, np.random.default_rng(i + 1), 50.0, 50.0, 0.3)
            vehicles.append(v)
            # Simulate a few steps so there's time series data
            from convoy_commander.vehicles.vehicle import VehicleCommand
            for _ in range(5):
                v.step(VehicleCommand(accel=0.5, turn_rate=0.0), 0.1)

        # Record time series
        for t in range(5):
            c.record_step(float(t) * 0.1, vehicles)
            c.record_headway(float(t) * 0.1, 1, 15.0, 12.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "summary.png"
            from convoy_commander.viz.plots import plot_metrics_summary
            plot_metrics_summary(c, vehicles, 0.1, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_metrics_summary_without_headway(self):
        c = MetricsCollector()
        config = SimConfig()
        vehicles: list[Vehicle] = []
        for i in range(2):
            v = Vehicle(i, config, np.random.default_rng(i + 1), 50.0, 50.0, 0.3)
            vehicles.append(v)

        for t in range(3):
            c.record_step(float(t) * 0.1, vehicles)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "summary.png"
            from convoy_commander.viz.plots import plot_metrics_summary
            plot_metrics_summary(c, vehicles, 0.1, out)
            assert out.exists()


# ---------------------------------------------------------------------------
# Assumptions Text
# ---------------------------------------------------------------------------


class TestAssumptionsText:
    def test_assumptions_mentions_gauss_markov(self):
        lines = _build_assumptions_section()
        text = "\n".join(lines)
        assert "Gauss-Markov" in text

    def test_assumptions_mentions_time_headway(self):
        lines = _build_assumptions_section()
        text = "\n".join(lines)
        assert "time headway" in text

    def test_assumptions_mentions_actuator_lag(self):
        lines = _build_assumptions_section()
        text = "\n".join(lines)
        assert "Actuator lag" in text

    def test_assumptions_mentions_corridor(self):
        lines = _build_assumptions_section()
        text = "\n".join(lines)
        assert "corridor" in text.lower()


# ---------------------------------------------------------------------------
# CLI Help
# ---------------------------------------------------------------------------


class TestCLIHelpText:
    def test_scenario_help_includes_platooning(self):
        import argparse
        from convoy_commander.cli import main
        import io
        import sys

        # Capture the help output
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        run_parser = subparsers.add_parser("run")
        run_parser.add_argument(
            "--scenario", default="baseline",
            help="Scenario name: baseline|gps_denied|comms_degraded|leader_failure|"
                 "obstacle_pop|gps_spoofed|silent_running|comms_blackout|sensor_drift_spike|platooning",
        )
        # Just verify the help string in the actual CLI module
        from convoy_commander import cli
        import inspect
        source = inspect.getsource(cli)
        assert "platooning" in source


# ---------------------------------------------------------------------------
# Integration: platooning scenario records new collector data
# ---------------------------------------------------------------------------


class TestPlatooningIntegration:
    def test_platooning_populates_headway_samples(self):
        config = get_scenario("platooning", duration=10.0, vehicles=4)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        assert len(result.collector.headway_samples) > 0

    def test_platooning_populates_corridor_samples(self):
        config = get_scenario("platooning", duration=10.0, vehicles=4)
        from convoy_commander.sim.runner import SimRunner
        runner = SimRunner(config)
        result = runner.run()
        assert len(result.collector.corridor_samples) > 0

    def test_platooning_report_generates_new_plots(self):
        config = get_scenario("platooning", duration=5.0, vehicles=3)
        from convoy_commander.sim.runner import SimRunner
        from convoy_commander.viz.report import generate_report
        runner = SimRunner(config)
        result = runner.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, Path(tmpdir))
            assert report_path.exists()
            plots_dir = Path(tmpdir) / "plots"
            assert (plots_dir / "headway_gaps.png").exists()
            assert (plots_dir / "string_stability.png").exists()
            assert (plots_dir / "corridor_adherence.png").exists()
