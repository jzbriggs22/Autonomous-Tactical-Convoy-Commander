"""Phase 5 tests: evaluate harness, reproducibility stamp, report --last."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from convoy_commander.core.config import SimConfig
from convoy_commander.evaluate import run_evaluation, RunRecord, EvalSummary
from convoy_commander.sim.runner import SimRunner
from convoy_commander.stamp import collect_stamp, ReproStamp


# ===========================================================================
# 1. Reproducibility Stamp
# ===========================================================================


class TestReproStamp:
    """Tests for the reproducibility stamp module."""

    def test_collect_stamp_returns_reprostamp(self):
        config = SimConfig(seed=99, scenario="baseline")
        stamp = collect_stamp(config)
        assert isinstance(stamp, ReproStamp)
        assert stamp.seed == 99
        assert stamp.scenario == "baseline"

    def test_stamp_has_python_version(self):
        config = SimConfig()
        stamp = collect_stamp(config)
        assert stamp.python_version
        assert "." in stamp.python_version

    def test_stamp_has_platform(self):
        config = SimConfig()
        stamp = collect_stamp(config)
        assert stamp.platform_info

    def test_stamp_config_dict_is_complete(self):
        config = SimConfig(seed=77, num_vehicles=4)
        stamp = collect_stamp(config)
        assert stamp.config_dict["seed"] == 77
        assert stamp.config_dict["num_vehicles"] == 4
        assert "comms" in stamp.config_dict

    def test_stamp_to_dict_serializable(self):
        config = SimConfig()
        stamp = collect_stamp(config)
        d = stamp.to_dict()
        assert isinstance(d, dict)
        assert d["seed"] == config.seed
        serialized = json.dumps(d, default=str)
        assert len(serialized) > 0

    def test_stamp_git_graceful_fallback(self):
        """If git is not available, stamp should still work."""
        config = SimConfig()
        with patch("convoy_commander.stamp.subprocess.check_output",
                   side_effect=FileNotFoundError):
            stamp = collect_stamp(config)
        assert stamp.git_commit == "unknown"
        assert stamp.git_dirty is False

    def test_stamp_has_package_version(self):
        config = SimConfig()
        stamp = collect_stamp(config)
        assert stamp.package_version
        assert "." in stamp.package_version

    def test_sim_result_has_stamp(self):
        """SimResult should carry the stamp after a run."""
        config = SimConfig(seed=42, duration=2.0, num_vehicles=2)
        config.world.obstacle_count = 2
        config.world.nogo_zone_count = 1
        runner = SimRunner(config)
        result = runner.run()
        assert result.stamp is not None
        assert result.stamp.seed == 42


# ===========================================================================
# 2. Evaluation Harness
# ===========================================================================


class TestEvaluationHarness:
    """Tests for the evaluation harness."""

    def test_run_evaluation_minimal(self, tmp_path: Path):
        """Run evaluation with 1 scenario x 1 seed."""
        summary, summary_dir = run_evaluation(
            scenarios=["baseline"],
            seeds=[42],
            duration=5.0,
            num_vehicles=3,
            output_base=tmp_path,
        )
        assert summary.total_runs == 1
        assert len(summary.runs) == 1
        assert (summary_dir / "summary.md").exists()
        assert (summary_dir / "summary.json").exists()

    def test_run_evaluation_creates_per_run_reports(self, tmp_path: Path):
        """Each run should produce its own report.md and metrics.json."""
        summary, summary_dir = run_evaluation(
            scenarios=["baseline"],
            seeds=[42, 7],
            duration=5.0,
            num_vehicles=3,
            output_base=tmp_path,
        )
        for r in summary.runs:
            run_dir = summary_dir / r.run_dir
            assert (run_dir / "report.md").exists()
            assert (run_dir / "metrics.json").exists()

    def test_summary_md_contains_table(self, tmp_path: Path):
        summary, summary_dir = run_evaluation(
            scenarios=["baseline"],
            seeds=[42],
            duration=5.0,
            num_vehicles=3,
            output_base=tmp_path,
        )
        md = (summary_dir / "summary.md").read_text()
        assert "Scenario" in md
        assert "Seed" in md
        assert "baseline" in md

    def test_summary_json_parseable(self, tmp_path: Path):
        summary, summary_dir = run_evaluation(
            scenarios=["baseline"],
            seeds=[42],
            duration=5.0,
            num_vehicles=3,
            output_base=tmp_path,
        )
        data = json.loads((summary_dir / "summary.json").read_text())
        assert data["total_runs"] == 1
        assert isinstance(data["runs"], list)

    def test_evaluation_multiple_scenarios(self, tmp_path: Path):
        """Run 2 scenarios x 2 seeds = 4 runs."""
        summary, _ = run_evaluation(
            scenarios=["baseline", "gps_denied"],
            seeds=[42, 7],
            duration=5.0,
            num_vehicles=3,
            output_base=tmp_path,
        )
        assert summary.total_runs == 4
        scenarios_seen = {r.scenario for r in summary.runs}
        assert scenarios_seen == {"baseline", "gps_denied"}

    def test_evaluation_success_rate_computed(self, tmp_path: Path):
        summary, _ = run_evaluation(
            scenarios=["baseline"],
            seeds=[42],
            duration=5.0,
            num_vehicles=3,
            output_base=tmp_path,
        )
        assert 0 <= summary.overall_success_rate <= 100

    def test_evaluation_per_run_has_config_json(self, tmp_path: Path):
        """Reproducibility stamp (config.json) should appear in each run."""
        summary, summary_dir = run_evaluation(
            scenarios=["baseline"],
            seeds=[42],
            duration=5.0,
            num_vehicles=3,
            output_base=tmp_path,
        )
        for r in summary.runs:
            run_dir = summary_dir / r.run_dir
            assert (run_dir / "config.json").exists()
            stamp = json.loads((run_dir / "config.json").read_text())
            assert "git_commit" in stamp
            assert "python_version" in stamp


# ===========================================================================
# 3. Report & Performance Notes
# ===========================================================================


class TestReportEnhancements:
    """Tests for report stamp and performance notes."""

    def test_report_contains_reproducibility_section(self, tmp_path: Path):
        config = SimConfig(seed=42, duration=3.0, num_vehicles=2)
        config.world.obstacle_count = 2
        config.world.nogo_zone_count = 1
        runner = SimRunner(config)
        result = runner.run()
        from convoy_commander.viz.report import generate_report
        report_path = generate_report(result, tmp_path / "test_run")
        md = report_path.read_text()
        assert "Reproducibility" in md
        assert "Git commit" in md

    def test_report_contains_performance_notes(self, tmp_path: Path):
        config = SimConfig(seed=42, duration=3.0, num_vehicles=2)
        config.world.obstacle_count = 2
        config.world.nogo_zone_count = 1
        runner = SimRunner(config)
        result = runner.run()
        from convoy_commander.viz.report import generate_report
        report_path = generate_report(result, tmp_path / "test_run")
        md = report_path.read_text()
        assert "Performance Notes" in md
        assert "Complexity" in md

    def test_config_json_saved(self, tmp_path: Path):
        config = SimConfig(seed=42, duration=3.0, num_vehicles=2)
        config.world.obstacle_count = 2
        config.world.nogo_zone_count = 1
        runner = SimRunner(config)
        result = runner.run()
        from convoy_commander.viz.report import generate_report
        generate_report(result, tmp_path / "test_run")
        config_path = tmp_path / "test_run" / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["seed"] == 42


# ===========================================================================
# 4. CLI
# ===========================================================================


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_report_last_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "convoy_commander", "report", "--help"],
            capture_output=True, text=True,
        )
        assert "--last" in result.stdout

    def test_cli_evaluate_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "convoy_commander", "evaluate", "--help"],
            capture_output=True, text=True,
        )
        assert "--scenarios" in result.stdout
        assert "--seeds" in result.stdout
        assert "--duration" in result.stdout
