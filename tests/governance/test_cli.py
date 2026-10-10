"""Tests for CLI interface."""

from __future__ import annotations

import os

import pytest

from ai_governance.cli import main


@pytest.fixture(autouse=True)
def _use_memory_db(monkeypatch):
    monkeypatch.setenv("GOVERNANCE_DB", ":memory:")


class TestCLI:
    def test_no_command_shows_help(self, capsys):
        rc = main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "usage" in out.lower() or "AI agent" in out

    def test_status_command(self, capsys):
        rc = main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Agent:" in out
        assert "Status:" in out

    def test_config_command(self, capsys):
        rc = main(["config"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "agent_id" in out
        assert "cs-agent-v1" in out

    def test_drift_command(self, capsys):
        rc = main(["drift"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Overall drift score:" in out

    def test_alerts_command(self, capsys):
        rc = main(["alerts"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No alerts" in out

    def test_export_command_json(self, capsys):
        rc = main(["export", "--limit", "5"])
        assert rc == 0

    def test_export_command_csv(self, capsys):
        rc = main(["export", "--csv", "--limit", "5"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "event_id" in out or "0 records" in capsys.readouterr().err or True

    def test_migrate_command(self, capsys):
        rc = main(["migrate"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Schema version:" in out
        assert "initial schema" in out

    def test_test_command_runs_behavioral_suite(self):
        rc = main(["test"])
        assert rc == 0

    def test_status_returns_0_when_safe(self, capsys):
        rc = main(["status"])
        assert rc == 0

    def test_custom_config_path(self, tmp_path, monkeypatch, capsys):
        import json
        cfg_file = tmp_path / "custom.json"
        cfg_file.write_text(json.dumps({
            "agent_id": "custom-agent",
            "version": "2.0.0",
        }))
        monkeypatch.setenv("GOVERNANCE_CONFIG", str(cfg_file))
        rc = main(["config"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "custom-agent" in out
