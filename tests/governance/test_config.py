"""Tests for GovernanceConfig, validators, default config, and serialization."""

import os
import tempfile

import pytest

from ai_governance.config import (
    AlertSeverity,
    DriftThreshold,
    GovernanceConfig,
    HighRiskPattern,
    MetricDirection,
    RollbackCondition,
)


class TestHighRiskPattern:
    def test_valid_pattern(self):
        p = HighRiskPattern(name="test", field="case_category", pattern=r"fraud")
        assert p.weight == 1.0

    def test_invalid_regex_raises(self):
        with pytest.raises(Exception):
            HighRiskPattern(name="test", field="x", pattern=r"[unclosed")

    def test_weight_bounds(self):
        with pytest.raises(Exception):
            HighRiskPattern(name="t", field="x", pattern="x", weight=0.0)
        with pytest.raises(Exception):
            HighRiskPattern(name="t", field="x", pattern="x", weight=11.0)


class TestDriftThreshold:
    def test_max_delta_must_be_positive(self):
        with pytest.raises(Exception):
            DriftThreshold(name="t", metric="resolution_rate", max_delta=0.0)

    def test_wildcard_category(self):
        t = DriftThreshold(name="t", category="*", metric="escalation_rate", max_delta=0.1)
        assert t.category == "*"

    def test_defaults(self):
        t = DriftThreshold(name="t", metric="resolution_rate", max_delta=0.05)
        assert t.direction == MetricDirection.EITHER
        assert t.severity == AlertSeverity.WARN


class TestRollbackCondition:
    def test_valid_expression(self):
        c = RollbackCondition(
            name="test",
            description="desc",
            expression="x < 0.5 and y > 10",
        )
        assert c.expression == "x < 0.5 and y > 10"

    def test_forbidden_token_import(self):
        with pytest.raises(Exception):
            RollbackCondition(
                name="t", description="d",
                expression="import os; os.unlink('/')",
            )

    def test_forbidden_token_dunder(self):
        with pytest.raises(Exception):
            RollbackCondition(
                name="t", description="d",
                expression="__class__.__mro__",
            )

    def test_invalid_syntax_raises(self):
        with pytest.raises(Exception):
            RollbackCondition(
                name="t", description="d",
                expression="if x: pass",  # statement, not expression
            )

    def test_cooldown_minimum(self):
        with pytest.raises(Exception):
            RollbackCondition(
                name="t", description="d",
                expression="x > 0",
                cooldown_seconds=10,  # below 60
            )


class TestGovernanceConfig:
    def test_default_customer_service_valid(self):
        cfg = GovernanceConfig.default_customer_service()
        assert len(cfg.high_risk_patterns) >= 3
        assert len(cfg.drift_thresholds) >= 3
        assert len(cfg.rollback_conditions) >= 1
        assert cfg.agent_id == "cs-agent-v1"

    def test_min_baseline_events_minimum(self):
        with pytest.raises(Exception):
            GovernanceConfig(min_baseline_events=2)

    def test_recent_window_minimum(self):
        with pytest.raises(Exception):
            GovernanceConfig(recent_window_size=5)

    def test_all_threshold_metrics_are_known(self):
        from ai_governance.drift import _EXTRACTORS
        cfg = GovernanceConfig.default_customer_service()
        for thr in cfg.drift_thresholds:
            assert thr.metric in _EXTRACTORS, (
                f"Threshold '{thr.name}' references unknown metric '{thr.metric}'"
            )


class TestConfigSerialization:
    def test_yaml_roundtrip(self):
        original = GovernanceConfig.default_customer_service()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        try:
            original.to_yaml(path)
            loaded = GovernanceConfig.from_yaml(path)
            assert loaded.agent_id == original.agent_id
            assert len(loaded.high_risk_patterns) == len(original.high_risk_patterns)
            assert len(loaded.drift_thresholds) == len(original.drift_thresholds)
            assert len(loaded.rollback_conditions) == len(original.rollback_conditions)
            assert loaded.min_baseline_events == original.min_baseline_events
        finally:
            os.unlink(path)

    def test_json_roundtrip(self):
        original = GovernanceConfig.default_customer_service()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            original.to_json(path)
            loaded = GovernanceConfig.from_json(path)
            assert loaded.agent_id == original.agent_id
            assert len(loaded.drift_thresholds) == len(original.drift_thresholds)
        finally:
            os.unlink(path)

    def test_yaml_to_string(self):
        cfg = GovernanceConfig.default_customer_service()
        text = cfg.to_yaml()
        assert "agent_id: cs-agent-v1" in text
        assert "fraud_claim" in text

    def test_json_to_string(self):
        cfg = GovernanceConfig.default_customer_service()
        text = cfg.to_json()
        assert '"agent_id"' in text
        import json
        data = json.loads(text)
        assert data["agent_id"] == "cs-agent-v1"

    def test_shipped_yaml_config_loads(self):
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "ai_governance", "default_config.yaml",
        )
        if os.path.exists(cfg_path):
            cfg = GovernanceConfig.from_yaml(cfg_path)
            assert cfg.agent_id == "cs-agent-v1"
            assert len(cfg.drift_thresholds) >= 3

    def test_yaml_preserves_enum_values(self):
        cfg = GovernanceConfig.default_customer_service()
        text = cfg.to_yaml()
        loaded = GovernanceConfig.from_yaml.__func__.__code__  # just check text
        assert "decrease" in text
        assert "critical" in text
        assert "rollback" in text


class TestConfigInputValidation:
    def test_yaml_rejects_non_mapping(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("- just a list\n- not a mapping\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            GovernanceConfig.from_yaml(f)

    def test_json_rejects_non_object(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="JSON object"):
            GovernanceConfig.from_json(f)

    def test_yaml_rejects_oversized_file(self, tmp_path):
        f = tmp_path / "huge.yaml"
        f.write_text("x: " + "a" * 2_000_000)
        with pytest.raises(ValueError, match="too large"):
            GovernanceConfig.from_yaml(f)

    def test_json_rejects_oversized_file(self, tmp_path):
        f = tmp_path / "huge.json"
        f.write_text('{"x": "' + "a" * 2_000_000 + '"}')
        with pytest.raises(ValueError, match="too large"):
            GovernanceConfig.from_json(f)

    def test_yaml_rejects_invalid_yaml(self, tmp_path):
        f = tmp_path / "invalid.yaml"
        f.write_text("}{not yaml at all")
        with pytest.raises(Exception):
            GovernanceConfig.from_yaml(f)

    def test_json_rejects_invalid_json(self, tmp_path):
        f = tmp_path / "invalid.json"
        f.write_text("}{not json")
        with pytest.raises(Exception):
            GovernanceConfig.from_json(f)

    def test_yaml_empty_dict_uses_defaults(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("{}")
        cfg = GovernanceConfig.from_yaml(f)
        assert cfg.agent_id == "default-agent"

    def test_json_empty_dict_uses_defaults(self, tmp_path):
        import json
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({}))
        cfg = GovernanceConfig.from_json(f)
        assert cfg.agent_id == "default-agent"
