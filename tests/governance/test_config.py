"""Tests for GovernanceConfig, validators, and default config."""

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
