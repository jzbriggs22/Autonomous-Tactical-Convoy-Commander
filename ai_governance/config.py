"""Governance configuration: drift thresholds, high-risk patterns, rollback rules.

Load this before the agent starts. It defines what "safe" means for this deployment.
"""

from __future__ import annotations

import ast
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HighRiskPattern(BaseModel):
    """Identifies high-risk cases by matching a metadata field against a regex."""

    name: str
    field: str  # field name in case metadata or the literal "case_category"
    pattern: str  # regex; case-insensitive match flags are applied at compile time
    weight: float = Field(default=1.0, ge=0.1, le=10.0)

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, v: str) -> str:
        re.compile(v)
        return v


class MetricDirection(str, Enum):
    INCREASE = "increase"  # flag when metric rises above threshold
    DECREASE = "decrease"  # flag when metric falls below threshold
    EITHER = "either"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"
    ROLLBACK = "rollback"


class DriftThreshold(BaseModel):
    """One drift threshold. A violation fires an alert at the given severity."""

    name: str
    category: str = "*"  # case category to watch; "*" matches all categories
    metric: str  # e.g. "resolution_rate", "escalation_rate", "high_risk_accuracy"
    max_delta: float = Field(gt=0.0)  # absolute change from baseline that triggers
    direction: MetricDirection = MetricDirection.EITHER
    min_baseline_samples: int = Field(default=30, ge=5)
    recent_window: int = Field(default=50, ge=10)  # min events needed in recent window
    severity: AlertSeverity = AlertSeverity.WARN


class RollbackCondition(BaseModel):
    """Named condition that triggers rollback when its expression evaluates True.

    Expressions run in a restricted namespace containing per-category metric
    variables (e.g. fraud_claim_escalation_rate, high_risk_count). Only
    arithmetic comparisons and boolean operators are allowed.
    """

    name: str
    description: str
    expression: str
    action: AlertSeverity = AlertSeverity.ROLLBACK
    cooldown_seconds: int = Field(default=3600, ge=60)

    @field_validator("expression")
    @classmethod
    def _expression_safe(cls, v: str) -> str:
        _FORBIDDEN = ["import", "__", "eval", "exec", "open", "os.", "sys.", "getattr"]
        for tok in _FORBIDDEN:
            if tok in v:
                raise ValueError(f"Forbidden token in expression: {tok!r}")
        # Validate it parses as a Python expression (not a statement)
        try:
            ast.parse(v, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc}") from exc
        return v


class GovernanceConfig(BaseModel):
    """Top-level governance configuration. Load this file before the agent starts."""

    agent_id: str = "default-agent"
    version: str = "1.0.0"

    high_risk_patterns: list[HighRiskPattern] = Field(default_factory=list)
    drift_thresholds: list[DriftThreshold] = Field(default_factory=list)
    rollback_conditions: list[RollbackCondition] = Field(default_factory=list)

    baseline_window_days: int = Field(default=30, ge=1)
    min_baseline_events: int = Field(default=30, ge=5)
    recent_window_size: int = Field(default=100, ge=10)

    @classmethod
    def default_customer_service(cls) -> "GovernanceConfig":
        """Governance config for a customer-service agent handling billing, fraud, policy."""
        return cls(
            agent_id="cs-agent-v1",
            high_risk_patterns=[
                HighRiskPattern(
                    name="billing_dispute",
                    field="case_category",
                    pattern=r"billing_dispute",
                    weight=2.0,
                ),
                HighRiskPattern(
                    name="fraud_claim",
                    field="case_category",
                    pattern=r"fraud",
                    weight=3.0,
                ),
                HighRiskPattern(
                    name="policy_sensitive",
                    field="case_category",
                    pattern=r"policy|legal|compliance",
                    weight=2.5,
                ),
                HighRiskPattern(
                    name="high_value_account",
                    field="account_tier",
                    pattern=r"enterprise|premium",
                    weight=1.5,
                ),
            ],
            drift_thresholds=[
                DriftThreshold(
                    name="billing_resolution_drop",
                    category="billing_dispute",
                    metric="resolution_rate",
                    max_delta=0.10,
                    direction=MetricDirection.DECREASE,
                    severity=AlertSeverity.WARN,
                ),
                DriftThreshold(
                    name="billing_escalation_spike",
                    category="billing_dispute",
                    metric="escalation_rate",
                    max_delta=0.15,
                    direction=MetricDirection.INCREASE,
                    severity=AlertSeverity.CRITICAL,
                ),
                DriftThreshold(
                    name="fraud_escalation_drop",
                    category="fraud_claim",
                    metric="escalation_rate",
                    max_delta=0.20,
                    direction=MetricDirection.DECREASE,
                    min_baseline_samples=20,
                    recent_window=30,
                    severity=AlertSeverity.CRITICAL,
                ),
                DriftThreshold(
                    name="global_resolution_drift",
                    category="*",
                    metric="resolution_rate",
                    max_delta=0.12,
                    direction=MetricDirection.EITHER,
                    severity=AlertSeverity.WARN,
                ),
                DriftThreshold(
                    name="high_risk_accuracy_floor",
                    category="*",
                    metric="high_risk_accuracy",
                    max_delta=0.10,
                    min_baseline_samples=20,
                    recent_window=30,
                    direction=MetricDirection.DECREASE,
                    severity=AlertSeverity.ROLLBACK,
                ),
            ],
            rollback_conditions=[
                RollbackCondition(
                    name="fraud_underescalation",
                    description=(
                        "Agent is resolving fraud cases without escalation — "
                        "financial risk threshold breached"
                    ),
                    expression="fraud_claim_escalation_rate < 0.60 and fraud_claim_count >= 10",
                    action=AlertSeverity.ROLLBACK,
                    cooldown_seconds=1800,
                ),
                RollbackCondition(
                    name="accuracy_collapse",
                    description=(
                        "High-risk decision accuracy dropped below acceptable floor"
                    ),
                    expression="high_risk_accuracy < 0.70 and high_risk_count >= 20",
                    action=AlertSeverity.ROLLBACK,
                    cooldown_seconds=3600,
                ),
            ],
        )
