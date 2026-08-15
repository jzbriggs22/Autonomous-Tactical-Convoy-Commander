"""Prometheus metrics for governance observability.

Exposes counters, histograms, and gauges for all governance operations.
Wire into the API via instrument_app() or use the collectors directly.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


registry = CollectorRegistry()

# ── counters ────────────────────────────────────────────────────────────────

events_ingested = Counter(
    "governance_events_ingested_total",
    "Total decision events ingested",
    ["agent_id", "case_category", "decision"],
    registry=registry,
)

high_risk_events = Counter(
    "governance_high_risk_events_total",
    "Decision events classified as high-risk",
    ["agent_id", "case_category"],
    registry=registry,
)

alerts_fired = Counter(
    "governance_alerts_fired_total",
    "Alerts fired by the alert engine",
    ["agent_id", "severity", "rule_name"],
    registry=registry,
)

rollbacks_triggered = Counter(
    "governance_rollbacks_triggered_total",
    "Rollback events triggered",
    ["agent_id", "trigger_rule"],
    registry=registry,
)

drift_checks = Counter(
    "governance_drift_checks_total",
    "Number of drift detection runs",
    ["agent_id"],
    registry=registry,
)

auth_rejected = Counter(
    "governance_auth_rejected_total",
    "Requests rejected by auth middleware",
    ["reason"],
    registry=registry,
)

# ── histograms ──────────────────────────────────────────────────────────────

request_duration = Histogram(
    "governance_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint", "status_code"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=registry,
)

ingestion_duration = Histogram(
    "governance_ingestion_duration_seconds",
    "Time to ingest a single decision event",
    ["agent_id"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    registry=registry,
)

# ── gauges ──────────────────────────────────────────────────────────────────

agent_safe = Gauge(
    "governance_agent_safe",
    "1 if agent is safe to continue, 0 if not",
    ["agent_id"],
    registry=registry,
)

drift_score = Gauge(
    "governance_drift_score",
    "Overall drift score from last detection run",
    ["agent_id"],
    registry=registry,
)

active_rollbacks = Gauge(
    "governance_active_rollbacks",
    "Number of unresolved rollbacks",
    ["agent_id"],
    registry=registry,
)

decision_count = Gauge(
    "governance_decisions_stored",
    "Total decisions in DB",
    ["agent_id"],
    registry=registry,
)


def metrics_response() -> tuple[bytes, str]:
    """Generate Prometheus text exposition. Returns (body, content_type)."""
    return generate_latest(registry), CONTENT_TYPE_LATEST
