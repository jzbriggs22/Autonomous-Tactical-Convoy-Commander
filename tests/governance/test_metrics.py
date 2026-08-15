"""Tests for Prometheus metrics integration."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.metrics import (
    registry,
    events_ingested,
    high_risk_events,
    alerts_fired,
    drift_checks,
    drift_score,
    agent_safe,
    ingestion_duration,
    metrics_response,
)
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean_auth():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 10
    cfg.recent_window_size = 30
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 10
        thr.recent_window = 15
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"] or "openmetrics" in resp.headers["content-type"]
        text = resp.text
        assert "governance_events_ingested_total" in text or "governance_" in text

    def test_metrics_endpoint_accessible(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200


class TestMetricsCollection:
    def test_metrics_response_returns_bytes(self):
        body, content_type = metrics_response()
        assert isinstance(body, bytes)
        assert "text/" in content_type

    def test_event_ingestion_increments_counter(self, client):
        before = self._get_counter_value(
            "governance_events_ingested_total",
            agent_id="cs-agent-v1",
            case_category="billing_dispute",
            decision="resolve",
        )
        client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": "billing_dispute",
            "decision": "resolve",
            "resolution_time_ms": 100,
        })
        after = self._get_counter_value(
            "governance_events_ingested_total",
            agent_id="cs-agent-v1",
            case_category="billing_dispute",
            decision="resolve",
        )
        assert after > before

    def test_high_risk_event_tracked(self, client):
        before = self._get_counter_value(
            "governance_high_risk_events_total",
            agent_id="cs-agent-v1",
            case_category="fraud_claim",
        )
        client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": "fraud_claim",
            "decision": "escalate",
            "resolution_time_ms": 100,
        })
        after = self._get_counter_value(
            "governance_high_risk_events_total",
            agent_id="cs-agent-v1",
            case_category="fraud_claim",
        )
        assert after > before

    def test_drift_check_increments(self, client):
        before = self._get_counter_value(
            "governance_drift_checks_total",
            agent_id="cs-agent-v1",
        )
        client.get("/drift")
        after = self._get_counter_value(
            "governance_drift_checks_total",
            agent_id="cs-agent-v1",
        )
        assert after == before + 1

    def test_ingestion_duration_recorded(self, client):
        client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": "returns",
            "decision": "resolve",
            "resolution_time_ms": 50,
        })
        body, _ = metrics_response()
        text = body.decode()
        assert "governance_ingestion_duration_seconds" in text

    def test_drift_score_gauge_set(self, client):
        client.get("/drift")
        body, _ = metrics_response()
        text = body.decode()
        assert "governance_drift_score" in text

    def test_agent_safe_gauge_set(self, client):
        client.get("/drift")
        body, _ = metrics_response()
        text = body.decode()
        assert "governance_agent_safe" in text

    @staticmethod
    def _get_counter_value(name: str, **labels) -> float:
        body, _ = metrics_response()
        text = body.decode()
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        target = f"{name}{{{label_str}}}"
        for line in text.split("\n"):
            if line.startswith(target):
                return float(line.split()[-1])
        return 0.0
