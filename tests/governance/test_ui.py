"""Tests for the PM-facing web dashboard endpoint."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth, configure, APIKey
from ai_governance.config import GovernanceConfig
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


class TestWebDashboard:
    def test_ui_returns_html(self, client):
        resp = client.get("/ui")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_ui_contains_governance_content(self, client):
        resp = client.get("/ui")
        text = resp.text
        assert "AI GOVERNANCE DASHBOARD" in text
        assert "Normal Performance" in text
        assert "Governance Signals" in text
        assert "safety-banner" in text

    def test_ui_has_auto_refresh_js(self, client):
        resp = client.get("/ui")
        assert "setInterval" in resp.text
        assert "5000" in resp.text  # 5 second refresh interval

    def test_ui_polls_dashboard_endpoint(self, client):
        resp = client.get("/ui")
        assert "/dashboard" in resp.text

    def test_ui_has_category_breakdown(self, client):
        resp = client.get("/ui")
        assert "Category Breakdown" in resp.text
        assert "cat-tbody" in resp.text

    def test_ui_has_alerts_section(self, client):
        resp = client.get("/ui")
        assert "Recent Alerts" in resp.text
        assert "alerts-list" in resp.text

    def test_ui_links_to_api_docs(self, client):
        resp = client.get("/ui")
        assert "/docs" in resp.text
        assert "/metrics" in resp.text
        assert "/audit" in resp.text

    def test_ui_accessible_without_auth_when_enabled(self, client):
        configure([APIKey(key="secret", role="admin")])
        resp = client.get("/ui")
        assert resp.status_code == 200

    def test_ui_shows_side_by_side_layout(self, client):
        resp = client.get("/ui")
        assert "Normal Performance" in resp.text
        assert "Governance Signals" in resp.text
        assert "grid-2" in resp.text

    def test_dashboard_api_returns_correct_shape(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_safe" in data
        assert "normal_metrics" in data
        assert "governance_signals" in data
        assert "category_breakdown" in data
        assert "recent_alerts" in data
        nm = data["normal_metrics"]
        assert "total_decisions" in nm
        assert "resolution_rate" in nm
        assert "escalation_rate" in nm
        assert "avg_response_time_ms" in nm
        gv = data["governance_signals"]
        assert "overall_drift_score" in gv
        assert "active_violations" in gv
        assert "rollback_events" in gv
        assert "high_risk_decisions_recent" in gv

    def test_dashboard_reflects_ingested_data(self, client):
        for i in range(10):
            client.post("/events", json={
                "case_id": str(uuid.uuid4()),
                "case_category": "fraud_claim",
                "decision": "escalate",
                "resolution_time_ms": 200,
            })
        resp = client.get("/dashboard")
        data = resp.json()
        assert data["normal_metrics"]["total_decisions"] >= 10
        assert data["governance_signals"]["high_risk_decisions_recent"] >= 10

    def test_dashboard_category_breakdown_populated(self, client):
        for i in range(5):
            client.post("/events", json={
                "case_id": str(uuid.uuid4()),
                "case_category": "billing_dispute",
                "decision": "resolve",
                "resolution_time_ms": 100,
            })
        resp = client.get("/dashboard")
        cats = resp.json()["category_breakdown"]
        names = [c["category"] for c in cats]
        assert "billing_dispute" in names
