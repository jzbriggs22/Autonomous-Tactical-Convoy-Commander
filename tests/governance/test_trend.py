"""Tests for drift trend analysis."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.ingestion import IngestionLayer, IngestRequest
from ai_governance.storage import GovernanceDB
from ai_governance.trend import TrendAnalyzer, TrendDirection, _linreg


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def setup():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 5
    cfg.recent_window_size = 20
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
        thr.recent_window = 10
    db = GovernanceDB(":memory:")
    return cfg, db


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 5
    cfg.recent_window_size = 20
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
        thr.recent_window = 10
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestLinearRegression:
    def test_perfect_line(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 2.0, 3.0, 4.0]
        slope, intercept = _linreg(xs, ys)
        assert abs(slope - 1.0) < 1e-9
        assert abs(intercept - 1.0) < 1e-9

    def test_flat_line(self):
        xs = [0.0, 1.0, 2.0]
        ys = [5.0, 5.0, 5.0]
        slope, _ = _linreg(xs, ys)
        assert abs(slope) < 1e-9

    def test_single_point(self):
        slope, _ = _linreg([0.0], [3.0])
        assert slope == 0.0

    def test_two_points(self):
        slope, _ = _linreg([0.0, 2.0], [0.0, 4.0])
        assert abs(slope - 2.0) < 1e-9


class TestTrendAnalyzer:
    def test_empty_db_empty_report(self, setup):
        cfg, db = setup
        analyzer = TrendAnalyzer(cfg, db, history_limit=30)
        report = analyzer.analyze()
        assert report.agent_id == cfg.agent_id
        assert len(report.accelerating) == 0
        assert len(report.rising) == 0
        assert report.overall_trajectory == "stable"

    def test_classify_stable(self, setup):
        cfg, db = setup
        analyzer = TrendAnalyzer(cfg, db)
        direction = analyzer._classify(0.0001, 0.0)
        assert direction == TrendDirection.STABLE

    def test_classify_rising(self, setup):
        cfg, db = setup
        analyzer = TrendAnalyzer(cfg, db)
        direction = analyzer._classify(0.01, 0.0)
        assert direction == TrendDirection.RISING

    def test_classify_accelerating(self, setup):
        cfg, db = setup
        analyzer = TrendAnalyzer(cfg, db)
        direction = analyzer._classify(0.01, 0.001)
        assert direction == TrendDirection.ACCELERATING

    def test_classify_recovering(self, setup):
        cfg, db = setup
        analyzer = TrendAnalyzer(cfg, db)
        direction = analyzer._classify(-0.01, -0.001)
        assert direction == TrendDirection.RECOVERING

    def test_report_with_history(self, setup):
        cfg, db = setup
        agent_id = cfg.agent_id
        for i in range(10):
            db.insert_metric_snapshot(agent_id, "billing_dispute", "resolution_rate", 0.8 + i * 0.01, 10)
        analyzer = TrendAnalyzer(cfg, db, history_limit=30)
        report = analyzer.analyze(category="billing_dispute")
        assert report.overall_trajectory in {"stable", "rising", "accelerating", "recovering"}

    def test_summary_format(self, setup):
        cfg, db = setup
        analyzer = TrendAnalyzer(cfg, db)
        report = analyzer.analyze()
        text = report.summary()
        assert "Trend Report" in text
        assert cfg.agent_id in text
        assert "stable" in text.lower() or "accelerating" in text.lower()

    def test_has_concerns_false_when_stable(self, setup):
        cfg, db = setup
        analyzer = TrendAnalyzer(cfg, db)
        report = analyzer.analyze()
        assert report.has_concerns is False


class TestTrendsAPI:
    def test_trends_endpoint(self, client):
        resp = client.get("/trends")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_id" in data
        assert "overall_trajectory" in data
        assert "has_concerns" in data
        assert "accelerating" in data
        assert "rising" in data
        assert "stable" in data
        assert "summary" in data

    def test_trends_with_category_filter(self, client):
        resp = client.get("/trends?category=billing_dispute")
        assert resp.status_code == 200

    def test_trends_with_limit(self, client):
        resp = client.get("/trends?limit=10")
        assert resp.status_code == 200

    def test_trends_with_history(self, client):
        for _ in range(5):
            client.post("/events", json={
                "case_id": str(uuid.uuid4()),
                "case_category": "billing_dispute",
                "decision": "resolve",
                "resolution_time_ms": 100,
            })
        client.get("/drift")
        resp = client.get("/trends")
        assert resp.status_code == 200
