"""Tests for predictive drift forecasting."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.forecast import BreachForecast, DriftForecaster, ForecastReport
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


def _cfg():
    return GovernanceConfig.default_customer_service()


def _ingest_events(db: GovernanceDB, cfg: GovernanceConfig, category: str,
                   count: int, decision: str = "resolve", resolution_ms: int = 500):
    from ai_governance.ingestion import IngestRequest, IngestionLayer
    layer = IngestionLayer(cfg, db)
    for i in range(count):
        layer.ingest(IngestRequest(
            case_id=f"c-{i}",
            case_category=category,
            decision=decision,
            resolution_time_ms=resolution_ms,
        ))


def _seed_metric_history(db: GovernanceDB, agent_id: str, category: str,
                         metric: str, values: list[float],
                         base_time: datetime | None = None):
    """Insert synthetic metric history for trend analysis."""
    base = base_time or datetime.now(timezone.utc) - timedelta(hours=len(values))
    with db._lock:
        for i, val in enumerate(values):
            ts = base + timedelta(hours=i)
            db._conn.execute(
                """INSERT INTO metric_snapshots
                   (agent_id, category, metric, value, sample_count, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (agent_id, category, metric, val, 100, ts.isoformat()),
            )
        db._conn.commit()


def _seed_baseline(db: GovernanceDB, agent_id: str, category: str,
                   metric: str, value: float, samples: int = 200):
    with db._lock:
        db._conn.execute(
            """INSERT OR REPLACE INTO baselines
               (agent_id, category, metric, value, sample_count, computed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (agent_id, category, metric, value, samples,
             datetime.now(timezone.utc).isoformat()),
        )
        db._conn.commit()


class TestDriftForecaster:
    def test_empty_db_returns_no_forecasts(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        forecaster = DriftForecaster(cfg, db)
        report = forecaster.forecast()
        assert isinstance(report, ForecastReport)
        assert report.forecasts == []
        assert report.agent_id == cfg.agent_id

    def test_stable_metrics_no_forecasts(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        values = [0.85] * 10
        _seed_metric_history(db, cfg.agent_id, "billing_dispute", "resolution_rate", values)
        _seed_baseline(db, cfg.agent_id, "billing_dispute", "resolution_rate", 0.85)
        forecaster = DriftForecaster(cfg, db)
        report = forecaster.forecast()
        assert len(report.forecasts) == 0

    def test_rising_metric_generates_forecast(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        values = [0.85 + i * 0.005 for i in range(20)]
        _seed_metric_history(db, cfg.agent_id, "billing_dispute", "resolution_rate", values)
        _seed_baseline(db, cfg.agent_id, "billing_dispute", "resolution_rate", 0.85)
        forecaster = DriftForecaster(cfg, db, horizon_hours=48.0)
        report = forecaster.forecast()
        for f in report.forecasts:
            assert f.hours_to_breach is None or f.hours_to_breach > 0
            assert f.severity in ("imminent", "near", "approaching")

    def test_forecast_report_summary(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        forecaster = DriftForecaster(cfg, db)
        report = forecaster.forecast()
        summary = report.summary()
        assert "Predictive Drift Forecast" in summary
        assert cfg.agent_id in summary

    def test_has_imminent_property(self):
        report = ForecastReport(
            agent_id="test",
            generated_at="2024-01-01T00:00:00",
            horizon_hours=24.0,
            forecasts=[
                BreachForecast(
                    category="billing", metric="resolution_rate",
                    current_value=0.9, baseline_value=0.85,
                    threshold_delta=0.15, current_delta=0.05,
                    remaining_delta=0.10, slope_per_hour=0.01,
                    hours_to_breach=1.5, severity="imminent",
                    confidence="high", data_points=20,
                    message="Test message",
                ),
            ],
        )
        assert report.has_imminent is True
        assert report.has_approaching is True

    def test_has_imminent_false_when_no_imminent(self):
        report = ForecastReport(
            agent_id="test",
            generated_at="2024-01-01T00:00:00",
            horizon_hours=24.0,
            forecasts=[
                BreachForecast(
                    category="billing", metric="resolution_rate",
                    current_value=0.9, baseline_value=0.85,
                    threshold_delta=0.15, current_delta=0.05,
                    remaining_delta=0.10, slope_per_hour=0.01,
                    hours_to_breach=10.0, severity="approaching",
                    confidence="moderate", data_points=15,
                    message="Test message",
                ),
            ],
        )
        assert report.has_imminent is False
        assert report.has_approaching is True

    def test_no_forecasts_means_not_approaching(self):
        report = ForecastReport(
            agent_id="test",
            generated_at="2024-01-01T00:00:00",
            horizon_hours=24.0,
            forecasts=[],
        )
        assert report.has_approaching is False

    def test_category_filter(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        values = [0.85 + i * 0.01 for i in range(20)]
        _seed_metric_history(db, cfg.agent_id, "billing_dispute", "resolution_rate", values)
        _seed_baseline(db, cfg.agent_id, "billing_dispute", "resolution_rate", 0.85)
        forecaster = DriftForecaster(cfg, db, horizon_hours=48.0)
        report = forecaster.forecast(category="nonexistent_cat")
        assert len(report.forecasts) == 0

    def test_horizon_filtering(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        forecaster = DriftForecaster(cfg, db, horizon_hours=0.1)
        report = forecaster.forecast()
        assert len(report.forecasts) == 0

    def test_classify_urgency(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        f = DriftForecaster(cfg, db)
        assert f._classify_urgency(1.0) == "imminent"
        assert f._classify_urgency(1.99) == "imminent"
        assert f._classify_urgency(2.0) == "near"
        assert f._classify_urgency(5.0) == "near"
        assert f._classify_urgency(7.99) == "near"
        assert f._classify_urgency(8.0) == "approaching"
        assert f._classify_urgency(20.0) == "approaching"
        assert f._classify_urgency(None) == "approaching"

    def test_assess_confidence(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        f = DriftForecaster(cfg, db)
        assert f._assess_confidence(25, 0.0001) == "high"
        assert f._assess_confidence(20, 0.0009) == "high"
        assert f._assess_confidence(20, 0.01) == "moderate"
        assert f._assess_confidence(15, 0.01) == "moderate"
        assert f._assess_confidence(10, 0.01) == "moderate"
        assert f._assess_confidence(5, 0.01) == "low"
        assert f._assess_confidence(3, 0.0) == "low"

    def test_will_breach_direction_increase(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        f = DriftForecaster(cfg, db)
        assert f._will_breach(0.01, 0.05, 0.15, "increase") is True
        assert f._will_breach(-0.01, 0.05, 0.15, "increase") is False
        assert f._will_breach(0.01, -0.05, 0.15, "increase") is False

    def test_will_breach_direction_decrease(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        f = DriftForecaster(cfg, db)
        assert f._will_breach(-0.01, -0.05, 0.15, "decrease") is True
        assert f._will_breach(0.01, -0.05, 0.15, "decrease") is False
        assert f._will_breach(-0.01, 0.05, 0.15, "decrease") is False

    def test_will_breach_direction_either(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        f = DriftForecaster(cfg, db)
        assert f._will_breach(0.01, 0.05, 0.15, "either") is True
        assert f._will_breach(-0.01, -0.05, 0.15, "either") is True
        assert f._will_breach(0.0, 0.05, 0.15, "either") is False

    def test_build_message(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        f = DriftForecaster(cfg, db)
        msg = f._build_message("billing", "resolution_rate", 5.2, 0.05, 0.15, 0.01)
        assert "billing" in msg
        assert "resolution_rate" in msg
        assert "5.2 hours" in msg
        assert "rising" in msg

    def test_build_message_none_hours(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        f = DriftForecaster(cfg, db)
        msg = f._build_message("billing", "resolution_rate", None, 0.05, 0.15, -0.01)
        assert "indeterminate" in msg
        assert "falling" in msg

    def test_forecast_sorted_by_hours(self):
        report = ForecastReport(
            agent_id="test",
            generated_at="2024-01-01T00:00:00",
            horizon_hours=24.0,
            forecasts=[
                BreachForecast(
                    category="a", metric="m1",
                    current_value=0.9, baseline_value=0.85,
                    threshold_delta=0.15, current_delta=0.05,
                    remaining_delta=0.10, slope_per_hour=0.01,
                    hours_to_breach=10.0, severity="approaching",
                    confidence="moderate", data_points=15,
                    message="later",
                ),
                BreachForecast(
                    category="b", metric="m2",
                    current_value=0.9, baseline_value=0.85,
                    threshold_delta=0.15, current_delta=0.05,
                    remaining_delta=0.10, slope_per_hour=0.01,
                    hours_to_breach=1.0, severity="imminent",
                    confidence="high", data_points=20,
                    message="sooner",
                ),
            ],
        )
        assert report.forecasts[0].hours_to_breach == 10.0
        assert report.forecasts[1].hours_to_breach == 1.0


class TestForecastAPI:
    @pytest.fixture
    def client(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        return TestClient(app)

    def test_forecast_endpoint_empty(self, client):
        resp = client.get("/forecast")
        assert resp.status_code == 200
        data = resp.json()
        assert data["forecast_count"] == 0
        assert data["forecasts"] == []
        assert "summary" in data

    def test_forecast_endpoint_with_horizon(self, client):
        resp = client.get("/forecast?horizon_hours=12")
        assert resp.status_code == 200
        data = resp.json()
        assert data["horizon_hours"] == 12.0

    def test_forecast_endpoint_with_category(self, client):
        resp = client.get("/forecast?category=billing_dispute")
        assert resp.status_code == 200
        data = resp.json()
        assert "forecasts" in data

    def test_forecast_horizon_capped(self, client):
        resp = client.get("/forecast?horizon_hours=500")
        assert resp.status_code == 200
        data = resp.json()
        assert data["horizon_hours"] == 168.0
