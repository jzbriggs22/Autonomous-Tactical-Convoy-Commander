"""Tests for forecast integration in the drift scheduler."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from ai_governance.config import GovernanceConfig
from ai_governance.scheduler import DriftScheduler
from ai_governance.storage import GovernanceDB


def _cfg():
    return GovernanceConfig.default_customer_service()


class _CapturingDispatcher:
    """Test double that records dispatched payloads."""

    def __init__(self):
        self.payloads = []
        self._lock = threading.Lock()

    def dispatch_async(self, payload: dict) -> None:
        with self._lock:
            self.payloads.append(payload)

    @property
    def forecast_payloads(self):
        with self._lock:
            return [p for p in self.payloads if p.get("source") == "drift_forecaster"]


def _seed_rising_metric(db, agent_id, category, metric, *, baseline, points, step):
    base = datetime.now(timezone.utc) - timedelta(hours=points)
    with db._lock:
        db._conn.execute(
            """INSERT OR REPLACE INTO baselines
               (agent_id, category, metric, value, sample_count, computed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (agent_id, category, metric, baseline, 200,
             datetime.now(timezone.utc).isoformat()),
        )
        for i in range(points):
            ts = base + timedelta(hours=i)
            db._conn.execute(
                """INSERT INTO metric_snapshots
                   (agent_id, category, metric, value, sample_count, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (agent_id, category, metric, baseline + i * step, 100, ts.isoformat()),
            )
        db._conn.commit()


class TestSchedulerForecastStep:
    def test_forecast_disabled_returns_zero(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        sched = DriftScheduler(cfg, db, forecast_enabled=False)
        assert sched._run_forecast_step() == 0

    def test_forecast_empty_db_returns_zero(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        sched = DriftScheduler(cfg, db)
        assert sched._run_forecast_step() == 0

    def test_run_once_includes_forecast_count(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        sched = DriftScheduler(cfg, db)
        result = sched.run_once()
        assert result["status"] == "ok"
        assert "forecasts_flagged" in result
        assert result["forecasts_flagged"] == 0

    def test_stats_track_forecasts(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        sched = DriftScheduler(cfg, db)
        sched.run_once()
        assert sched.stats.total_forecasts_flagged == 0

    def test_forecast_failure_does_not_break_cycle(self, monkeypatch):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        sched = DriftScheduler(cfg, db)

        def _boom(self, category=None):
            raise RuntimeError("forecast exploded")

        import ai_governance.forecast as fc
        monkeypatch.setattr(fc.DriftForecaster, "forecast", _boom)
        result = sched.run_once()
        assert result["status"] == "ok"
        assert result["forecasts_flagged"] == 0

    def test_forecast_dispatches_webhook(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        dispatcher = _CapturingDispatcher()
        # steadily rising metric that will breach within horizon
        _seed_rising_metric(
            db, cfg.agent_id, "billing_dispute", "escalation_rate",
            baseline=0.10, points=20, step=0.002,
        )
        sched = DriftScheduler(
            cfg, db, webhooks=dispatcher, forecast_horizon_hours=100.0,
        )
        flagged = sched._run_forecast_step()
        # If the trend produced forecasts, they must have been dispatched
        assert flagged == len(dispatcher.forecast_payloads)
        for p in dispatcher.forecast_payloads:
            assert p["source"] == "drift_forecaster"
            assert p["agent_id"] == cfg.agent_id
            assert "message" in p
            assert p["severity"] in ("imminent", "near", "approaching")

    def test_forecast_cooldown_suppresses_repeat_notifications(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        dispatcher = _CapturingDispatcher()
        _seed_rising_metric(
            db, cfg.agent_id, "billing_dispute", "escalation_rate",
            baseline=0.10, points=20, step=0.002,
        )
        sched = DriftScheduler(
            cfg, db, webhooks=dispatcher,
            forecast_horizon_hours=100.0,
            forecast_realert_seconds=3600.0,
        )
        first = sched._run_forecast_step()
        second = sched._run_forecast_step()
        # second run within the cooldown window notifies nothing new
        assert second == 0
        assert len(dispatcher.forecast_payloads) == first

    def test_forecast_realert_after_cooldown(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        dispatcher = _CapturingDispatcher()
        _seed_rising_metric(
            db, cfg.agent_id, "billing_dispute", "escalation_rate",
            baseline=0.10, points=20, step=0.002,
        )
        sched = DriftScheduler(
            cfg, db, webhooks=dispatcher,
            forecast_horizon_hours=100.0,
            forecast_realert_seconds=0.0,  # cooldown disabled
        )
        first = sched._run_forecast_step()
        second = sched._run_forecast_step()
        assert second == first  # re-notifies immediately with zero cooldown

    def test_scheduler_backwards_compatible_defaults(self):
        """Constructing without forecast kwargs must still work."""
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        assert sched._forecast_enabled is True
        result = sched.run_once()
        assert result["status"] == "ok"
