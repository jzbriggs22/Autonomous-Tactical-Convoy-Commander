"""Tests for the drift scheduler background worker."""

from __future__ import annotations

import time
import threading
import uuid

import pytest

from ai_governance.config import GovernanceConfig
from ai_governance.scheduler import DriftScheduler, SchedulerStats
from ai_governance.storage import GovernanceDB
from ai_governance.webhooks import WebhookDispatcher


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


class TestSchedulerLifecycle:
    def test_start_stop(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        assert not sched.is_running
        sched.start()
        assert sched.is_running
        sched.stop()
        assert not sched.is_running

    def test_double_start_is_noop(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        sched.start()
        thread_id = sched._thread.ident
        sched.start()
        assert sched._thread.ident == thread_id
        sched.stop()

    def test_stop_without_start_is_safe(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        sched.stop()

    def test_run_once_without_start(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        result = sched.run_once()
        assert result["status"] == "ok"
        assert "drift_score" in result


class TestSchedulerExecution:
    def test_run_once_returns_metrics(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        result = sched.run_once()
        assert result["status"] == "ok"
        assert result["drift_score"] == 0.0
        assert "violations" in result
        assert "duration_ms" in result

    def test_stats_updated_after_run(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        sched.run_once()
        stats = sched.stats
        assert stats.total_runs == 1
        assert stats.last_run_at is not None
        assert stats.consecutive_failures == 0
        assert stats.last_run_duration_ms > 0

    def test_history_accumulates(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        sched.run_once()
        sched.run_once()
        sched.run_once()
        assert len(sched.history) == 3
        assert all(h["status"] == "ok" for h in sched.history)

    def test_history_bounded(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0, max_history=5)
        for _ in range(10):
            sched.run_once()
        assert len(sched.history) == 5
        assert sched.stats.total_runs == 10

    def test_background_thread_runs_cycle(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=0.1)
        sched.start()
        time.sleep(0.5)
        sched.stop()
        assert sched.stats.total_runs >= 2

    def test_webhook_dispatch_on_alerts(self, setup):
        cfg, db = setup
        webhooks = WebhookDispatcher()
        sched = DriftScheduler(cfg, db, interval_seconds=60.0, webhooks=webhooks)
        result = sched.run_once()
        assert result["status"] == "ok"


class TestSchedulerInterval:
    def test_get_interval(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=120.0)
        assert sched.interval == 120.0

    def test_set_interval(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=120.0)
        sched.interval = 60.0
        assert sched.interval == 60.0

    def test_set_interval_too_small_raises(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=120.0)
        with pytest.raises(ValueError):
            sched.interval = 0.5


class TestSchedulerErrorHandling:
    def test_failure_increments_stats(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        sched._detector = None  # force AttributeError
        result = sched.run_once()
        assert result["status"] == "error"
        assert "error" in result
        stats = sched.stats
        assert stats.total_failures == 1
        assert stats.consecutive_failures == 1

    def test_failure_then_success_resets_consecutive(self, setup):
        cfg, db = setup
        from ai_governance.drift import DriftDetector
        sched = DriftScheduler(cfg, db, interval_seconds=60.0)
        real_detector = sched._detector
        sched._detector = None
        sched.run_once()
        assert sched.stats.consecutive_failures == 1
        sched._detector = real_detector
        sched.run_once()
        assert sched.stats.consecutive_failures == 0
        assert sched.stats.total_failures == 1

    def test_thread_survives_error(self, setup):
        cfg, db = setup
        sched = DriftScheduler(cfg, db, interval_seconds=0.1)
        sched._detector = None
        sched.start()
        time.sleep(0.4)
        assert sched.is_running
        assert sched.stats.total_failures >= 2
        sched.stop()
