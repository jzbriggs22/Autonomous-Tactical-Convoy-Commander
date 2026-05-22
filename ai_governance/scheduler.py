"""Scheduled drift detection and health monitoring.

Runs drift detection at configurable intervals in a background thread.
Fires alerts and webhooks automatically — no manual API calls needed.

Usage:
    scheduler = DriftScheduler(config, db, interval_seconds=300)
    scheduler.start()
    # ... later ...
    scheduler.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .alerts import AlertEngine
from .config import GovernanceConfig
from .drift import DriftDetector, DriftReport
from .storage import GovernanceDB
from .webhooks import WebhookDispatcher

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    total_runs: int = 0
    total_violations_found: int = 0
    total_alerts_fired: int = 0
    total_rollbacks_triggered: int = 0
    last_run_at: Optional[datetime] = None
    last_drift_score: float = 0.0
    last_run_duration_ms: float = 0.0
    consecutive_failures: int = 0
    total_failures: int = 0


class DriftScheduler:
    """Background drift detection on a fixed interval.

    Thread-safe: start/stop can be called from any thread.
    The scheduler catches all exceptions during runs to avoid
    crashing the background thread.
    """

    def __init__(
        self,
        config: GovernanceConfig,
        db: GovernanceDB,
        *,
        interval_seconds: float = 300.0,
        webhooks: Optional[WebhookDispatcher] = None,
        max_history: int = 100,
    ) -> None:
        self._config = config
        self._db = db
        self._detector = DriftDetector(config, db)
        self._engine = AlertEngine(config, db)
        self._webhooks = webhooks
        self._interval = interval_seconds
        self._max_history = max_history

        self._stats = SchedulerStats()
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> SchedulerStats:
        with self._lock:
            return SchedulerStats(
                total_runs=self._stats.total_runs,
                total_violations_found=self._stats.total_violations_found,
                total_alerts_fired=self._stats.total_alerts_fired,
                total_rollbacks_triggered=self._stats.total_rollbacks_triggered,
                last_run_at=self._stats.last_run_at,
                last_drift_score=self._stats.last_drift_score,
                last_run_duration_ms=self._stats.last_run_duration_ms,
                consecutive_failures=self._stats.consecutive_failures,
                total_failures=self._stats.total_failures,
            )

    @property
    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, value: float) -> None:
        if value < 1.0:
            raise ValueError("Interval must be at least 1 second")
        self._interval = value

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="drift-scheduler",
        )
        self._thread.start()
        logger.info(
            "Drift scheduler started (interval=%.1fs, agent=%s)",
            self._interval, self._config.agent_id,
        )

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("Drift scheduler stopped")

    def run_once(self) -> dict:
        """Execute a single drift detection cycle. Returns the run result."""
        return self._execute_cycle()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._execute_cycle()
            self._stop_event.wait(timeout=self._interval)

    def _execute_cycle(self) -> dict:
        t0 = time.monotonic()
        now = datetime.now(timezone.utc)
        result: dict = {"timestamp": now.isoformat(), "status": "ok"}

        try:
            report = self._detector.detect()
            fired = self._engine.evaluate(report)

            duration_ms = (time.monotonic() - t0) * 1000
            rollback_count = sum(1 for f in fired if f.triggered_rollback)

            result.update({
                "drift_score": round(report.overall_drift_score, 4),
                "violations": len(report.violations),
                "alerts_fired": len(fired),
                "rollbacks_triggered": rollback_count,
                "categories_analyzed": report.categories_analyzed,
                "duration_ms": round(duration_ms, 2),
            })

            if self._webhooks and fired:
                for f in fired:
                    self._webhooks.dispatch_async({
                        "source": "drift_scheduler",
                        "alert_id": f.alert_id,
                        "rule_name": f.rule_name,
                        "severity": f.severity,
                        "message": f.message,
                        "triggered_rollback": f.triggered_rollback,
                        "agent_id": self._config.agent_id,
                    })

            with self._lock:
                self._stats.total_runs += 1
                self._stats.total_violations_found += len(report.violations)
                self._stats.total_alerts_fired += len(fired)
                self._stats.total_rollbacks_triggered += rollback_count
                self._stats.last_run_at = now
                self._stats.last_drift_score = report.overall_drift_score
                self._stats.last_run_duration_ms = duration_ms
                self._stats.consecutive_failures = 0
                self._history.append(result)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            logger.exception("Drift scheduler cycle failed")
            result.update({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": round(duration_ms, 2),
            })
            with self._lock:
                self._stats.total_failures += 1
                self._stats.consecutive_failures += 1
                self._stats.last_run_at = now
                self._stats.last_run_duration_ms = duration_ms
                self._history.append(result)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        return result
