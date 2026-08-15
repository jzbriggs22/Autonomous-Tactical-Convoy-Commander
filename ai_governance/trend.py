"""Drift trend analysis: detect acceleration before thresholds are breached.

Fits a linear regression to the most recent metric_snapshot history and
classifies the trajectory as: stable, warning, or accelerating. Provides
early warning when drift is rising even if it hasn't crossed the configured
max_delta yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import GovernanceConfig
from .storage import GovernanceDB


class TrendDirection(str):
    STABLE = "stable"
    RISING = "rising"
    FALLING = "falling"
    ACCELERATING = "accelerating"
    RECOVERING = "recovering"


@dataclass
class MetricTrend:
    agent_id: str
    category: str
    metric: str
    direction: str
    slope_per_run: float
    slope_per_run_squared: float  # second derivative — positive = accelerating
    current_value: float
    baseline_value: Optional[float]
    pct_of_threshold: Optional[float]  # how close to the configured max_delta (0–1+)
    data_points: int
    window_hours: float


@dataclass
class TrendReport:
    agent_id: str
    generated_at: str
    accelerating: list[MetricTrend]
    rising: list[MetricTrend]
    recovering: list[MetricTrend]
    stable: list[MetricTrend]
    overall_trajectory: str

    @property
    def has_concerns(self) -> bool:
        return bool(self.accelerating or self.rising)

    def summary(self) -> str:
        lines = [
            f"Trend Report — {self.agent_id}",
            f"  Overall: {self.overall_trajectory}",
            f"  Accelerating metrics: {len(self.accelerating)}",
            f"  Rising metrics:       {len(self.rising)}",
            f"  Recovering metrics:   {len(self.recovering)}",
            f"  Stable metrics:       {len(self.stable)}",
        ]
        for t in self.accelerating:
            pct = f" ({t.pct_of_threshold:.0%} of threshold)" if t.pct_of_threshold else ""
            lines.append(
                f"  [!!] {t.category}/{t.metric}: slope={t.slope_per_run:+.4f}{pct}"
            )
        for t in self.rising:
            pct = f" ({t.pct_of_threshold:.0%} of threshold)" if t.pct_of_threshold else ""
            lines.append(
                f"  [!]  {t.category}/{t.metric}: slope={t.slope_per_run:+.4f}{pct}"
            )
        return "\n".join(lines)


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) via ordinary least squares."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0, sum_y / n
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


class TrendAnalyzer:
    """Fits regression lines to metric history to detect drift acceleration."""

    _SLOPE_STABLE_THRESHOLD = 0.001      # |slope| below this → stable
    _SLOPE_RISING_THRESHOLD = 0.005      # slope above this → rising
    _ACCEL_THRESHOLD = 0.0005            # second-deriv above this → accelerating

    def __init__(
        self,
        config: GovernanceConfig,
        db: GovernanceDB,
        *,
        history_limit: int = 30,
    ) -> None:
        self._config = config
        self._db = db
        self._history_limit = history_limit

    def analyze(self, category: Optional[str] = None) -> TrendReport:
        now = datetime.now(timezone.utc)
        threshold_map = {
            (t.category, t.metric): t.max_delta
            for t in self._config.drift_thresholds
        }

        all_trends: list[MetricTrend] = []
        metrics_to_check = self._collect_metrics(category)

        for cat, metric in metrics_to_check:
            history = self._db.get_metric_history(
                self._config.agent_id, cat, metric, limit=self._history_limit
            )
            if len(history) < 3:
                continue

            history_asc = list(reversed(history))
            ts0 = history_asc[0][0].timestamp()
            xs = [(ts.timestamp() - ts0) / 3600 for ts, _, _ in history_asc]
            ys = [val for _, val, _ in history_asc]

            slope, _ = _linreg(xs, ys)
            mid = len(xs) // 2
            slope1, _ = _linreg(xs[:mid], ys[:mid])
            slope2, _ = _linreg(xs[mid:], ys[mid:])
            accel = slope2 - slope1

            baseline_row = self._db.get_baseline(self._config.agent_id, cat, metric)
            baseline_val = baseline_row[0] if baseline_row else None

            max_delta = threshold_map.get((cat, metric)) or threshold_map.get(("*", metric))
            pct_of_threshold = None
            if max_delta and baseline_val is not None:
                current = ys[-1]
                delta = abs(current - baseline_val)
                pct_of_threshold = delta / max_delta if max_delta > 0 else None

            window_hours = (xs[-1] - xs[0]) if len(xs) > 1 else 0.0

            direction = self._classify(slope, accel)
            trend = MetricTrend(
                agent_id=self._config.agent_id,
                category=cat,
                metric=metric,
                direction=direction,
                slope_per_run=slope,
                slope_per_run_squared=accel,
                current_value=ys[-1],
                baseline_value=baseline_val,
                pct_of_threshold=pct_of_threshold,
                data_points=len(history_asc),
                window_hours=window_hours,
            )
            all_trends.append(trend)

        accelerating = [t for t in all_trends if t.direction == TrendDirection.ACCELERATING]
        rising = [t for t in all_trends if t.direction == TrendDirection.RISING]
        recovering = [t for t in all_trends if t.direction == TrendDirection.RECOVERING]
        stable = [t for t in all_trends if t.direction == TrendDirection.STABLE]

        if accelerating:
            overall = "accelerating"
        elif rising:
            overall = "rising"
        elif recovering:
            overall = "recovering"
        else:
            overall = "stable"

        return TrendReport(
            agent_id=self._config.agent_id,
            generated_at=now.isoformat(),
            accelerating=accelerating,
            rising=rising,
            recovering=recovering,
            stable=stable,
            overall_trajectory=overall,
        )

    def _collect_metrics(self, category: Optional[str]) -> list[tuple[str, str]]:
        from .drift import _EXTRACTORS
        cats = (
            [category] if category
            else list({t.category for t in self._config.drift_thresholds if t.category != "*"})
        )
        return [(cat, metric) for cat in cats for metric in _EXTRACTORS]

    def _classify(self, slope: float, accel: float) -> str:
        if slope > self._SLOPE_RISING_THRESHOLD and accel > self._ACCEL_THRESHOLD:
            return TrendDirection.ACCELERATING
        if slope > self._SLOPE_RISING_THRESHOLD:
            return TrendDirection.RISING
        if slope < -self._SLOPE_RISING_THRESHOLD and accel < -self._ACCEL_THRESHOLD:
            return TrendDirection.RECOVERING
        return TrendDirection.STABLE
