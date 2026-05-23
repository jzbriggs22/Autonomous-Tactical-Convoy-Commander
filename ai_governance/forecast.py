"""Predictive drift alerts: forecast threshold breaches before they happen.

Uses the linear regression slopes from TrendAnalyzer to extrapolate when
each metric will cross its configured max_delta. Fires predictive alerts
for metrics projected to breach within a configurable horizon (default 24h).

This gives PMs actionable lead time: "billing_dispute resolution_rate will
breach the 0.15 threshold in ~6 hours at current trajectory."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import GovernanceConfig
from .storage import GovernanceDB
from .trend import TrendAnalyzer, TrendReport, MetricTrend, _linreg


@dataclass
class BreachForecast:
    category: str
    metric: str
    current_value: float
    baseline_value: float
    threshold_delta: float
    current_delta: float
    remaining_delta: float
    slope_per_hour: float
    hours_to_breach: Optional[float]
    severity: str  # "imminent" (<2h), "near" (<8h), "approaching" (<24h)
    confidence: str  # "high" (many data points, steady slope), "moderate", "low"
    data_points: int
    message: str


@dataclass
class ForecastReport:
    agent_id: str
    generated_at: str
    horizon_hours: float
    forecasts: list[BreachForecast]

    @property
    def has_imminent(self) -> bool:
        return any(f.severity == "imminent" for f in self.forecasts)

    @property
    def has_approaching(self) -> bool:
        return len(self.forecasts) > 0

    def summary(self) -> str:
        lines = [
            f"Predictive Drift Forecast — {self.agent_id}",
            f"  Horizon: {self.horizon_hours:.0f}h",
            f"  Forecasts: {len(self.forecasts)}",
        ]
        for f in self.forecasts:
            h = f"in {f.hours_to_breach:.1f}h" if f.hours_to_breach is not None else "unknown ETA"
            lines.append(
                f"  [{f.severity.upper()}] {f.category}/{f.metric}: "
                f"breach {h} (current delta={f.current_delta:.4f}, "
                f"max={f.threshold_delta:.4f}, slope={f.slope_per_hour:+.6f}/h)"
            )
        return "\n".join(lines)


class DriftForecaster:
    """Forecasts threshold breaches from metric trend data."""

    def __init__(
        self,
        config: GovernanceConfig,
        db: GovernanceDB,
        *,
        horizon_hours: float = 24.0,
        history_limit: int = 30,
    ) -> None:
        self._config = config
        self._db = db
        self._horizon = horizon_hours
        self._analyzer = TrendAnalyzer(config, db, history_limit=history_limit)

    def forecast(self, category: Optional[str] = None) -> ForecastReport:
        now = datetime.now(timezone.utc)
        trend_report = self._analyzer.analyze(category=category)

        threshold_map = {}
        direction_map = {}
        for t in self._config.drift_thresholds:
            threshold_map[(t.category, t.metric)] = t.max_delta
            direction_map[(t.category, t.metric)] = t.direction.value

        all_trends = (
            trend_report.accelerating + trend_report.rising +
            trend_report.recovering + trend_report.stable
        )

        forecasts: list[BreachForecast] = []
        for trend in all_trends:
            if trend.baseline_value is None:
                continue

            max_delta = (
                threshold_map.get((trend.category, trend.metric))
                or threshold_map.get(("*", trend.metric))
            )
            if max_delta is None:
                continue

            direction = (
                direction_map.get((trend.category, trend.metric))
                or direction_map.get(("*", trend.metric), "either")
            )

            current_delta = trend.current_value - trend.baseline_value
            abs_delta = abs(current_delta)

            if abs_delta >= max_delta:
                continue

            remaining = max_delta - abs_delta
            slope = trend.slope_per_run

            will_breach = self._will_breach(slope, current_delta, max_delta, direction)
            if not will_breach:
                continue

            effective_slope = abs(slope) if slope != 0 else 0
            if effective_slope < 1e-9:
                continue

            hours_to_breach = remaining / effective_slope if effective_slope > 0 else None

            if hours_to_breach is not None and hours_to_breach > self._horizon:
                continue

            severity = self._classify_urgency(hours_to_breach)
            confidence = self._assess_confidence(trend.data_points, trend.slope_per_run_squared)

            message = self._build_message(
                trend.category, trend.metric, hours_to_breach,
                current_delta, max_delta, slope,
            )

            forecasts.append(BreachForecast(
                category=trend.category,
                metric=trend.metric,
                current_value=trend.current_value,
                baseline_value=trend.baseline_value,
                threshold_delta=max_delta,
                current_delta=round(current_delta, 6),
                remaining_delta=round(remaining, 6),
                slope_per_hour=round(slope, 8),
                hours_to_breach=round(hours_to_breach, 2) if hours_to_breach is not None else None,
                severity=severity,
                confidence=confidence,
                data_points=trend.data_points,
                message=message,
            ))

        forecasts.sort(key=lambda f: f.hours_to_breach if f.hours_to_breach is not None else float("inf"))

        return ForecastReport(
            agent_id=self._config.agent_id,
            generated_at=now.isoformat(),
            horizon_hours=self._horizon,
            forecasts=forecasts,
        )

    def _will_breach(self, slope: float, current_delta: float, max_delta: float, direction: str) -> bool:
        if direction == "increase":
            return slope > 0 and current_delta > 0 and current_delta < max_delta
        if direction == "decrease":
            return slope < 0 and current_delta < 0 and abs(current_delta) < max_delta
        return abs(slope) > 1e-9 and abs(current_delta) < max_delta

    def _classify_urgency(self, hours: Optional[float]) -> str:
        if hours is None:
            return "approaching"
        if hours < 2:
            return "imminent"
        if hours < 8:
            return "near"
        return "approaching"

    def _assess_confidence(self, data_points: int, acceleration: float) -> str:
        if data_points >= 20 and abs(acceleration) < 0.001:
            return "high"
        if data_points >= 10:
            return "moderate"
        return "low"

    def _build_message(
        self,
        category: str,
        metric: str,
        hours: Optional[float],
        current_delta: float,
        max_delta: float,
        slope: float,
    ) -> str:
        eta = f"in {hours:.1f} hours" if hours is not None else "at an indeterminate time"
        pct = abs(current_delta) / max_delta * 100 if max_delta > 0 else 0
        direction = "rising" if slope > 0 else "falling"
        return (
            f"'{metric}' in category '{category}' is {direction} at "
            f"{abs(slope):.6f}/hour. Currently {pct:.0f}% of threshold "
            f"({abs(current_delta):.4f}/{max_delta:.4f}). "
            f"Projected to breach {eta}."
        )
