"""Drift detector: baseline computation + sliding-window comparison.

The baseline is computed from the first N events per category (the "warmup"
period where behavior was known good). Drift is detected by comparing a recent
sliding window of events against those frozen baselines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import GovernanceConfig, MetricDirection
from .storage import DecisionRecord, GovernanceDB


@dataclass
class CategoryMetrics:
    """Computed behavioral metrics for one case category over a window of events."""

    category: str
    total_events: int
    resolution_rate: float
    escalation_rate: float
    denial_rate: float
    avg_response_time_ms: float
    high_risk_count: int
    high_risk_escalation_rate: float
    accuracy: Optional[float]         # None when <5 ground-truth events available
    high_risk_accuracy: Optional[float]
    decision_entropy: float           # Shannon entropy across decision classes


@dataclass
class DriftResult:
    category: str
    metric: str
    rule_name: str
    baseline_value: float
    recent_value: float
    delta: float          # signed: positive = metric went up
    max_allowed: float
    severity: str
    is_violated: bool


@dataclass
class DriftReport:
    agent_id: str
    categories_analyzed: list[str]
    violations: list[DriftResult]
    all_results: list[DriftResult]
    overall_drift_score: float            # 0.0 = no drift, 1.0 = maximal drift
    category_drift_scores: dict[str, float]
    recent_metrics: dict[str, CategoryMetrics]


# ── metric extraction ─────────────────────────────────────────────────────────

def _compute_metrics(records: list[DecisionRecord], category: str = "") -> CategoryMetrics:
    if not records:
        return CategoryMetrics(
            category=category, total_events=0, resolution_rate=0.0,
            escalation_rate=0.0, denial_rate=0.0, avg_response_time_ms=0.0,
            high_risk_count=0, high_risk_escalation_rate=0.0,
            accuracy=None, high_risk_accuracy=None, decision_entropy=0.0,
        )

    n = len(records)
    counts: dict[str, int] = {}
    total_time = 0
    correct = gt_total = 0
    hr_correct = hr_gt_total = hr_escalated = hr_total = 0

    for r in records:
        counts[r.decision] = counts.get(r.decision, 0) + 1
        total_time += r.resolution_time_ms
        if r.ground_truth is not None:
            gt_total += 1
            if r.decision == r.ground_truth:
                correct += 1
        if r.is_high_risk:
            hr_total += 1
            if r.decision == "escalate":
                hr_escalated += 1
            if r.ground_truth is not None:
                hr_gt_total += 1
                if r.decision == r.ground_truth:
                    hr_correct += 1

    entropy = -sum(
        (c / n) * math.log2(c / n)
        for c in counts.values()
        if c > 0
    )

    return CategoryMetrics(
        category=category or (records[0].case_category if records else ""),
        total_events=n,
        resolution_rate=counts.get("resolve", 0) / n,
        escalation_rate=counts.get("escalate", 0) / n,
        denial_rate=counts.get("deny", 0) / n,
        avg_response_time_ms=total_time / n,
        high_risk_count=hr_total,
        high_risk_escalation_rate=hr_escalated / max(hr_total, 1),
        accuracy=correct / gt_total if gt_total >= 5 else None,
        high_risk_accuracy=hr_correct / hr_gt_total if hr_gt_total >= 5 else None,
        decision_entropy=entropy,
    )


# Maps metric name → getter from CategoryMetrics
_EXTRACTORS: dict[str, Callable[[CategoryMetrics], Optional[float]]] = {
    "resolution_rate":          lambda m: m.resolution_rate,
    "escalation_rate":          lambda m: m.escalation_rate,
    "denial_rate":              lambda m: m.denial_rate,
    "avg_response_time_ms":     lambda m: m.avg_response_time_ms,
    "high_risk_escalation_rate": lambda m: m.high_risk_escalation_rate,
    "accuracy":                 lambda m: m.accuracy,
    "high_risk_accuracy":       lambda m: m.high_risk_accuracy,
    "decision_entropy":         lambda m: m.decision_entropy,
}


# ── detector ──────────────────────────────────────────────────────────────────

class DriftDetector:
    """Computes drift scores by comparing recent behavior against frozen baselines."""

    def __init__(self, config: GovernanceConfig, db: GovernanceDB) -> None:
        self._config = config
        self._db = db

    def compute_baseline(
        self, categories: Optional[list[str]] = None
    ) -> dict[str, CategoryMetrics]:
        """
        Freeze the current baseline from the oldest events in the DB.

        Call once after the warmup period (before enabling drift monitoring).
        Subsequent calls overwrite the stored baseline.
        """
        # Fetch oldest events (oldest_first=True gives us the warmup window)
        all_records = self._db.get_recent_decisions(
            self._config.agent_id,
            limit=self._config.min_baseline_events * 20,
            oldest_first=True,
        )

        by_cat: dict[str, list[DecisionRecord]] = {}
        for rec in all_records:
            by_cat.setdefault(rec.case_category, []).append(rec)

        results: dict[str, CategoryMetrics] = {}
        all_baseline_recs: list[DecisionRecord] = []

        for cat, recs in by_cat.items():
            if categories and cat not in categories:
                continue
            baseline_recs = recs[: self._config.min_baseline_events]
            if len(baseline_recs) < self._config.min_baseline_events:
                continue

            metrics = _compute_metrics(baseline_recs, category=cat)
            results[cat] = metrics
            all_baseline_recs.extend(baseline_recs)

            for metric_name, extractor in _EXTRACTORS.items():
                val = extractor(metrics)
                if val is not None:
                    self._db.upsert_baseline(
                        self._config.agent_id, cat, metric_name, val, len(baseline_recs)
                    )

        # Store aggregate "*" baseline for wildcard thresholds
        if all_baseline_recs:
            agg = _compute_metrics(all_baseline_recs, category="*")
            for metric_name, extractor in _EXTRACTORS.items():
                val = extractor(agg)
                if val is not None:
                    self._db.upsert_baseline(
                        self._config.agent_id, "*", metric_name, val, len(all_baseline_recs)
                    )

        return results

    def detect(self) -> DriftReport:
        """
        Run drift detection against all stored thresholds.

        Returns a DriftReport with violations, drift scores per category,
        and the overall drift score.
        """
        recent_all = self._db.get_recent_decisions(
            self._config.agent_id,
            limit=self._config.recent_window_size * 6,  # fetch extra so each cat has enough
        )

        # Group by category, take newest N per category
        by_cat: dict[str, list[DecisionRecord]] = {}
        for rec in recent_all:
            by_cat.setdefault(rec.case_category, []).append(rec)

        recent_metrics: dict[str, CategoryMetrics] = {}
        for cat, recs in by_cat.items():
            window = recs[: self._config.recent_window_size]
            metrics = _compute_metrics(window, category=cat)
            recent_metrics[cat] = metrics
            # Record metric snapshots for trend tracking
            for mname, extractor in _EXTRACTORS.items():
                val = extractor(metrics)
                if val is not None:
                    self._db.insert_metric_snapshot(
                        self._config.agent_id, cat, mname, val, metrics.total_events
                    )

        violations: list[DriftResult] = []
        all_results: list[DriftResult] = []
        cat_max_drift: dict[str, float] = {}

        for thr in self._config.drift_thresholds:
            cats = list(recent_metrics.keys()) if thr.category == "*" else [thr.category]

            for cat in cats:
                if cat not in recent_metrics:
                    continue
                m = recent_metrics[cat]
                if m.total_events < thr.recent_window:
                    continue

                # Use the threshold's category for baseline lookup (supports "*" baseline)
                baseline = self._db.get_baseline(
                    self._config.agent_id, thr.category, thr.metric
                )
                if baseline is None or baseline[1] < thr.min_baseline_samples:
                    continue

                baseline_val, _ = baseline
                extractor = _EXTRACTORS.get(thr.metric)
                if extractor is None:
                    continue
                recent_val = extractor(m)
                if recent_val is None:
                    continue

                delta = recent_val - baseline_val
                abs_delta = abs(delta)
                violated = (
                    (thr.direction == MetricDirection.EITHER and abs_delta > thr.max_delta)
                    or (thr.direction == MetricDirection.INCREASE and delta > thr.max_delta)
                    or (thr.direction == MetricDirection.DECREASE and delta < -thr.max_delta)
                )

                # Drift fraction: how far past the threshold are we? Capped at 1.0.
                drift_frac = min(abs_delta / thr.max_delta, 2.0) / 2.0
                cat_max_drift[cat] = max(cat_max_drift.get(cat, 0.0), drift_frac)

                result = DriftResult(
                    category=cat,
                    metric=thr.metric,
                    rule_name=thr.name,
                    baseline_value=baseline_val,
                    recent_value=recent_val,
                    delta=delta,
                    max_allowed=thr.max_delta,
                    severity=thr.severity.value,
                    is_violated=violated,
                )
                all_results.append(result)
                if violated:
                    violations.append(result)

        overall = max(cat_max_drift.values(), default=0.0)
        return DriftReport(
            agent_id=self._config.agent_id,
            categories_analyzed=list(recent_metrics.keys()),
            violations=violations,
            all_results=all_results,
            overall_drift_score=overall,
            category_drift_scores=cat_max_drift,
            recent_metrics=recent_metrics,
        )

    def get_category_metrics(self, category: str) -> Optional[CategoryMetrics]:
        recs = self._db.get_recent_decisions(
            self._config.agent_id,
            category=category,
            limit=self._config.recent_window_size,
        )
        return _compute_metrics(recs, category=category) if recs else None
