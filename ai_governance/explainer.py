"""Decision explainer: why was this case classified as high-risk?

Reconstructs the risk classification decision from the stored event record
and the active governance config. Returns a structured explanation with:
  - Which patterns fired (and their weights)
  - Composite risk score breakdown
  - Governance threshold context for this category
  - Human-readable audit narrative

Useful for audits, PM review, and customer-facing transparency reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .config import GovernanceConfig
from .storage import DecisionRecord, GovernanceDB


@dataclass
class PatternMatch:
    pattern_name: str
    field: str
    pattern: str
    matched_value: str
    weight: float
    contribution: float  # weight / max_possible_weight


@dataclass
class DecisionExplanation:
    event_id: str
    agent_id: str
    timestamp: str
    case_id: str
    case_category: str
    decision: str
    is_high_risk: bool
    high_risk_score: float

    matched_patterns: list[PatternMatch]
    unmatched_patterns: list[str]
    max_possible_score: float
    config_version: str

    category_thresholds: list[dict]
    baseline_metrics: dict[str, Optional[float]]
    audit_narrative: str

    @property
    def dominant_pattern(self) -> Optional[str]:
        if not self.matched_patterns:
            return None
        return max(self.matched_patterns, key=lambda p: p.weight).pattern_name


class DecisionExplainer:
    """Explains a stored governance decision."""

    def __init__(self, config: GovernanceConfig, db: GovernanceDB) -> None:
        self._config = config
        self._db = db
        self._compiled = [
            (p, re.compile(p.pattern, re.IGNORECASE))
            for p in config.high_risk_patterns
        ]

    def explain(self, event_id: str) -> Optional[DecisionExplanation]:
        rec = self._db.get_decision_by_id(event_id, self._config.agent_id)
        if rec is None:
            return None
        return self._build_explanation(rec)

    def _build_explanation(self, rec: DecisionRecord) -> DecisionExplanation:
        checkable = {
            "case_category": rec.case_category,
            "decision": rec.decision,
            **{k: str(v) for k, v in (rec.metadata or {}).items()},
        }
        max_weight = sum(p.weight for p in self._config.high_risk_patterns) or 1.0

        matched: list[PatternMatch] = []
        unmatched: list[str] = []

        for pattern, compiled in self._compiled:
            target = checkable.get(pattern.field, "")
            if compiled.search(target):
                matched.append(PatternMatch(
                    pattern_name=pattern.name,
                    field=pattern.field,
                    pattern=pattern.pattern,
                    matched_value=target,
                    weight=pattern.weight,
                    contribution=pattern.weight / max_weight,
                ))
            else:
                unmatched.append(pattern.name)

        cat_thresholds = [
            {
                "name": t.name,
                "metric": t.metric,
                "max_delta": t.max_delta,
                "direction": t.direction.value,
                "severity": t.severity.value,
            }
            for t in self._config.drift_thresholds
            if t.category in (rec.case_category, "*")
        ]

        from .drift import _EXTRACTORS
        baseline_metrics = {}
        for metric in _EXTRACTORS:
            row = self._db.get_baseline(self._config.agent_id, rec.case_category, metric)
            baseline_metrics[metric] = row[0] if row else None

        narrative = self._build_narrative(rec, matched, unmatched, max_weight)

        return DecisionExplanation(
            event_id=rec.event_id,
            agent_id=rec.agent_id,
            timestamp=rec.timestamp.isoformat(),
            case_id=rec.case_id,
            case_category=rec.case_category,
            decision=rec.decision,
            is_high_risk=rec.is_high_risk,
            high_risk_score=round(rec.high_risk_score, 4),
            matched_patterns=matched,
            unmatched_patterns=unmatched,
            max_possible_score=1.0,
            config_version=self._config.version,
            category_thresholds=cat_thresholds,
            baseline_metrics=baseline_metrics,
            audit_narrative=narrative,
        )

    def _build_narrative(
        self,
        rec: DecisionRecord,
        matched: list[PatternMatch],
        unmatched: list[str],
        max_weight: float,
    ) -> str:
        if not matched:
            return (
                f"Event {rec.event_id} (category='{rec.case_category}', "
                f"decision='{rec.decision}') was NOT classified as high-risk. "
                f"No risk patterns matched."
            )

        pattern_desc = ", ".join(f"'{m.pattern_name}'" for m in matched)
        score_pct = f"{rec.high_risk_score:.0%}"
        lines = [
            f"Event {rec.event_id} classified as {'high-risk' if rec.is_high_risk else 'low-risk'} "
            f"(score={score_pct}).",
            f"Category: '{rec.case_category}', Decision: '{rec.decision}'.",
            f"Matched patterns: {pattern_desc}.",
        ]
        if len(matched) == 1:
            m = matched[0]
            lines.append(
                f"Pattern '{m.pattern_name}' matched field '{m.field}' "
                f"(value: '{m.matched_value}') with weight {m.weight:.1f}/{max_weight:.1f}."
            )
        else:
            lines.append(f"Total weight: {sum(m.weight for m in matched):.1f}/{max_weight:.1f}.")
        if unmatched:
            lines.append(f"Patterns that did NOT match: {', '.join(repr(u) for u in unmatched)}.")
        return " ".join(lines)
