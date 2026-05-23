"""Ground-truth feedback pipeline.

Bulk import of ground-truth labels for governance decisions. Computes
accuracy metrics across categories, tracks labeling coverage, and
identifies categories with degrading accuracy.

The feedback loop: agent decisions → human review → ground-truth labels →
accuracy metrics → drift detection → rollback if accuracy drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import GovernanceConfig
from .ingestion import VALID_DECISIONS, ValidationError
from .storage import GovernanceDB


@dataclass
class FeedbackLabel:
    event_id: str
    ground_truth: str


@dataclass
class FeedbackResult:
    total_submitted: int
    applied: int
    not_found: int
    invalid: int
    errors: list[str]


@dataclass
class CategoryAccuracy:
    category: str
    total_labeled: int
    correct: int
    incorrect: int
    accuracy: float
    high_risk_labeled: int
    high_risk_correct: int
    high_risk_accuracy: Optional[float]


@dataclass
class AccuracyReport:
    agent_id: str
    generated_at: str
    total_decisions: int
    total_labeled: int
    labeling_coverage: float
    overall_accuracy: Optional[float]
    categories: list[CategoryAccuracy]
    unlabeled_categories: list[str]

    def summary(self) -> str:
        lines = [
            f"Accuracy Report — {self.agent_id}",
            f"  Total decisions: {self.total_decisions}",
            f"  Labeled: {self.total_labeled} ({self.labeling_coverage:.1%} coverage)",
        ]
        if self.overall_accuracy is not None:
            lines.append(f"  Overall accuracy: {self.overall_accuracy:.1%}")
        else:
            lines.append("  Overall accuracy: N/A (need 5+ labels)")
        if self.categories:
            lines.append("")
            lines.append("  Per-category:")
            for cat in self.categories:
                hr = ""
                if cat.high_risk_accuracy is not None:
                    hr = f" (high-risk: {cat.high_risk_accuracy:.1%})"
                lines.append(
                    f"    {cat.category}: {cat.accuracy:.1%} "
                    f"({cat.total_labeled} labeled, {cat.correct}/{cat.total_labeled} correct){hr}"
                )
        if self.unlabeled_categories:
            lines.append(f"  Unlabeled categories: {', '.join(self.unlabeled_categories)}")
        return "\n".join(lines)


class FeedbackPipeline:
    """Manages ground-truth label ingestion and accuracy computation."""

    def __init__(self, config: GovernanceConfig, db: GovernanceDB) -> None:
        self._config = config
        self._db = db

    def apply_labels(self, labels: list[FeedbackLabel]) -> FeedbackResult:
        applied = 0
        not_found = 0
        invalid = 0
        errors: list[str] = []

        for label in labels:
            if label.ground_truth not in VALID_DECISIONS:
                invalid += 1
                errors.append(
                    f"Invalid ground_truth '{label.ground_truth}' for event {label.event_id}"
                )
                continue

            found = self._db.set_ground_truth(
                label.event_id, self._config.agent_id, label.ground_truth
            )
            if found:
                applied += 1
            else:
                not_found += 1
                errors.append(f"Event {label.event_id} not found")

        return FeedbackResult(
            total_submitted=len(labels),
            applied=applied,
            not_found=not_found,
            invalid=invalid,
            errors=errors[:50],
        )

    def compute_accuracy(self, limit: int = 10000) -> AccuracyReport:
        now = datetime.now(timezone.utc)
        total_decisions = self._db.count_decisions(self._config.agent_id)

        all_recs = self._db.get_recent_decisions(
            self._config.agent_id, limit=limit, oldest_first=True
        )

        labeled = [r for r in all_recs if r.ground_truth is not None]
        total_labeled = len(labeled)
        coverage = total_labeled / max(total_decisions, 1)

        overall_correct = sum(1 for r in labeled if r.decision == r.ground_truth)
        overall_accuracy = (
            overall_correct / total_labeled if total_labeled >= 5 else None
        )

        by_cat: dict[str, list] = {}
        all_cats: set[str] = set()
        for r in all_recs:
            all_cats.add(r.case_category)
        for r in labeled:
            by_cat.setdefault(r.case_category, []).append(r)

        categories: list[CategoryAccuracy] = []
        for cat in sorted(by_cat.keys()):
            recs = by_cat[cat]
            n = len(recs)
            correct = sum(1 for r in recs if r.decision == r.ground_truth)
            incorrect = n - correct

            hr = [r for r in recs if r.is_high_risk]
            hr_correct = sum(1 for r in hr if r.decision == r.ground_truth)
            hr_acc = hr_correct / len(hr) if len(hr) >= 3 else None

            categories.append(CategoryAccuracy(
                category=cat,
                total_labeled=n,
                correct=correct,
                incorrect=incorrect,
                accuracy=correct / n if n > 0 else 0.0,
                high_risk_labeled=len(hr),
                high_risk_correct=hr_correct,
                high_risk_accuracy=hr_acc,
            ))

        labeled_cats = set(by_cat.keys())
        unlabeled = sorted(all_cats - labeled_cats)

        return AccuracyReport(
            agent_id=self._config.agent_id,
            generated_at=now.isoformat(),
            total_decisions=total_decisions,
            total_labeled=total_labeled,
            labeling_coverage=coverage,
            overall_accuracy=overall_accuracy,
            categories=categories,
            unlabeled_categories=unlabeled,
        )
