"""Data retention policy engine.

Configurable TTL-based cleanup for governance data. Each data type
(decisions, alerts, audit entries, metric snapshots) can have its own
retention period. Retention runs are logged in the audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .storage import GovernanceDB

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    decisions_days: int = 90
    alerts_days: int = 180
    metric_snapshots_days: int = 90
    acknowledged_alerts_only: bool = True


@dataclass
class RetentionResult:
    decisions_deleted: int = 0
    alerts_deleted: int = 0
    snapshots_deleted: int = 0
    executed_at: Optional[str] = None

    @property
    def total_deleted(self) -> int:
        return self.decisions_deleted + self.alerts_deleted + self.snapshots_deleted


class RetentionManager:
    """Applies retention policies to governance data stores."""

    def __init__(self, db: GovernanceDB, policy: Optional[RetentionPolicy] = None) -> None:
        self._db = db
        self._policy = policy or RetentionPolicy()

    @property
    def policy(self) -> RetentionPolicy:
        return self._policy

    @policy.setter
    def policy(self, value: RetentionPolicy) -> None:
        self._policy = value

    def apply(self, agent_id: str) -> RetentionResult:
        now = datetime.now(timezone.utc)
        result = RetentionResult(executed_at=now.isoformat())

        result.decisions_deleted = self._db.purge_old_decisions(
            agent_id, keep_days=self._policy.decisions_days,
        )
        result.alerts_deleted = self._db.purge_old_alerts(
            agent_id, keep_days=self._policy.alerts_days,
        )
        result.snapshots_deleted = self._db.purge_old_snapshots(
            agent_id, keep_days=self._policy.metric_snapshots_days,
        )

        if result.total_deleted > 0:
            logger.info(
                "Retention applied for %s: %d decisions, %d alerts, %d snapshots deleted",
                agent_id, result.decisions_deleted, result.alerts_deleted,
                result.snapshots_deleted,
            )

        return result

    def dry_run(self, agent_id: str) -> dict:
        """Estimate what would be deleted without actually deleting."""
        counts = self._db.get_table_counts(agent_id)
        return {
            "current_counts": counts,
            "policy": {
                "decisions_days": self._policy.decisions_days,
                "alerts_days": self._policy.alerts_days,
                "metric_snapshots_days": self._policy.metric_snapshots_days,
            },
        }
