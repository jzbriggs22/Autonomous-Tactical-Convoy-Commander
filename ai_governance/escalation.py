"""Alert escalation engine.

Escalates unacknowledged alerts after a configurable timeout. Each
severity level has a max wait time; if an alert goes unacknowledged
past that window the escalation engine fires a re-alert at the next
severity level and dispatches webhooks.

Escalation ladder: info → warn → critical → rollback
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from .storage import AlertRecord, GovernanceDB

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = ["info", "warn", "critical", "rollback"]
_SEVERITY_NEXT = {
    "info": "warn",
    "warn": "critical",
    "critical": "rollback",
    "rollback": None,  # already at max
}


@dataclass
class EscalationRule:
    from_severity: str
    escalate_after_seconds: float


_DEFAULT_RULES = [
    EscalationRule("info", 3600.0 * 24),   # info → warn after 24h
    EscalationRule("warn", 3600.0 * 4),    # warn → critical after 4h
    EscalationRule("critical", 3600.0),    # critical → rollback after 1h
]


@dataclass
class EscalationEvent:
    original_alert_id: str
    new_alert_id: str
    from_severity: str
    to_severity: str
    rule_name: str
    message: str
    escalated_at: str


class AlertEscalator:
    """Checks for overdue unacknowledged alerts and escalates them."""

    def __init__(
        self,
        db: GovernanceDB,
        agent_id: str,
        *,
        rules: Optional[list[EscalationRule]] = None,
        webhooks=None,
    ) -> None:
        self._db = db
        self._agent_id = agent_id
        self._rules = rules or _DEFAULT_RULES
        self._webhooks = webhooks
        self._rule_map = {r.from_severity: r for r in self._rules}

    def run(self) -> list[EscalationEvent]:
        """Check all unacknowledged alerts and escalate overdue ones."""
        now = datetime.now(timezone.utc)
        events: list[EscalationEvent] = []

        for rule in self._rules:
            cutoff = now - timedelta(seconds=rule.escalate_after_seconds)
            overdue = self._db.get_unacknowledged_alerts_older_than(self._agent_id, cutoff)
            for alert in overdue:
                if alert.severity != rule.from_severity:
                    continue
                next_sev = _SEVERITY_NEXT.get(alert.severity)
                if next_sev is None:
                    continue
                event = self._escalate(alert, next_sev, now)
                events.append(event)
                logger.warning(
                    "Escalated alert %s from %s to %s (unacknowledged for %.0fs)",
                    alert.alert_id, alert.severity, next_sev,
                    (now - alert.timestamp).total_seconds(),
                )

        return events

    def _escalate(
        self, alert: AlertRecord, to_severity: str, now: datetime
    ) -> EscalationEvent:
        from .storage import AlertRecord as AR
        new_id = str(uuid.uuid4())
        elapsed_h = (now - alert.timestamp).total_seconds() / 3600
        message = (
            f"[ESCALATED to {to_severity.upper()}] Alert '{alert.rule_name}' "
            f"unacknowledged for {elapsed_h:.1f}h. "
            f"Original: {alert.message[:200]}"
        )
        self._db.insert_alert(AR(
            alert_id=new_id,
            agent_id=self._agent_id,
            timestamp=now,
            rule_name=f"{alert.rule_name}:escalated",
            severity=to_severity,
            message=message,
            metrics={
                "original_alert_id": alert.alert_id,
                "original_severity": alert.severity,
                "elapsed_hours": round(elapsed_h, 2),
            },
        ))
        if self._webhooks:
            self._webhooks.dispatch_async({
                "source": "escalator",
                "original_alert_id": alert.alert_id,
                "new_alert_id": new_id,
                "from_severity": alert.severity,
                "to_severity": to_severity,
                "rule_name": alert.rule_name,
                "message": message,
                "agent_id": self._agent_id,
            })
        return EscalationEvent(
            original_alert_id=alert.alert_id,
            new_alert_id=new_id,
            from_severity=alert.severity,
            to_severity=to_severity,
            rule_name=alert.rule_name,
            message=message,
            escalated_at=now.isoformat(),
        )
