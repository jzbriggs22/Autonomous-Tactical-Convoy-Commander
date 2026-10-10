"""Deployment readiness checker.

Validates governance preconditions before an agent goes live:
- Sufficient baseline data per category
- No active rollbacks
- Audit chain intact
- All required categories covered
- Config version pinned and fingerprinted

Returns a structured ReadinessReport so the CI/CD pipeline can gate
deployments automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .alerts import AlertEngine
from .audit import AuditLog
from .config import GovernanceConfig
from .storage import GovernanceDB


@dataclass
class ReadinessCheck:
    name: str
    passed: bool
    severity: str  # "blocker" | "warning" | "info"
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class ReadinessReport:
    agent_id: str
    config_version: str
    config_fingerprint: str
    checked_at: str
    ready: bool
    blockers: list[ReadinessCheck]
    warnings: list[ReadinessCheck]
    passed: list[ReadinessCheck]

    @property
    def total_checks(self) -> int:
        return len(self.blockers) + len(self.warnings) + len(self.passed)

    def summary(self) -> str:
        status = "READY" if self.ready else "NOT READY"
        lines = [
            f"Deployment Readiness: {status}",
            f"  Agent:  {self.agent_id}",
            f"  Config: v{self.config_version} ({self.config_fingerprint})",
            f"  Checks: {self.total_checks} total, "
            f"{len(self.blockers)} blockers, {len(self.warnings)} warnings",
        ]
        if self.blockers:
            lines.append("")
            lines.append("  BLOCKERS:")
            for c in self.blockers:
                lines.append(f"    [X] {c.name}: {c.message}")
        if self.warnings:
            lines.append("")
            lines.append("  WARNINGS:")
            for c in self.warnings:
                lines.append(f"    [!] {c.name}: {c.message}")
        if self.passed:
            lines.append("")
            lines.append("  PASSED:")
            for c in self.passed:
                lines.append(f"    [+] {c.name}")
        return "\n".join(lines)


class ReadinessChecker:
    """Runs all pre-deployment governance checks."""

    def __init__(
        self,
        config: GovernanceConfig,
        db: GovernanceDB,
        *,
        required_categories: Optional[list[str]] = None,
    ) -> None:
        self._config = config
        self._db = db
        self._engine = AlertEngine(config, db)
        self._audit = AuditLog(db)
        self._required_categories = required_categories or []

    def check(self) -> ReadinessReport:
        now = datetime.now(timezone.utc)
        checks: list[ReadinessCheck] = []

        checks.extend(self._check_baseline_data())
        checks.append(self._check_no_active_rollbacks())
        checks.append(self._check_audit_chain())
        checks.append(self._check_config_fingerprint())
        checks.extend(self._check_required_categories())
        checks.append(self._check_thresholds_configured())
        checks.append(self._check_rollback_conditions())

        blockers = [c for c in checks if not c.passed and c.severity == "blocker"]
        warnings = [c for c in checks if not c.passed and c.severity == "warning"]
        passed = [c for c in checks if c.passed]

        return ReadinessReport(
            agent_id=self._config.agent_id,
            config_version=self._config.version,
            config_fingerprint=self._config.fingerprint,
            checked_at=now.isoformat(),
            ready=len(blockers) == 0,
            blockers=blockers,
            warnings=warnings,
            passed=passed,
        )

    # ── individual checks ──────────────────────────────────────────────────────

    def _check_baseline_data(self) -> list[ReadinessCheck]:
        results = []
        cats = {t.category for t in self._config.drift_thresholds if t.category != "*"}
        for cat in sorted(cats):
            count = self._db.count_decisions(self._config.agent_id, category=cat)
            needed = self._config.min_baseline_events
            has_baseline = self._db.get_baseline(self._config.agent_id, cat, "resolution_rate") is not None
            if has_baseline and count >= needed:
                results.append(ReadinessCheck(
                    name=f"baseline_data_{cat}",
                    passed=True,
                    severity="blocker",
                    message=f"'{cat}' has {count} events and a computed baseline",
                    detail={"count": count, "needed": needed},
                ))
            elif count >= needed and not has_baseline:
                results.append(ReadinessCheck(
                    name=f"baseline_data_{cat}",
                    passed=False,
                    severity="warning",
                    message=f"'{cat}' has {count} events but baseline not yet computed — call POST /baseline/compute",
                    detail={"count": count, "needed": needed},
                ))
            else:
                results.append(ReadinessCheck(
                    name=f"baseline_data_{cat}",
                    passed=False,
                    severity="blocker",
                    message=f"'{cat}' only has {count}/{needed} events for baseline",
                    detail={"count": count, "needed": needed},
                ))
        if not cats:
            results.append(ReadinessCheck(
                name="baseline_data",
                passed=True,
                severity="info",
                message="No category-specific thresholds configured",
            ))
        return results

    def _check_no_active_rollbacks(self) -> ReadinessCheck:
        is_safe, reason = self._engine.is_agent_safe()
        if is_safe:
            return ReadinessCheck(
                name="no_active_rollbacks",
                passed=True,
                severity="blocker",
                message="No active rollbacks",
            )
        return ReadinessCheck(
            name="no_active_rollbacks",
            passed=False,
            severity="blocker",
            message=f"Active rollback prevents deployment: {reason}",
        )

    def _check_audit_chain(self) -> ReadinessCheck:
        valid, broken_seq = self._audit.verify_chain(self._config.agent_id)
        if valid:
            return ReadinessCheck(
                name="audit_chain_integrity",
                passed=True,
                severity="blocker",
                message="Audit chain is intact",
            )
        return ReadinessCheck(
            name="audit_chain_integrity",
            passed=False,
            severity="blocker",
            message=f"Audit chain broken at seq={broken_seq} — data integrity compromised",
            detail={"broken_at_seq": broken_seq},
        )

    def _check_config_fingerprint(self) -> ReadinessCheck:
        fp = self._config.fingerprint
        if fp and len(fp) >= 6:
            return ReadinessCheck(
                name="config_fingerprinted",
                passed=True,
                severity="warning",
                message=f"Config fingerprinted: {fp}",
                detail={"fingerprint": fp, "version": self._config.version},
            )
        return ReadinessCheck(
            name="config_fingerprinted",
            passed=False,
            severity="warning",
            message="Config fingerprint missing or too short — ensure governance config is pinned",
        )

    def _check_required_categories(self) -> list[ReadinessCheck]:
        if not self._required_categories:
            return []
        results = []
        for cat in self._required_categories:
            count = self._db.count_decisions(self._config.agent_id, category=cat)
            if count > 0:
                results.append(ReadinessCheck(
                    name=f"required_category_{cat}",
                    passed=True,
                    severity="blocker",
                    message=f"Required category '{cat}' has {count} events",
                ))
            else:
                results.append(ReadinessCheck(
                    name=f"required_category_{cat}",
                    passed=False,
                    severity="blocker",
                    message=f"Required category '{cat}' has no events in database",
                ))
        return results

    def _check_thresholds_configured(self) -> ReadinessCheck:
        count = len(self._config.drift_thresholds)
        if count >= 2:
            return ReadinessCheck(
                name="drift_thresholds_configured",
                passed=True,
                severity="blocker",
                message=f"{count} drift threshold(s) configured",
                detail={"count": count},
            )
        return ReadinessCheck(
            name="drift_thresholds_configured",
            passed=False,
            severity="blocker",
            message=f"Only {count} drift threshold(s) configured — minimum 2 required for meaningful detection",
            detail={"count": count},
        )

    def _check_rollback_conditions(self) -> ReadinessCheck:
        count = len(self._config.rollback_conditions)
        if count >= 1:
            return ReadinessCheck(
                name="rollback_conditions_configured",
                passed=True,
                severity="warning",
                message=f"{count} rollback condition(s) configured",
                detail={"count": count},
            )
        return ReadinessCheck(
            name="rollback_conditions_configured",
            passed=False,
            severity="warning",
            message="No rollback conditions configured — agent will alert but never auto-rollback",
            detail={"count": 0},
        )
