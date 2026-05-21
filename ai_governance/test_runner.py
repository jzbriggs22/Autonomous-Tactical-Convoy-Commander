"""Governance test runner: connects behavioral test outcomes to rollback.

When behavioral/drift tests fail, this runner programmatically triggers
a rollback via the AlertEngine — satisfying the requirement that
"If behavioral tests fail, the rollback engine triggers."

Usage:
    # As a script (CI/pre-commit hook):
    python -m ai_governance.test_runner

    # Programmatically:
    from ai_governance.test_runner import GovernanceTestRunner
    runner = GovernanceTestRunner(config, db)
    result = runner.run()
    if not result.passed:
        # rollback already triggered
        print(result.summary)
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .alerts import AlertEngine
from .audit import AuditLog
from .config import GovernanceConfig
from .storage import AlertRecord, GovernanceDB, RollbackRecord


@dataclass
class TestRunResult:
    passed: bool
    exit_code: int
    tests_run: int
    tests_failed: int
    tests_passed: int
    rollback_triggered: bool
    rollback_id: Optional[str]
    summary: str
    raw_output: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GovernanceTestRunner:
    """Runs the behavioral test suite and triggers rollback on failure.

    The runner executes ``pytest tests/governance/test_behavioral.py``
    as a subprocess, parses the exit code and output, and — when tests
    fail — inserts a rollback record and alert via the same path the
    alert engine uses.  This ensures the PM dashboard immediately
    reflects the failure, the audit log records it, and the agent is
    halted until a human resolves the rollback.
    """

    def __init__(
        self,
        config: GovernanceConfig,
        db: GovernanceDB,
        *,
        test_path: str = "tests/governance/test_behavioral.py",
    ) -> None:
        self._config = config
        self._db = db
        self._test_path = test_path

    def run(self, extra_args: list[str] = None) -> TestRunResult:
        """Run behavioral tests, trigger rollback on failure."""
        cmd = [
            sys.executable, "-m", "pytest",
            self._test_path,
            "-v", "--tb=short", "--timeout=60",
        ]
        if extra_args:
            cmd.extend(extra_args)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return self._handle_failure(
                exit_code=-1,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                raw_output="Test suite timed out (120s)",
                reason="Behavioral test suite timed out",
            )

        passed_count, failed_count, total = self._parse_pytest_output(proc.stdout)

        if proc.returncode == 0:
            self._record_audit_pass(total, passed_count)
            return TestRunResult(
                passed=True,
                exit_code=0,
                tests_run=total,
                tests_passed=passed_count,
                tests_failed=0,
                rollback_triggered=False,
                rollback_id=None,
                summary=f"All {total} behavioral tests passed",
                raw_output=proc.stdout,
            )

        return self._handle_failure(
            exit_code=proc.returncode,
            tests_run=total,
            tests_passed=passed_count,
            tests_failed=failed_count,
            raw_output=proc.stdout + proc.stderr,
            reason=f"{failed_count} of {total} behavioral tests failed",
        )

    def _handle_failure(
        self,
        exit_code: int,
        tests_run: int,
        tests_passed: int,
        tests_failed: int,
        raw_output: str,
        reason: str,
    ) -> TestRunResult:
        rollback_id = str(uuid.uuid4())
        alert_id = str(uuid.uuid4())

        self._db.insert_alert(AlertRecord(
            alert_id=alert_id,
            agent_id=self._config.agent_id,
            timestamp=datetime.now(timezone.utc),
            rule_name="behavioral_test_failure",
            severity="rollback",
            message=f"[ROLLBACK] Behavioral regression suite failed: {reason}",
            metrics={
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "exit_code": exit_code,
            },
        ))

        self._db.insert_rollback(RollbackRecord(
            rollback_id=rollback_id,
            agent_id=self._config.agent_id,
            timestamp=datetime.now(timezone.utc),
            trigger_rule="behavioral_test_failure",
            reason=reason,
            metrics={
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
            },
        ))

        try:
            audit = AuditLog(self._db)
            audit.append(
                self._config.agent_id,
                "rollback.triggered",
                "governance_test_runner",
                "behavioral_test",
                resource_id=rollback_id,
                detail={
                    "trigger": "behavioral_test_failure",
                    "tests_failed": tests_failed,
                    "tests_run": tests_run,
                },
            )
        except Exception:
            pass

        return TestRunResult(
            passed=False,
            exit_code=exit_code,
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            rollback_triggered=True,
            rollback_id=rollback_id,
            summary=f"ROLLBACK triggered: {reason}",
            raw_output=raw_output,
        )

    def _record_audit_pass(self, total: int, passed: int) -> None:
        try:
            audit = AuditLog(self._db)
            audit.append(
                self._config.agent_id,
                "behavioral_tests.passed",
                "governance_test_runner",
                "behavioral_test",
                detail={"tests_run": total, "tests_passed": passed},
            )
        except Exception:
            pass

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple[int, int, int]:
        """Extract pass/fail counts from pytest output. Returns (passed, failed, total)."""
        passed = failed = 0
        for line in output.splitlines():
            line = line.strip()
            if "passed" in line or "failed" in line:
                import re
                p = re.search(r"(\d+) passed", line)
                f = re.search(r"(\d+) failed", line)
                if p:
                    passed = int(p.group(1))
                if f:
                    failed = int(f.group(1))
        total = passed + failed
        return passed, failed, total


def main() -> None:
    """CLI entry point for running as ``python -m ai_governance.test_runner``."""
    import os
    config = GovernanceConfig.default_customer_service()
    db = GovernanceDB(os.environ.get("GOVERNANCE_DB", ":memory:"))
    runner = GovernanceTestRunner(config, db)
    result = runner.run()

    if result.passed:
        print(f"\n✓ {result.summary}")
        sys.exit(0)
    else:
        print(f"\n✗ {result.summary}")
        print(f"  Rollback ID: {result.rollback_id}")
        print(f"  Tests: {result.tests_passed} passed, {result.tests_failed} failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
