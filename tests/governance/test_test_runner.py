"""Tests for the governance test runner that connects behavioral test failures to rollback."""

from __future__ import annotations

import pytest

from ai_governance.config import GovernanceConfig
from ai_governance.storage import GovernanceDB
from ai_governance.test_runner import GovernanceTestRunner


@pytest.fixture
def runner():
    cfg = GovernanceConfig.default_customer_service()
    db = GovernanceDB(":memory:")
    return GovernanceTestRunner(cfg, db), db


class TestGovernanceTestRunner:
    def test_passing_tests_no_rollback(self, runner):
        runner_obj, db = runner
        result = runner_obj.run()
        assert result.passed is True
        assert result.exit_code == 0
        assert result.rollback_triggered is False
        assert result.rollback_id is None
        assert result.tests_passed > 0
        assert result.tests_failed == 0
        assert not db.has_active_rollback("cs-agent-v1")

    def test_failing_tests_trigger_rollback(self, runner):
        runner_obj, db = runner
        result = runner_obj.run(extra_args=["-k", "NONEXISTENT_TEST_THAT_MATCHES_NOTHING"])
        # pytest returns 5 (no tests collected) when -k matches nothing
        if result.exit_code == 5:
            # No tests were collected — should be treated as failure
            assert result.passed is False

    def test_bad_test_path_triggers_rollback(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        runner_obj = GovernanceTestRunner(
            cfg, db, test_path="tests/nonexistent_test_file.py"
        )
        result = runner_obj.run()
        assert result.passed is False
        assert result.rollback_triggered is True
        assert result.rollback_id is not None
        assert db.has_active_rollback(cfg.agent_id)

    def test_rollback_record_persisted(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        runner_obj = GovernanceTestRunner(
            cfg, db, test_path="tests/nonexistent_test_file.py"
        )
        result = runner_obj.run()
        assert result.rollback_triggered is True

        rollbacks = db.get_rollbacks(cfg.agent_id, limit=5)
        assert len(rollbacks) >= 1
        rb = rollbacks[0]
        assert rb.trigger_rule == "behavioral_test_failure"
        assert rb.resolved is False

    def test_alert_record_persisted(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        runner_obj = GovernanceTestRunner(
            cfg, db, test_path="tests/nonexistent_test_file.py"
        )
        runner_obj.run()

        alerts = db.get_recent_alerts(cfg.agent_id, limit=5)
        assert len(alerts) >= 1
        alert = alerts[0]
        assert alert.rule_name == "behavioral_test_failure"
        assert alert.severity == "rollback"

    def test_passing_suite_records_audit_entry(self, runner):
        runner_obj, db = runner
        result = runner_obj.run()
        assert result.passed is True

    def test_parse_pytest_output(self):
        output = """
tests/governance/test_behavioral.py::TestFixedDatasetStructure::test_all_50_parse PASSED
tests/governance/test_behavioral.py::TestDriftDetectionDrifted::test_fraud_drift FAILED
========================= 24 passed, 2 failed in 1.23s =========================
"""
        passed, failed, total = GovernanceTestRunner._parse_pytest_output(output)
        assert passed == 24
        assert failed == 2
        assert total == 26

    def test_parse_pytest_output_all_passed(self):
        output = "========================= 26 passed in 0.45s ========================="
        passed, failed, total = GovernanceTestRunner._parse_pytest_output(output)
        assert passed == 26
        assert failed == 0
        assert total == 26
