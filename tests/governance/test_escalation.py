"""Tests for alert escalation engine."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.escalation import AlertEscalator, EscalationRule, _SEVERITY_NEXT
from ai_governance.storage import AlertRecord, GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def setup():
    cfg = GovernanceConfig.default_customer_service()
    db = GovernanceDB(":memory:")
    return cfg, db


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


def _insert_old_alert(db, agent_id, severity, hours_ago):
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    rec = AlertRecord(
        alert_id=str(uuid.uuid4()),
        agent_id=agent_id,
        timestamp=ts,
        rule_name="test_rule",
        severity=severity,
        message=f"Test {severity} alert",
        metrics={},
        acknowledged=False,
    )
    db.insert_alert(rec)
    return rec


class TestSeverityLadder:
    def test_severity_order(self):
        assert _SEVERITY_NEXT["info"] == "warn"
        assert _SEVERITY_NEXT["warn"] == "critical"
        assert _SEVERITY_NEXT["critical"] == "rollback"
        assert _SEVERITY_NEXT["rollback"] is None

    def test_escalation_rule_defaults(self):
        from ai_governance.escalation import _DEFAULT_RULES
        severities = [r.from_severity for r in _DEFAULT_RULES]
        assert "info" in severities
        assert "warn" in severities
        assert "critical" in severities


class TestAlertEscalator:
    def test_no_overdue_alerts(self, setup):
        cfg, db = setup
        escalator = AlertEscalator(db, cfg.agent_id)
        events = escalator.run()
        assert len(events) == 0

    def test_fresh_alert_not_escalated(self, setup):
        cfg, db = setup
        _insert_old_alert(db, cfg.agent_id, "warn", hours_ago=0.1)
        rules = [EscalationRule("warn", escalate_after_seconds=3600)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        assert len(events) == 0

    def test_overdue_warn_escalates_to_critical(self, setup):
        cfg, db = setup
        _insert_old_alert(db, cfg.agent_id, "warn", hours_ago=5)
        rules = [EscalationRule("warn", escalate_after_seconds=3600)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        assert len(events) == 1
        assert events[0].from_severity == "warn"
        assert events[0].to_severity == "critical"

    def test_overdue_info_escalates_to_warn(self, setup):
        cfg, db = setup
        _insert_old_alert(db, cfg.agent_id, "info", hours_ago=25)
        rules = [EscalationRule("info", escalate_after_seconds=3600 * 24)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        assert len(events) == 1
        assert events[0].to_severity == "warn"

    def test_escalation_event_structure(self, setup):
        cfg, db = setup
        _insert_old_alert(db, cfg.agent_id, "warn", hours_ago=5)
        rules = [EscalationRule("warn", escalate_after_seconds=3600)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        e = events[0]
        assert e.original_alert_id
        assert e.new_alert_id != e.original_alert_id
        assert e.from_severity == "warn"
        assert e.to_severity == "critical"
        assert "ESCALATED" in e.message
        assert e.escalated_at

    def test_escalation_creates_new_alert(self, setup):
        cfg, db = setup
        old_alert = _insert_old_alert(db, cfg.agent_id, "warn", hours_ago=5)
        rules = [EscalationRule("warn", escalate_after_seconds=3600)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        new_alerts = db.get_recent_alerts(cfg.agent_id, limit=10)
        new_ids = {a.alert_id for a in new_alerts}
        assert events[0].new_alert_id in new_ids

    def test_acknowledged_alert_not_escalated(self, setup):
        cfg, db = setup
        old_alert = _insert_old_alert(db, cfg.agent_id, "warn", hours_ago=5)
        db.acknowledge_alert(cfg.agent_id, old_alert.alert_id)
        rules = [EscalationRule("warn", escalate_after_seconds=3600)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        assert len(events) == 0

    def test_rollback_not_escalated(self, setup):
        cfg, db = setup
        _insert_old_alert(db, cfg.agent_id, "rollback", hours_ago=10)
        rules = [EscalationRule("rollback", escalate_after_seconds=60)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        assert len(events) == 0

    def test_multiple_overdue_alerts(self, setup):
        cfg, db = setup
        for _ in range(3):
            _insert_old_alert(db, cfg.agent_id, "warn", hours_ago=5)
        rules = [EscalationRule("warn", escalate_after_seconds=3600)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        assert len(events) == 3

    def test_custom_rules(self, setup):
        cfg, db = setup
        _insert_old_alert(db, cfg.agent_id, "info", hours_ago=2)
        rules = [EscalationRule("info", escalate_after_seconds=3600)]
        escalator = AlertEscalator(db, cfg.agent_id, rules=rules)
        events = escalator.run()
        assert len(events) == 1
        assert events[0].to_severity == "warn"


class TestEscalationAPI:
    def test_escalate_endpoint_no_overdue(self, client):
        resp = client.post("/admin/escalate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["escalated"] == 0
        assert data["events"] == []

    def test_escalate_endpoint_structure(self, client):
        resp = client.post("/admin/escalate")
        data = resp.json()
        assert "escalated" in data
        assert "events" in data
