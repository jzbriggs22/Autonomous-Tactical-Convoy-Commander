"""Tests for deployment readiness checker."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.ingestion import IngestionLayer, IngestRequest
from ai_governance.readiness import ReadinessChecker, ReadinessReport
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def setup():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 5
    cfg.recent_window_size = 20
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
        thr.recent_window = 10
    db = GovernanceDB(":memory:")
    return cfg, db


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 5
    cfg.recent_window_size = 20
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
        thr.recent_window = 10
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestReadinessChecker:
    def test_empty_db_has_blockers(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        assert not report.ready
        assert len(report.blockers) > 0

    def test_report_structure(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        assert report.agent_id == cfg.agent_id
        assert report.config_version == cfg.version
        assert report.config_fingerprint == cfg.fingerprint
        assert isinstance(report.ready, bool)
        assert isinstance(report.blockers, list)
        assert isinstance(report.warnings, list)
        assert isinstance(report.passed, list)

    def test_total_checks(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        assert report.total_checks == len(report.blockers) + len(report.warnings) + len(report.passed)

    def test_summary_contains_status(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        text = report.summary()
        assert "Deployment Readiness" in text
        assert cfg.agent_id in text
        assert "READY" in text or "NOT READY" in text

    def test_no_active_rollbacks_passes_when_safe(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        rollback_checks = [c for c in report.passed if c.name == "no_active_rollbacks"]
        assert len(rollback_checks) == 1

    def test_thresholds_check_passes(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        threshold_checks = [
            c for c in (report.passed + report.blockers)
            if c.name == "drift_thresholds_configured"
        ]
        assert len(threshold_checks) == 1

    def test_audit_chain_passes_on_fresh_db(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        audit_checks = [c for c in report.passed if c.name == "audit_chain_integrity"]
        assert len(audit_checks) == 1

    def test_required_categories_blocker(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db, required_categories=["fraud_claim"])
        report = checker.check()
        required_checks = [
            c for c in report.blockers
            if c.name == "required_category_fraud_claim"
        ]
        assert len(required_checks) == 1

    def test_required_categories_passes_with_data(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        for _ in range(3):
            layer.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="fraud_claim",
                decision="escalate",
                resolution_time_ms=100,
            ))
        checker = ReadinessChecker(cfg, db, required_categories=["fraud_claim"])
        report = checker.check()
        required_checks = [
            c for c in report.passed
            if c.name == "required_category_fraud_claim"
        ]
        assert len(required_checks) == 1

    def test_baseline_blocker_without_data(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        baseline_blockers = [
            c for c in report.blockers
            if c.name.startswith("baseline_data_")
        ]
        assert len(baseline_blockers) > 0

    def test_config_fingerprint_passes(self, setup):
        cfg, db = setup
        checker = ReadinessChecker(cfg, db)
        report = checker.check()
        fp_checks = [
            c for c in (report.passed + report.warnings)
            if c.name == "config_fingerprinted"
        ]
        assert len(fp_checks) == 1


class TestReadinessAPI:
    def test_readiness_endpoint(self, client):
        resp = client.get("/admin/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert "ready" in data
        assert "agent_id" in data
        assert "blockers" in data
        assert "warnings" in data
        assert "passed" in data
        assert "summary" in data

    def test_readiness_with_required_categories(self, client):
        resp = client.get("/admin/readiness?required_categories=fraud_claim,billing")
        assert resp.status_code == 200
        data = resp.json()
        assert not data["ready"]
        names = [c["name"] for c in data["blockers"]]
        assert "required_category_fraud_claim" in names

    def test_readiness_check_detail(self, client):
        resp = client.get("/admin/readiness")
        data = resp.json()
        all_checks = data["blockers"] + data["warnings"] + data["passed"]
        for check in all_checks:
            assert "name" in check
            assert "passed" in check
            assert "severity" in check
            assert "message" in check

    def test_agents_endpoint(self, client):
        resp = client.get("/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert isinstance(agents, list)

    def test_agents_after_ingest(self, client):
        for _ in range(3):
            client.post("/events", json={
                "case_id": str(uuid.uuid4()),
                "case_category": "billing",
                "decision": "resolve",
                "resolution_time_ms": 100,
            })
        resp = client.get("/agents")
        agents = resp.json()
        assert len(agents) >= 1
        current = [a for a in agents if a["is_current"]]
        assert len(current) == 1
        assert current[0]["decision_count"] == 3
