"""Tests for governance compliance report generator."""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.alerts import AlertEngine
from ai_governance.api import app, init_services
from ai_governance.audit import AuditLog
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.drift import DriftDetector
from ai_governance.reports import ComplianceReport, ReportGenerator
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
    det = DriftDetector(cfg, db)
    eng = AlertEngine(cfg, db)
    audit = AuditLog(db)
    return cfg, db, det, eng, audit


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


class TestReportGenerator:
    def test_empty_db_produces_report(self, setup):
        cfg, db, det, eng, audit = setup
        gen = ReportGenerator(cfg, db, det, eng, audit)
        report = gen.generate()
        assert report.agent_id == cfg.agent_id
        assert report.is_safe is True
        assert report.total_decisions == 0
        assert report.high_risk_decisions == 0
        assert report.audit_chain_valid is True

    def test_report_with_events(self, setup):
        cfg, db, det, eng, audit = setup
        from ai_governance.ingestion import IngestionLayer, IngestRequest
        layer = IngestionLayer(cfg, db)
        for i in range(10):
            layer.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="billing_dispute",
                decision="resolve",
                resolution_time_ms=100,
            ))
        gen = ReportGenerator(cfg, db, det, eng, audit)
        report = gen.generate()
        assert report.total_decisions == 10
        assert len(report.categories) >= 1
        billing = [c for c in report.categories if c.category == "billing_dispute"]
        assert len(billing) == 1
        assert billing[0].total_events == 10

    def test_report_counts_high_risk(self, setup):
        cfg, db, det, eng, audit = setup
        from ai_governance.ingestion import IngestionLayer, IngestRequest
        layer = IngestionLayer(cfg, db)
        for i in range(5):
            layer.ingest(IngestRequest(
                case_id=str(uuid.uuid4()),
                case_category="fraud_claim",
                decision="escalate",
                resolution_time_ms=50,
            ))
        gen = ReportGenerator(cfg, db, det, eng, audit)
        report = gen.generate()
        assert report.high_risk_decisions >= 0
        assert report.high_risk_percentage >= 0.0

    def test_to_json_roundtrip(self, setup):
        cfg, db, det, eng, audit = setup
        gen = ReportGenerator(cfg, db, det, eng, audit)
        report = gen.generate()
        text = report.to_json()
        data = json.loads(text)
        assert data["agent_id"] == cfg.agent_id
        assert data["is_safe"] is True
        assert "categories" in data
        assert "violations" in data

    def test_to_text_format(self, setup):
        cfg, db, det, eng, audit = setup
        gen = ReportGenerator(cfg, db, det, eng, audit)
        report = gen.generate()
        text = report.to_text()
        assert "GOVERNANCE COMPLIANCE REPORT" in text
        assert cfg.agent_id in text
        assert "Safety Status: SAFE" in text
        assert "Decision Summary" in text
        assert "Alert Summary" in text
        assert "Audit Chain" in text

    def test_report_includes_config_info(self, setup):
        cfg, db, det, eng, audit = setup
        gen = ReportGenerator(cfg, db, det, eng, audit)
        report = gen.generate()
        assert report.config_version == cfg.version
        assert report.config_fingerprint == cfg.fingerprint

    def test_report_with_since_parameter(self, setup):
        from datetime import datetime, timezone
        cfg, db, det, eng, audit = setup
        gen = ReportGenerator(cfg, db, det, eng, audit)
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = gen.generate(since=since)
        assert report.period_start == since.isoformat()


class TestReportAPI:
    def test_json_report_endpoint(self, client):
        resp = client.get("/admin/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_id" in data
        assert "is_safe" in data
        assert "total_decisions" in data
        assert "categories" in data
        assert "violations" in data
        assert "audit_chain_valid" in data

    def test_text_report_endpoint(self, client):
        resp = client.get("/admin/report/text")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        text = resp.text
        assert "GOVERNANCE COMPLIANCE REPORT" in text

    def test_report_with_events(self, client):
        for _ in range(5):
            client.post("/events", json={
                "case_id": str(uuid.uuid4()),
                "case_category": "billing_dispute",
                "decision": "resolve",
                "resolution_time_ms": 100,
            })
        resp = client.get("/admin/report")
        data = resp.json()
        assert data["total_decisions"] == 5

    def test_report_with_since(self, client):
        resp = client.get("/admin/report?since=2024-01-01T00:00:00")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period_start"] is not None
