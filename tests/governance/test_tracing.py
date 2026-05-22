"""Tests for OpenTelemetry tracing integration."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import GovernanceDB
from ai_governance import tracing


@pytest.fixture(autouse=True)
def _clear_spans():
    tracing.clear_spans()
    yield
    tracing.clear_spans()


@pytest.fixture(autouse=True)
def _clean_auth():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 10
    cfg.recent_window_size = 30
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 10
        thr.recent_window = 15
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestTracingModule:
    def test_span_context_manager_records_span(self):
        with tracing.span("test.operation", key="value"):
            pass
        names = tracing.span_names()
        assert "test.operation" in names

    def test_span_records_attributes(self):
        with tracing.span("test.attrs", agent_id="ag1", score=0.75):
            pass
        spans = tracing.get_finished_spans()
        sp = next(s for s in spans if s.name == "test.attrs")
        assert sp.attributes.get("agent_id") == "ag1"
        assert sp.attributes.get("score") == 0.75

    def test_span_marks_error_on_exception(self):
        from opentelemetry.trace import StatusCode
        with pytest.raises(ValueError):
            with tracing.span("test.error"):
                raise ValueError("boom")
        spans = tracing.get_finished_spans()
        sp = next(s for s in spans if s.name == "test.error")
        assert sp.status.status_code == StatusCode.ERROR

    def test_clear_spans_works(self):
        with tracing.span("first"):
            pass
        tracing.clear_spans()
        assert tracing.get_finished_spans() == []

    def test_none_attributes_not_set(self):
        with tracing.span("test.none", optional=None, required="yes"):
            pass
        spans = tracing.get_finished_spans()
        sp = next(s for s in spans if s.name == "test.none")
        assert "optional" not in sp.attributes
        assert sp.attributes.get("required") == "yes"


class TestIngestionTracing:
    def test_ingest_creates_span(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)
        ing.ingest(IngestRequest(
            case_id="t1", case_category="billing_dispute",
            decision="resolve", resolution_time_ms=100,
        ))
        assert "governance.ingest" in tracing.span_names()

    def test_ingest_span_has_attributes(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)
        ing.ingest(IngestRequest(
            case_id="t2", case_category="fraud_claim",
            decision="escalate", resolution_time_ms=200,
        ))
        spans = tracing.get_finished_spans()
        sp = next(s for s in spans if s.name == "governance.ingest")
        assert sp.attributes.get("agent_id") == cfg.agent_id
        assert sp.attributes.get("case_category") == "fraud_claim"
        assert sp.attributes.get("decision") == "escalate"
        assert "is_high_risk" in sp.attributes

    def test_high_risk_reflected_in_span(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)
        ing.ingest(IngestRequest(
            case_id="hr1", case_category="fraud_claim",
            decision="escalate", resolution_time_ms=100,
        ))
        spans = tracing.get_finished_spans()
        sp = next(s for s in spans if s.name == "governance.ingest")
        assert sp.attributes.get("is_high_risk") is True


class TestDriftTracing:
    def test_drift_detect_creates_span(self, client):
        client.get("/drift")
        assert "governance.drift.detect" in tracing.span_names()

    def test_drift_span_has_attributes(self, client):
        client.get("/drift")
        spans = tracing.get_finished_spans()
        sp = next(s for s in spans if s.name == "governance.drift.detect")
        assert "violations" in sp.attributes
        assert "overall_drift_score" in sp.attributes
        assert "categories_checked" in sp.attributes

    def test_violations_count_in_span(self, client):
        client.get("/drift")
        spans = tracing.get_finished_spans()
        sp = next(s for s in spans if s.name == "governance.drift.detect")
        assert sp.attributes["violations"] >= 0


class TestOtlpConfig:
    def test_configure_otlp_no_op_without_endpoint(self):
        tracing.configure_otlp(endpoint=None)

    def test_configure_otlp_no_op_without_package(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.grpc.trace_exporter", None)
        tracing.configure_otlp(endpoint="http://localhost:4317")
