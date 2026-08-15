"""Tests for the decision explanation engine."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.explainer import DecisionExplainer, DecisionExplanation
from ai_governance.ingestion import IngestionLayer, IngestRequest
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
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
    db = GovernanceDB(":memory:")
    return cfg, db


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 5
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestDecisionExplainer:
    def test_explain_nonexistent_returns_none(self, setup):
        cfg, db = setup
        explainer = DecisionExplainer(cfg, db)
        result = explainer.explain("nonexistent-id")
        assert result is None

    def test_explain_low_risk_event(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        result = layer.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category="routine",
            decision="resolve",
            resolution_time_ms=100,
        ))
        explainer = DecisionExplainer(cfg, db)
        explanation = explainer.explain(result.event_id)
        assert explanation is not None
        assert explanation.event_id == result.event_id
        assert explanation.is_high_risk is False
        assert len(explanation.matched_patterns) == 0
        assert "NOT classified as high-risk" in explanation.audit_narrative

    def test_explain_high_risk_event(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        result = layer.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category="fraud_claim",
            decision="escalate",
            resolution_time_ms=50,
        ))
        explainer = DecisionExplainer(cfg, db)
        explanation = explainer.explain(result.event_id)
        assert explanation is not None
        assert explanation.is_high_risk is True
        assert len(explanation.matched_patterns) > 0
        assert explanation.high_risk_score > 0.0
        assert explanation.dominant_pattern is not None

    def test_explanation_fields(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        result = layer.ingest(IngestRequest(
            case_id="case-123",
            case_category="billing_dispute",
            decision="resolve",
            resolution_time_ms=200,
        ))
        explainer = DecisionExplainer(cfg, db)
        expl = explainer.explain(result.event_id)
        assert expl.case_id == "case-123"
        assert expl.case_category == "billing_dispute"
        assert expl.decision == "resolve"
        assert expl.config_version == cfg.version
        assert isinstance(expl.category_thresholds, list)
        assert isinstance(expl.baseline_metrics, dict)
        assert isinstance(expl.unmatched_patterns, list)

    def test_pattern_match_details(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        result = layer.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category="fraud_claim",
            decision="escalate",
            resolution_time_ms=50,
        ))
        explainer = DecisionExplainer(cfg, db)
        expl = explainer.explain(result.event_id)
        for match in expl.matched_patterns:
            assert match.pattern_name
            assert match.field
            assert match.pattern
            assert match.weight > 0
            assert 0.0 <= match.contribution <= 1.0

    def test_dominant_pattern_is_highest_weight(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        result = layer.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category="fraud_claim",
            decision="escalate",
            resolution_time_ms=50,
        ))
        explainer = DecisionExplainer(cfg, db)
        expl = explainer.explain(result.event_id)
        if len(expl.matched_patterns) > 1:
            dom = expl.dominant_pattern
            dom_match = next(m for m in expl.matched_patterns if m.pattern_name == dom)
            assert all(dom_match.weight >= m.weight for m in expl.matched_patterns)

    def test_no_patterns_dominant_is_none(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        result = layer.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category="routine",
            decision="resolve",
            resolution_time_ms=100,
        ))
        explainer = DecisionExplainer(cfg, db)
        expl = explainer.explain(result.event_id)
        assert expl.dominant_pattern is None

    def test_audit_narrative_contains_key_info(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        result = layer.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category="fraud_claim",
            decision="escalate",
            resolution_time_ms=50,
        ))
        explainer = DecisionExplainer(cfg, db)
        expl = explainer.explain(result.event_id)
        narrative = expl.audit_narrative
        assert expl.event_id in narrative
        assert "fraud_claim" in narrative

    def test_category_thresholds_relevant(self, setup):
        cfg, db = setup
        layer = IngestionLayer(cfg, db)
        result = layer.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category="billing_dispute",
            decision="resolve",
            resolution_time_ms=100,
        ))
        explainer = DecisionExplainer(cfg, db)
        expl = explainer.explain(result.event_id)
        for thr in expl.category_thresholds:
            assert "name" in thr
            assert "metric" in thr
            assert "severity" in thr


class TestExplainerAPI:
    def test_explain_not_found(self, client):
        resp = client.get("/events/nonexistent-id/explain")
        assert resp.status_code == 404

    def test_explain_low_risk_event(self, client):
        ingest_resp = client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": "routine",
            "decision": "resolve",
            "resolution_time_ms": 100,
        })
        event_id = ingest_resp.json()["event_id"]
        resp = client.get(f"/events/{event_id}/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_id"] == event_id
        assert data["is_high_risk"] is False
        assert "audit_narrative" in data
        assert "matched_patterns" in data
        assert "category_thresholds" in data

    def test_explain_high_risk_event(self, client):
        ingest_resp = client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": "fraud_claim",
            "decision": "escalate",
            "resolution_time_ms": 50,
        })
        event_id = ingest_resp.json()["event_id"]
        resp = client.get(f"/events/{event_id}/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_high_risk"] is True
        assert data["high_risk_score"] > 0.0
        assert len(data["matched_patterns"]) > 0
        assert data["dominant_pattern"] is not None
        for p in data["matched_patterns"]:
            assert "name" in p
            assert "weight" in p
            assert "contribution" in p
