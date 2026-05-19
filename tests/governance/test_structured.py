"""Tests for the constrained-decoding structured output layer.

Covers GovernanceDecision model validation, DecisionDecoder.decode(),
ingestion_structured() integration, and the new API endpoints.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.config import GovernanceConfig
from ai_governance.ingestion import IngestionLayer
from ai_governance.storage import GovernanceDB
from ai_governance.structured import DecodeError, DecisionDecoder, GovernanceDecision


# ── GovernanceDecision model ─────────────────────────────────────────────────

class TestGovernanceDecisionModel:
    def test_valid_full(self):
        d = GovernanceDecision(
            case_category="fraud_claim",
            risk_level="critical",
            decision="escalate",
            confidence=0.95,
            flags=["high_value_customer", "prior_fraud"],
        )
        assert d.case_category == "fraud_claim"
        assert d.is_high_risk is True
        assert d.flags == ["high_value_customer", "prior_fraud"]

    def test_valid_minimal(self):
        d = GovernanceDecision(
            case_category="returns",
            risk_level="low",
            decision="resolve",
            confidence=0.8,
        )
        assert d.flags == []
        assert d.is_high_risk is False

    def test_risk_levels_valid(self):
        for level in ("low", "medium", "high", "critical"):
            d = GovernanceDecision(
                case_category="test",
                risk_level=level,
                decision="resolve",
                confidence=0.5,
            )
            assert d.risk_level == level

    def test_invalid_risk_level(self):
        with pytest.raises(Exception):
            GovernanceDecision(
                case_category="test",
                risk_level="extreme",
                decision="resolve",
                confidence=0.5,
            )

    def test_valid_decisions(self):
        for dec in ("resolve", "escalate", "deny", "defer", "partial_resolve"):
            GovernanceDecision(
                case_category="test",
                risk_level="low",
                decision=dec,
                confidence=0.7,
            )

    def test_invalid_decision(self):
        with pytest.raises(Exception):
            GovernanceDecision(
                case_category="test",
                risk_level="low",
                decision="approve",
                confidence=0.7,
            )

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            GovernanceDecision(
                case_category="test",
                risk_level="low",
                decision="resolve",
                confidence=1.1,
            )
        with pytest.raises(Exception):
            GovernanceDecision(
                case_category="test",
                risk_level="low",
                decision="resolve",
                confidence=-0.01,
            )

    def test_confidence_boundaries(self):
        GovernanceDecision(case_category="t", risk_level="low", decision="resolve", confidence=0.0)
        GovernanceDecision(case_category="t", risk_level="low", decision="resolve", confidence=1.0)

    def test_empty_category_rejected(self):
        with pytest.raises(Exception):
            GovernanceDecision(
                case_category="   ",
                risk_level="low",
                decision="resolve",
                confidence=0.5,
            )

    def test_is_high_risk_property(self):
        low = GovernanceDecision(case_category="t", risk_level="low", decision="resolve", confidence=0.5)
        med = GovernanceDecision(case_category="t", risk_level="medium", decision="resolve", confidence=0.5)
        high = GovernanceDecision(case_category="t", risk_level="high", decision="escalate", confidence=0.5)
        crit = GovernanceDecision(case_category="t", risk_level="critical", decision="escalate", confidence=0.5)
        assert low.is_high_risk is False
        assert med.is_high_risk is False
        assert high.is_high_risk is True
        assert crit.is_high_risk is True


# ── DecisionDecoder ───────────────────────────────────────────────────────────

class TestDecisionDecoder:
    @pytest.fixture
    def dec(self):
        return DecisionDecoder()

    def test_decode_dict(self, dec):
        raw = {
            "case_category": "billing_dispute",
            "risk_level": "high",
            "decision": "escalate",
            "confidence": 0.9,
            "flags": ["overdue"],
        }
        result = dec.decode(raw)
        assert isinstance(result, GovernanceDecision)
        assert result.case_category == "billing_dispute"

    def test_decode_json_string(self, dec):
        raw = json.dumps({
            "case_category": "fraud_claim",
            "risk_level": "critical",
            "decision": "escalate",
            "confidence": 0.99,
        })
        result = dec.decode(raw)
        assert result.risk_level == "critical"

    def test_decode_governance_decision_passthrough(self, dec):
        original = GovernanceDecision(
            case_category="returns",
            risk_level="low",
            decision="resolve",
            confidence=0.7,
        )
        assert dec.decode(original) is original

    def test_decode_invalid_risk_level(self, dec):
        with pytest.raises(DecodeError):
            dec.decode({"case_category": "x", "risk_level": "extreme",
                        "decision": "resolve", "confidence": 0.5})

    def test_decode_invalid_decision(self, dec):
        with pytest.raises(DecodeError):
            dec.decode({"case_category": "x", "risk_level": "low",
                        "decision": "approve", "confidence": 0.5})

    def test_decode_missing_required_field(self, dec):
        with pytest.raises(DecodeError):
            dec.decode({"case_category": "x", "risk_level": "low"})

    def test_decode_invalid_json_string(self, dec):
        with pytest.raises(DecodeError):
            dec.decode("{not valid json")

    def test_decode_wrong_type(self, dec):
        with pytest.raises(DecodeError):
            dec.decode(12345)

    def test_decode_confidence_out_of_bounds(self, dec):
        with pytest.raises(DecodeError):
            dec.decode({"case_category": "x", "risk_level": "low",
                        "decision": "resolve", "confidence": 2.0})

    def test_schema_json_contains_required_fields(self, dec):
        schema = json.loads(dec.schema_json())
        props = schema["properties"]
        assert "case_category" in props
        assert "risk_level" in props
        assert "decision" in props
        assert "confidence" in props
        assert "flags" in props

    def test_schema_json_enumerates_risk_levels(self, dec):
        schema = json.loads(dec.schema_json())
        risk_enum = schema["properties"]["risk_level"]["enum"]
        assert set(risk_enum) == {"low", "medium", "high", "critical"}

    def test_outlines_unavailable_error(self, monkeypatch):
        import ai_governance.structured as s
        monkeypatch.setattr(s, "_OUTLINES_AVAILABLE", False)
        with pytest.raises(ImportError):
            DecisionDecoder.wrap_anthropic_outlines(None)

    def test_instructor_unavailable_error(self, monkeypatch):
        import ai_governance.structured as s
        monkeypatch.setattr(s, "_INSTRUCTOR_AVAILABLE", False)
        with pytest.raises(ImportError):
            DecisionDecoder.wrap_anthropic_instructor(None)


# ── ingest_structured ────────────────────────────────────────────────────────

class TestIngestStructured:
    @pytest.fixture
    def ingestion(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        return IngestionLayer(cfg, db)

    def test_ingest_structured_basic(self, ingestion):
        d = GovernanceDecision(
            case_category="fraud_claim",
            risk_level="critical",
            decision="escalate",
            confidence=0.95,
        )
        result = ingestion.ingest_structured(d)
        assert result.event_id
        assert result.is_high_risk is True  # fraud_claim matches pattern

    def test_ingest_structured_metadata_stored(self, ingestion):
        d = GovernanceDecision(
            case_category="returns",
            risk_level="low",
            decision="resolve",
            confidence=0.8,
            flags=["routine"],
        )
        result = ingestion.ingest_structured(d)
        assert result.event_id

    def test_ingest_structured_case_id_propagated(self, ingestion):
        d = GovernanceDecision(
            case_category="billing_dispute",
            risk_level="high",
            decision="escalate",
            confidence=0.85,
        )
        result = ingestion.ingest_structured(d, case_id="CASE-999")
        assert result.event_id

    def test_ingest_structured_low_risk_not_flagged(self, ingestion):
        d = GovernanceDecision(
            case_category="shipping_inquiry",
            risk_level="low",
            decision="resolve",
            confidence=0.99,
        )
        result = ingestion.ingest_structured(d)
        assert result.is_high_risk is False


# ── API endpoints ─────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestStructuredAPIEndpoint:
    def test_post_structured_decision_object(self, client):
        resp = client.post("/events/structured", json={
            "decision": {
                "case_category": "fraud_claim",
                "risk_level": "critical",
                "decision": "escalate",
                "confidence": 0.97,
                "flags": ["prior_fraud"],
            }
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["event_id"]
        assert data["risk_level"] == "critical"
        assert data["confidence"] == 0.97
        assert data["is_high_risk"] is True

    def test_post_structured_raw_json_string(self, client):
        raw = json.dumps({
            "case_category": "billing_dispute",
            "risk_level": "high",
            "decision": "escalate",
            "confidence": 0.88,
        })
        resp = client.post("/events/structured", json={"raw": raw})
        assert resp.status_code == 201
        assert resp.json()["risk_level"] == "high"

    def test_post_structured_invalid_risk_level(self, client):
        resp = client.post("/events/structured", json={
            "decision": {
                "case_category": "test",
                "risk_level": "extreme",
                "decision": "resolve",
                "confidence": 0.5,
            }
        })
        assert resp.status_code == 422

    def test_post_structured_missing_both_fields(self, client):
        resp = client.post("/events/structured", json={})
        assert resp.status_code == 422

    def test_post_structured_raw_invalid_json(self, client):
        resp = client.post("/events/structured", json={"raw": "not json"})
        assert resp.status_code == 422

    def test_post_structured_invalid_decision_value(self, client):
        resp = client.post("/events/structured", json={
            "decision": {
                "case_category": "test",
                "risk_level": "low",
                "decision": "approve",
                "confidence": 0.7,
            }
        })
        assert resp.status_code == 422

    def test_get_decision_schema(self, client):
        resp = client.get("/events/schema")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["title"] == "GovernanceDecision"
        props = schema["properties"]
        assert "risk_level" in props
        assert "decision" in props
        assert "confidence" in props

    def test_structured_event_in_audit_log(self, client):
        client.post("/events/structured", json={
            "decision": {
                "case_category": "fraud_claim",
                "risk_level": "critical",
                "decision": "escalate",
                "confidence": 0.9,
            }
        })
        audit = client.get("/audit?action=decision.ingested").json()
        assert len(audit) >= 1
        assert audit[0]["detail"]["structured"] is True

    def test_structured_low_risk_not_flagged(self, client):
        resp = client.post("/events/structured", json={
            "decision": {
                "case_category": "shipping_inquiry",
                "risk_level": "low",
                "decision": "resolve",
                "confidence": 0.99,
            }
        })
        assert resp.status_code == 201
        assert resp.json()["is_high_risk"] is False
