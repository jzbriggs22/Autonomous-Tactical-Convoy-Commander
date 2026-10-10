"""Tests for the ingestion layer: validation, risk classification, and storage."""

import uuid
from datetime import datetime, timezone

import pytest

from ai_governance.ingestion import IngestRequest, IngestionLayer, ValidationError

from .conftest import make_request, seed_decisions


class TestValidation:
    def test_rejects_empty_case_id(self, ingestion):
        with pytest.raises(ValidationError, match="case_id"):
            ingestion.ingest(make_request(case_id="  "))

    def test_rejects_empty_category(self, ingestion):
        req = make_request()
        req.case_category = ""
        with pytest.raises(ValidationError, match="case_category"):
            ingestion.ingest(req)

    def test_rejects_invalid_decision(self, ingestion):
        req = make_request()
        req.decision = "maybe"
        with pytest.raises(ValidationError, match="decision"):
            ingestion.ingest(req)

    def test_rejects_negative_response_time(self, ingestion):
        req = make_request(rt_ms=-1)
        with pytest.raises(ValidationError, match="resolution_time_ms"):
            ingestion.ingest(req)

    def test_all_valid_decisions_accepted(self, ingestion):
        for decision in ("resolve", "escalate", "deny", "defer", "partial_resolve"):
            r = ingestion.ingest(make_request(decision=decision))
            assert r.event_id

    def test_rejects_empty_event_id(self, ingestion):
        req = make_request()
        req.event_id = "   "
        with pytest.raises(ValidationError, match="event_id"):
            ingestion.ingest(req)

    def test_rejects_invalid_inline_ground_truth(self, ingestion):
        """ground_truth supplied at ingest time must pass the same validation
        as add_ground_truth — arbitrary labels would corrupt accuracy metrics."""
        req = make_request(ground_truth="definitely_not_a_decision")
        with pytest.raises(ValidationError, match="ground truth"):
            ingestion.ingest(req)

    def test_accepts_valid_inline_ground_truth(self, ingestion, db, config):
        r = ingestion.ingest(make_request(ground_truth="escalate"))
        rec = db.get_decision_by_id(r.event_id, config.agent_id)
        assert rec.ground_truth == "escalate"

    def test_naive_timestamp_normalized_to_utc(self, ingestion, db, config):
        """Naive ISO timestamps are common client input; they must come back
        timezone-aware or dashboard windowing raises TypeError."""
        req = make_request()
        req.timestamp = datetime(2026, 8, 11, 12, 0, 0)  # no tzinfo
        r = ingestion.ingest(req)
        rec = db.get_decision_by_id(r.event_id, config.agent_id)
        assert rec.timestamp.tzinfo is not None
        # subtracting from an aware now must not raise
        _ = datetime.now(timezone.utc) - rec.timestamp


class TestRiskClassification:
    def test_fraud_category_is_high_risk(self, ingestion):
        r = ingestion.ingest(make_request(category="fraud_claim", decision="escalate"))
        assert r.is_high_risk is True
        assert "fraud_claim" in r.matched_patterns

    def test_policy_category_is_high_risk(self, ingestion):
        r = ingestion.ingest(make_request(category="policy_question"))
        assert r.is_high_risk is True

    def test_normal_category_is_not_high_risk(self, ingestion):
        r = ingestion.ingest(make_request(category="returns"))
        assert r.is_high_risk is False
        assert r.matched_patterns == []

    def test_enterprise_account_metadata_is_high_risk(self, ingestion):
        r = ingestion.ingest(make_request(
            category="returns",
            metadata={"account_tier": "enterprise"},
        ))
        assert r.is_high_risk is True
        assert "high_value_account" in r.matched_patterns

    def test_risk_score_bounded(self, ingestion):
        # Even if multiple patterns match, score stays ≤ 1.0
        r = ingestion.ingest(make_request(
            category="fraud_claim",
            metadata={"account_tier": "enterprise"},
        ))
        assert 0.0 < r.high_risk_score <= 1.0

    def test_multiple_patterns_increase_score(self, ingestion):
        single = ingestion.ingest(make_request(category="fraud_claim"))
        multi = ingestion.ingest(make_request(
            category="fraud_claim",
            metadata={"account_tier": "enterprise"},
        ))
        assert multi.high_risk_score >= single.high_risk_score

    def test_billing_dispute_is_high_risk(self, ingestion):
        r = ingestion.ingest(make_request(category="billing_dispute"))
        assert r.is_high_risk is True


class TestPersistence:
    def test_event_stored_with_correct_fields(self, ingestion, db, config):
        event_id = str(uuid.uuid4())
        req = IngestRequest(
            event_id=event_id,
            case_id="case-123",
            case_category="billing_dispute",
            decision="resolve",
            resolution_time_ms=450,
            metadata={"region": "us-west"},
            timestamp=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )
        ingestion.ingest(req)
        records = db.get_recent_decisions(config.agent_id, category="billing_dispute")
        assert len(records) == 1
        r = records[0]
        assert r.event_id == event_id
        assert r.case_id == "case-123"
        assert r.decision == "resolve"
        assert r.resolution_time_ms == 450
        assert r.metadata["region"] == "us-west"

    def test_batch_ingest_stores_all(self, ingestion, db, config):
        reqs = [make_request(category="billing_dispute") for _ in range(10)]
        results = ingestion.ingest_batch(reqs)
        assert len(results) == 10
        assert db.count_decisions(config.agent_id) == 10

    def test_ground_truth_update(self, ingestion, db, config):
        r = ingestion.ingest(make_request(category="fraud_claim", decision="resolve"))
        updated = ingestion.add_ground_truth(r.event_id, "escalate")
        assert updated is True
        records = db.get_recent_decisions(config.agent_id)
        assert records[0].ground_truth == "escalate"

    def test_ground_truth_invalid_raises(self, ingestion):
        r = ingestion.ingest(make_request())
        with pytest.raises(ValidationError):
            ingestion.add_ground_truth(r.event_id, "unknown_label")

    def test_ground_truth_missing_event_returns_false(self, ingestion):
        ok = ingestion.add_ground_truth("nonexistent-id", "resolve")
        assert ok is False

    def test_idempotent_event_id(self, ingestion, db, config):
        event_id = str(uuid.uuid4())
        req = make_request()
        req.event_id = event_id
        ingestion.ingest(req)
        ingestion.ingest(req)  # second ingest — same event_id
        assert db.count_decisions(config.agent_id) == 1
