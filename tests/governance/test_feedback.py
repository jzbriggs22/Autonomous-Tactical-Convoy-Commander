"""Tests for ground-truth feedback pipeline."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.feedback import (
    AccuracyReport,
    CategoryAccuracy,
    FeedbackLabel,
    FeedbackPipeline,
    FeedbackResult,
)
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


def _cfg():
    return GovernanceConfig.default_customer_service()


def _ingest(db, cfg, events):
    layer = IngestionLayer(cfg, db)
    ids = []
    for ev in events:
        r = layer.ingest(IngestRequest(
            case_id=ev.get("case_id", "c1"),
            case_category=ev["category"],
            decision=ev["decision"],
            resolution_time_ms=ev.get("resolution_ms", 500),
        ))
        ids.append(r.event_id)
    return ids


class TestFeedbackLabel:
    def test_apply_labels_to_existing_events(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        ids = _ingest(db, cfg, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "deny"},
        ])
        pipeline = FeedbackPipeline(cfg, db)
        result = pipeline.apply_labels([
            FeedbackLabel(event_id=ids[0], ground_truth="resolve"),
            FeedbackLabel(event_id=ids[1], ground_truth="deny"),
        ])
        assert result.total_submitted == 2
        assert result.applied == 2
        assert result.not_found == 0
        assert result.invalid == 0
        assert result.errors == []

    def test_apply_labels_not_found(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        pipeline = FeedbackPipeline(cfg, db)
        result = pipeline.apply_labels([
            FeedbackLabel(event_id="nonexistent-id", ground_truth="resolve"),
        ])
        assert result.applied == 0
        assert result.not_found == 1
        assert len(result.errors) == 1
        assert "not found" in result.errors[0]

    def test_apply_labels_invalid_ground_truth(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        pipeline = FeedbackPipeline(cfg, db)
        result = pipeline.apply_labels([
            FeedbackLabel(event_id="some-id", ground_truth="INVALID_DECISION"),
        ])
        assert result.applied == 0
        assert result.invalid == 1
        assert "Invalid ground_truth" in result.errors[0]

    def test_apply_labels_mixed(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        ids = _ingest(db, cfg, [
            {"category": "billing_dispute", "decision": "resolve"},
        ])
        pipeline = FeedbackPipeline(cfg, db)
        result = pipeline.apply_labels([
            FeedbackLabel(event_id=ids[0], ground_truth="resolve"),
            FeedbackLabel(event_id="missing", ground_truth="deny"),
            FeedbackLabel(event_id="whatever", ground_truth="INVALID"),
        ])
        assert result.total_submitted == 3
        assert result.applied == 1
        assert result.not_found == 1
        assert result.invalid == 1
        assert len(result.errors) == 2

    def test_errors_capped_at_50(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        pipeline = FeedbackPipeline(cfg, db)
        labels = [FeedbackLabel(event_id=f"bad-{i}", ground_truth="resolve") for i in range(100)]
        result = pipeline.apply_labels(labels)
        assert result.not_found == 100
        assert len(result.errors) == 50

    def test_empty_labels_list(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        pipeline = FeedbackPipeline(cfg, db)
        result = pipeline.apply_labels([])
        assert result.total_submitted == 0
        assert result.applied == 0


class TestAccuracyComputation:
    def test_no_decisions_returns_empty(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        pipeline = FeedbackPipeline(cfg, db)
        report = pipeline.compute_accuracy()
        assert isinstance(report, AccuracyReport)
        assert report.total_decisions == 0
        assert report.total_labeled == 0
        assert report.overall_accuracy is None
        assert report.categories == []

    def test_accuracy_with_labeled_decisions(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        ids = _ingest(db, cfg, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "deny"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "deny"},
            {"category": "billing_dispute", "decision": "resolve"},
        ])
        pipeline = FeedbackPipeline(cfg, db)
        pipeline.apply_labels([
            FeedbackLabel(event_id=ids[0], ground_truth="resolve"),
            FeedbackLabel(event_id=ids[1], ground_truth="deny"),
            FeedbackLabel(event_id=ids[2], ground_truth="deny"),
            FeedbackLabel(event_id=ids[3], ground_truth="deny"),
            FeedbackLabel(event_id=ids[4], ground_truth="resolve"),
        ])
        report = pipeline.compute_accuracy()
        assert report.total_decisions == 5
        assert report.total_labeled == 5
        assert report.overall_accuracy == pytest.approx(4 / 5)
        assert len(report.categories) == 1
        cat = report.categories[0]
        assert cat.category == "billing_dispute"
        assert cat.correct == 4
        assert cat.incorrect == 1

    def test_accuracy_needs_5_labels_for_overall(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        ids = _ingest(db, cfg, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
        ])
        pipeline = FeedbackPipeline(cfg, db)
        pipeline.apply_labels([
            FeedbackLabel(event_id=ids[i], ground_truth="resolve") for i in range(4)
        ])
        report = pipeline.compute_accuracy()
        assert report.overall_accuracy is None

    def test_multi_category_accuracy(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        ids = _ingest(db, cfg, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "deny"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "account_closure", "decision": "resolve"},
            {"category": "account_closure", "decision": "deny"},
        ])
        pipeline = FeedbackPipeline(cfg, db)
        pipeline.apply_labels([
            FeedbackLabel(event_id=ids[0], ground_truth="resolve"),
            FeedbackLabel(event_id=ids[1], ground_truth="resolve"),
            FeedbackLabel(event_id=ids[2], ground_truth="resolve"),
            FeedbackLabel(event_id=ids[3], ground_truth="resolve"),
            FeedbackLabel(event_id=ids[4], ground_truth="deny"),
        ])
        report = pipeline.compute_accuracy()
        assert report.total_labeled == 5
        cats = {c.category: c for c in report.categories}
        assert "billing_dispute" in cats
        assert "account_closure" in cats
        assert cats["billing_dispute"].correct == 2
        assert cats["billing_dispute"].incorrect == 1
        assert cats["account_closure"].correct == 2

    def test_unlabeled_categories_tracked(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        ids = _ingest(db, cfg, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "account_closure", "decision": "resolve"},
        ])
        pipeline = FeedbackPipeline(cfg, db)
        pipeline.apply_labels([
            FeedbackLabel(event_id=ids[0], ground_truth="resolve"),
        ])
        report = pipeline.compute_accuracy()
        assert "account_closure" in report.unlabeled_categories
        assert "billing_dispute" not in report.unlabeled_categories

    def test_labeling_coverage(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        ids = _ingest(db, cfg, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
        ])
        pipeline = FeedbackPipeline(cfg, db)
        pipeline.apply_labels([
            FeedbackLabel(event_id=ids[0], ground_truth="resolve"),
            FeedbackLabel(event_id=ids[1], ground_truth="resolve"),
        ])
        report = pipeline.compute_accuracy()
        assert report.labeling_coverage == pytest.approx(0.5)

    def test_summary_output(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        ids = _ingest(db, cfg, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "deny"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
        ])
        pipeline = FeedbackPipeline(cfg, db)
        pipeline.apply_labels([
            FeedbackLabel(event_id=ids[i], ground_truth="resolve") for i in range(5)
        ])
        report = pipeline.compute_accuracy()
        summary = report.summary()
        assert "Accuracy Report" in summary
        assert "billing_dispute" in summary
        assert "coverage" in summary

    def test_summary_na_when_few_labels(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        pipeline = FeedbackPipeline(cfg, db)
        report = pipeline.compute_accuracy()
        assert "N/A" in report.summary()


class TestFeedbackAPI:
    @pytest.fixture
    def client(self):
        cfg = _cfg()
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        return TestClient(app)

    def _ingest_via_api(self, client, events):
        ids = []
        for ev in events:
            resp = client.post("/events", json={
                "case_id": ev.get("case_id", "c1"),
                "case_category": ev["category"],
                "decision": ev["decision"],
                "resolution_time_ms": ev.get("resolution_ms", 500),
            })
            assert resp.status_code == 201
            ids.append(resp.json()["event_id"])
        return ids

    def test_apply_labels_endpoint(self, client):
        ids = self._ingest_via_api(client, [
            {"category": "billing_dispute", "decision": "resolve"},
        ])
        resp = client.post("/feedback/labels", json={
            "labels": [{"event_id": ids[0], "ground_truth": "resolve"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["applied"] == 1
        assert data["total_submitted"] == 1

    def test_apply_labels_missing_fields(self, client):
        resp = client.post("/feedback/labels", json={
            "labels": [{"event_id": "x"}],
        })
        assert resp.status_code == 422

    def test_apply_labels_empty_list(self, client):
        resp = client.post("/feedback/labels", json={"labels": []})
        assert resp.status_code == 422

    def test_apply_labels_not_found(self, client):
        resp = client.post("/feedback/labels", json={
            "labels": [{"event_id": "missing-id", "ground_truth": "resolve"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["not_found"] == 1

    def test_apply_labels_invalid_ground_truth(self, client):
        resp = client.post("/feedback/labels", json={
            "labels": [{"event_id": "x", "ground_truth": "INVALID"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["invalid"] == 1

    def test_accuracy_endpoint_empty(self, client):
        resp = client.get("/feedback/accuracy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_decisions"] == 0
        assert data["overall_accuracy"] is None
        assert "summary" in data

    def test_accuracy_endpoint_with_data(self, client):
        ids = self._ingest_via_api(client, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "deny"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "deny"},
        ])
        client.post("/feedback/labels", json={
            "labels": [
                {"event_id": ids[0], "ground_truth": "resolve"},
                {"event_id": ids[1], "ground_truth": "deny"},
                {"event_id": ids[2], "ground_truth": "resolve"},
                {"event_id": ids[3], "ground_truth": "resolve"},
                {"event_id": ids[4], "ground_truth": "deny"},
            ],
        })
        resp = client.get("/feedback/accuracy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_labeled"] == 5
        assert data["overall_accuracy"] == 1.0
        assert len(data["categories"]) == 1

    def test_full_pipeline_roundtrip(self, client):
        ids = self._ingest_via_api(client, [
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "billing_dispute", "decision": "deny"},
            {"category": "billing_dispute", "decision": "resolve"},
            {"category": "account_closure", "decision": "deny"},
            {"category": "account_closure", "decision": "resolve"},
        ])
        client.post("/feedback/labels", json={
            "labels": [
                {"event_id": ids[0], "ground_truth": "resolve"},
                {"event_id": ids[1], "ground_truth": "resolve"},
                {"event_id": ids[2], "ground_truth": "resolve"},
                {"event_id": ids[3], "ground_truth": "deny"},
                {"event_id": ids[4], "ground_truth": "deny"},
            ],
        })
        resp = client.get("/feedback/accuracy")
        data = resp.json()
        assert data["total_labeled"] == 5
        assert data["overall_accuracy"] is not None
        cats = {c["category"]: c for c in data["categories"]}
        assert cats["billing_dispute"]["correct"] == 2
        assert cats["billing_dispute"]["incorrect"] == 1
        assert cats["account_closure"]["correct"] == 1
        assert cats["account_closure"]["incorrect"] == 1
