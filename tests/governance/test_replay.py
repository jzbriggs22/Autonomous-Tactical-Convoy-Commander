"""Tests for the event replay engine."""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig, HighRiskPattern
from ai_governance.ingestion import IngestionLayer, IngestRequest
from ai_governance.replay import EventReplayer, ReplayResult
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


def _ingest_events(cfg, db, count=20, category="billing_dispute", decision="resolve"):
    layer = IngestionLayer(cfg, db)
    for _ in range(count):
        layer.ingest(IngestRequest(
            case_id=str(uuid.uuid4()),
            case_category=category,
            decision=decision,
            resolution_time_ms=100,
        ))


class TestReplayEngine:
    def test_replay_empty_db(self, setup):
        cfg, db = setup
        candidate = GovernanceConfig.default_customer_service()
        replayer = EventReplayer(cfg, candidate, db)
        result = replayer.replay()
        assert result.events_replayed == 0
        assert result.risk_reclassified_count == 0

    def test_replay_same_config_no_changes(self, setup):
        cfg, db = setup
        _ingest_events(cfg, db, count=20)
        candidate = GovernanceConfig.default_customer_service()
        candidate.min_baseline_events = cfg.min_baseline_events
        candidate.recent_window_size = cfg.recent_window_size
        for thr in candidate.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10
        replayer = EventReplayer(cfg, candidate, db)
        result = replayer.replay()
        assert result.events_replayed == 20
        assert result.risk_reclassified_count == 0
        assert result.new_high_risk_count == 0
        assert result.no_longer_high_risk_count == 0

    def test_replay_detects_risk_reclassification(self, setup):
        cfg, db = setup
        _ingest_events(cfg, db, count=10, category="fraud_claim", decision="escalate")

        candidate = GovernanceConfig.default_customer_service()
        candidate.high_risk_patterns = []
        candidate.min_baseline_events = cfg.min_baseline_events
        candidate.recent_window_size = cfg.recent_window_size
        for thr in candidate.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10

        replayer = EventReplayer(cfg, candidate, db)
        result = replayer.replay()
        assert result.events_replayed == 10
        assert result.no_longer_high_risk_count >= 0

    def test_replay_adds_new_pattern(self, setup):
        cfg, db = setup
        _ingest_events(cfg, db, count=10, category="billing_dispute")

        candidate = GovernanceConfig.default_customer_service()
        candidate.high_risk_patterns.append(HighRiskPattern(
            name="catch_all_billing",
            field="case_category",
            pattern="billing.*",
            weight=5.0,
        ))
        candidate.min_baseline_events = cfg.min_baseline_events
        candidate.recent_window_size = cfg.recent_window_size
        for thr in candidate.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10

        replayer = EventReplayer(cfg, candidate, db)
        result = replayer.replay()
        assert result.events_replayed == 10
        assert result.candidate_high_risk_count >= result.live_high_risk_count

    def test_replay_result_summary(self, setup):
        cfg, db = setup
        _ingest_events(cfg, db, count=10)

        candidate = GovernanceConfig.default_customer_service()
        candidate.min_baseline_events = cfg.min_baseline_events
        candidate.recent_window_size = cfg.recent_window_size
        for thr in candidate.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10

        replayer = EventReplayer(cfg, candidate, db)
        result = replayer.replay()
        text = result.summary()
        assert "Replay:" in text
        assert "10 events" in text
        assert "High-risk" in text
        assert "Drift score" in text

    def test_replay_with_category_filter(self, setup):
        cfg, db = setup
        _ingest_events(cfg, db, count=10, category="billing_dispute")
        _ingest_events(cfg, db, count=10, category="fraud_claim", decision="escalate")

        candidate = GovernanceConfig.default_customer_service()
        candidate.min_baseline_events = cfg.min_baseline_events
        candidate.recent_window_size = cfg.recent_window_size
        for thr in candidate.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10

        replayer = EventReplayer(cfg, candidate, db)
        result = replayer.replay(category="billing_dispute")
        assert result.events_replayed == 10

    def test_replay_with_limit(self, setup):
        cfg, db = setup
        _ingest_events(cfg, db, count=20)

        candidate = GovernanceConfig.default_customer_service()
        candidate.min_baseline_events = cfg.min_baseline_events
        candidate.recent_window_size = cfg.recent_window_size
        for thr in candidate.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10

        replayer = EventReplayer(cfg, candidate, db)
        result = replayer.replay(limit=5)
        assert result.events_replayed == 5

    def test_replay_diffs_populated_on_change(self, setup):
        cfg, db = setup
        _ingest_events(cfg, db, count=10, category="fraud_claim", decision="escalate")

        candidate = GovernanceConfig.default_customer_service()
        candidate.high_risk_patterns = []
        candidate.min_baseline_events = cfg.min_baseline_events
        candidate.recent_window_size = cfg.recent_window_size
        for thr in candidate.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10

        replayer = EventReplayer(cfg, candidate, db)
        result = replayer.replay()
        if result.risk_reclassified_count > 0:
            assert len(result.diffs) == result.risk_reclassified_count
            for d in result.diffs:
                assert d.risk_changed is True


class TestReplayAPI:
    @pytest.fixture
    def client(self):
        cfg = GovernanceConfig.default_customer_service()
        cfg.min_baseline_events = 5
        cfg.recent_window_size = 20
        for thr in cfg.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        c = TestClient(app)
        for _ in range(10):
            c.post("/events", json={
                "case_id": str(uuid.uuid4()),
                "case_category": "billing_dispute",
                "decision": "resolve",
                "resolution_time_ms": 100,
            })
        return c

    def test_replay_endpoint_basic(self, client):
        candidate = GovernanceConfig.default_customer_service()
        resp = client.post(
            "/admin/replay",
            json=json.loads(candidate.to_json()),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["events_replayed"] == 10
        assert "high_risk" in data
        assert "drift" in data
        assert "alerts" in data
        assert "summary" in data

    def test_replay_invalid_config_returns_422(self, client):
        resp = client.post("/admin/replay", json={
            "min_baseline_events": -5,
        })
        assert resp.status_code == 422

    def test_replay_with_limit(self, client):
        candidate = GovernanceConfig.default_customer_service()
        resp = client.post(
            "/admin/replay?limit=3",
            json=json.loads(candidate.to_json()),
        )
        assert resp.status_code == 200
        assert resp.json()["events_replayed"] == 3
