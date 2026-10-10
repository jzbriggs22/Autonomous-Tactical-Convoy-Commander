"""Tests for scheduler API endpoints."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import ai_governance.api as api_mod
from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    api_mod._scheduler = None
    yield
    if api_mod._scheduler is not None and api_mod._scheduler.is_running:
        api_mod._scheduler.stop()
    api_mod._scheduler = None
    reset_auth()


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    cfg.min_baseline_events = 5
    for thr in cfg.drift_thresholds:
        thr.min_baseline_samples = 5
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestSchedulerAPI:
    def test_status_when_not_started(self, client):
        resp = client.get("/admin/scheduler/status")
        assert resp.status_code == 200
        assert resp.json()["running"] is False

    def test_start_scheduler(self, client):
        resp = client.post("/admin/scheduler/start?interval_seconds=60")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["interval"] == 60.0

    def test_start_already_running(self, client):
        client.post("/admin/scheduler/start?interval_seconds=60")
        resp = client.post("/admin/scheduler/start?interval_seconds=60")
        assert resp.json()["status"] == "already_running"

    def test_stop_scheduler(self, client):
        client.post("/admin/scheduler/start?interval_seconds=60")
        resp = client.post("/admin/scheduler/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_stop_when_not_running(self, client):
        resp = client.post("/admin/scheduler/stop")
        assert resp.json()["status"] == "not_running"

    def test_status_after_start(self, client):
        client.post("/admin/scheduler/start?interval_seconds=0.1")
        time.sleep(0.4)
        resp = client.get("/admin/scheduler/status")
        data = resp.json()
        assert data["running"] is True
        assert data["total_runs"] >= 1

    def test_history_endpoint(self, client):
        client.post("/admin/scheduler/start?interval_seconds=0.1")
        time.sleep(0.4)
        resp = client.get("/admin/scheduler/history")
        assert resp.status_code == 200
        history = resp.json()
        assert len(history) >= 1
        assert history[0]["status"] == "ok"

    def test_history_empty_when_not_started(self, client):
        resp = client.get("/admin/scheduler/history")
        assert resp.status_code == 200
        assert resp.json() == []
