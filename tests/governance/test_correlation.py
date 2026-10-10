"""Tests for correlation ID middleware."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.correlation import (
    HEADER_NAME,
    CorrelationMiddleware,
    get_correlation_id,
)
from ai_governance.storage import GovernanceDB


@pytest.fixture(autouse=True)
def _clean():
    reset_auth()
    yield
    reset_auth()


@pytest.fixture
def client():
    cfg = GovernanceConfig.default_customer_service()
    db = GovernanceDB(":memory:")
    init_services(cfg, db)
    return TestClient(app)


class TestCorrelationID:
    def test_auto_generated_id(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        cid = resp.headers.get(HEADER_NAME)
        assert cid is not None
        uuid.UUID(cid)

    def test_caller_provided_id_reused(self, client):
        my_id = "req-12345-abc"
        resp = client.get("/health", headers={HEADER_NAME: my_id})
        assert resp.status_code == 200
        assert resp.headers[HEADER_NAME] == my_id

    def test_different_requests_get_different_ids(self, client):
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.headers[HEADER_NAME] != r2.headers[HEADER_NAME]

    def test_correlation_id_on_post(self, client):
        resp = client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": "billing_dispute",
            "decision": "resolve",
            "resolution_time_ms": 100,
        })
        assert resp.status_code == 201
        cid = resp.headers.get(HEADER_NAME)
        assert cid is not None

    def test_correlation_id_on_error(self, client):
        resp = client.post("/events", json={
            "case_id": "",
            "case_category": "billing",
            "decision": "resolve",
            "resolution_time_ms": 100,
        })
        assert resp.status_code == 422
        cid = resp.headers.get(HEADER_NAME)
        assert cid is not None

    def test_get_correlation_id_default(self):
        assert get_correlation_id() == ""
