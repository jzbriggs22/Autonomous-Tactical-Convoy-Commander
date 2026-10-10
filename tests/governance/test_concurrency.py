"""Tests for thread safety and concurrent access patterns."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

import pytest

from ai_governance.config import GovernanceConfig
from ai_governance.ingestion import IngestRequest, IngestionLayer
from ai_governance.storage import GovernanceDB
from ai_governance.audit import AuditLog
from ai_governance.drift import DriftDetector


class TestConcurrentIngestion:
    def test_concurrent_writes_no_lost_events(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)
        errors = []
        n_per_thread = 50
        n_threads = 4

        def worker(thread_id: int):
            for i in range(n_per_thread):
                try:
                    ing.ingest(IngestRequest(
                        case_id=f"t{thread_id}-{i}",
                        case_category="billing_dispute",
                        decision="resolve",
                        resolution_time_ms=100,
                    ))
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        total = db.count_decisions(cfg.agent_id)
        assert total == n_per_thread * n_threads

    def test_concurrent_read_write_no_crash(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)
        det = DriftDetector(cfg, db)
        errors = []

        def writer():
            for i in range(50):
                try:
                    ing.ingest(IngestRequest(
                        case_id=f"w-{i}",
                        case_category="fraud_claim",
                        decision="escalate",
                        resolution_time_ms=100,
                    ))
                except Exception as exc:
                    errors.append(exc)

        def reader():
            for _ in range(20):
                try:
                    det.detect()
                except Exception as exc:
                    errors.append(exc)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join(timeout=30)
        t_read.join(timeout=30)
        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_audit_chain_integrity(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        audit = AuditLog(db)
        errors = []
        n_per_thread = 20
        n_threads = 3

        def writer(thread_id: int):
            for i in range(n_per_thread):
                try:
                    audit.append(
                        cfg.agent_id, "decision.ingested", "system", "event",
                        detail={"thread": thread_id, "index": i},
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        entries = audit.get_entries(cfg.agent_id, limit=200)
        assert len(entries) == n_per_thread * n_threads

        valid, broken = audit.verify_chain(cfg.agent_id)
        assert valid is True, f"Chain broken at seq {broken}"

    def test_concurrent_exports_no_crash(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        ing = IngestionLayer(cfg, db)
        errors = []

        for i in range(30):
            ing.ingest(IngestRequest(
                case_id=f"export-{i}",
                case_category="returns",
                decision="resolve",
                resolution_time_ms=100,
            ))

        def exporter():
            for _ in range(20):
                try:
                    recs = db.export_decisions(cfg.agent_id, limit=100)
                    assert len(recs) > 0
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=exporter) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert len(errors) == 0


class TestConcurrentAPIAccess:
    def test_concurrent_api_requests(self):
        from fastapi.testclient import TestClient
        from ai_governance.api import app, init_services
        from ai_governance.auth import reset as reset_auth

        reset_auth()
        cfg = GovernanceConfig.default_customer_service()
        cfg.min_baseline_events = 5
        cfg.recent_window_size = 20
        for thr in cfg.drift_thresholds:
            thr.min_baseline_samples = 5
            thr.recent_window = 10
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        client = TestClient(app)
        errors = []

        def post_events():
            for i in range(20):
                try:
                    resp = client.post("/events", json={
                        "case_id": str(uuid.uuid4()),
                        "case_category": "billing_dispute",
                        "decision": "resolve",
                        "resolution_time_ms": 100,
                    })
                    assert resp.status_code == 201
                except Exception as exc:
                    errors.append(exc)

        def read_dashboard():
            for _ in range(10):
                try:
                    resp = client.get("/dashboard")
                    assert resp.status_code == 200
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=post_events),
            threading.Thread(target=post_events),
            threading.Thread(target=read_dashboard),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert len(errors) == 0
        reset_auth()
