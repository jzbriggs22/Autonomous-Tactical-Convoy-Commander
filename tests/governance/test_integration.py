"""End-to-end integration test: full lifecycle through the HTTP API.

Exercises the complete happy path → drift → rollback → resume cycle:
1. Ingest warmup events via API
2. Compute baseline
3. Verify agent is safe
4. Ingest drifted events
5. Run drift detection → alerts fire
6. Verify agent is unsafe (rollback triggered)
7. PM resolves rollback
8. Verify agent is safe again
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.config import (
    AlertSeverity,
    DriftThreshold,
    GovernanceConfig,
    MetricDirection,
    RollbackCondition,
)
from ai_governance.storage import GovernanceDB
from ai_governance.webhooks import WebhookDispatcher, WebhookTarget


class _WebhookCapture(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _WebhookCapture.received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def full_stack():
    """Fully wired stack: tight config + webhook capture + test client."""
    _WebhookCapture.received = []
    server = HTTPServer(("127.0.0.1", 0), _WebhookCapture)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cfg = GovernanceConfig(
        agent_id="integration-agent",
        min_baseline_events=10,
        recent_window_size=30,
        high_risk_patterns=GovernanceConfig.default_customer_service().high_risk_patterns,
        drift_thresholds=[
            DriftThreshold(
                name="fraud_esc_drop",
                category="fraud_claim",
                metric="escalation_rate",
                max_delta=0.15,
                direction=MetricDirection.DECREASE,
                min_baseline_samples=10,
                recent_window=15,
                severity=AlertSeverity.CRITICAL,
            ),
        ],
        rollback_conditions=[
            RollbackCondition(
                name="fraud_danger",
                description="Fraud cases not being escalated",
                expression="fraud_claim_escalation_rate < 0.50 and fraud_claim_count >= 8",
                cooldown_seconds=60,
            ),
        ],
    )
    db = GovernanceDB(":memory:")
    webhooks = WebhookDispatcher([
        WebhookTarget(url=f"http://127.0.0.1:{port}/webhook"),
    ])
    init_services(cfg, db, webhooks=webhooks)
    client = TestClient(app)

    yield client, _WebhookCapture.received
    server.shutdown()


def _inject_events(client, n, category, decision, hours_ago_start):
    now = datetime.now(timezone.utc)
    for i in range(n):
        ts = now - timedelta(hours=hours_ago_start - (i * hours_ago_start / n))
        client.post("/events", json={
            "case_id": str(uuid.uuid4()),
            "case_category": category,
            "decision": decision,
            "resolution_time_ms": 500,
            "timestamp": ts.isoformat(),
        })


class TestFullLifecycle:
    def test_warmup_to_drift_to_rollback_to_resume(self, full_stack):
        client, webhook_log = full_stack

        # ── Step 1: Warmup — agent properly escalates fraud ──────────────
        _inject_events(client, 15, "fraud_claim", "escalate", hours_ago_start=48)
        _inject_events(client, 10, "billing_dispute", "resolve", hours_ago_start=48)

        status = client.get("/status").json()
        assert status["is_safe"] is True

        # ── Step 2: Compute baseline ─────────────────────────────────────
        baseline = client.post("/baseline/compute", json={}).json()
        assert "fraud_claim" in baseline["categories_computed"]

        # ── Step 3: Dashboard shows green ────────────────────────────────
        dash = client.get("/dashboard").json()
        assert dash["is_safe"] is True
        assert dash["governance_signals"]["overall_drift_score"] == 0.0
        assert dash["governance_signals"]["active_violations"] == 0

        fraud_row = next(
            (r for r in dash["category_breakdown"] if r["category"] == "fraud_claim"),
            None,
        )
        assert fraud_row is not None
        assert fraud_row["severity"] == "ok"

        # ── Step 4: Drifted events — fraud stops being escalated ─────────
        _inject_events(client, 20, "fraud_claim", "resolve", hours_ago_start=1)

        # ── Step 5: Drift detection fires alerts ─────────────────────────
        drift = client.get("/drift").json()
        assert drift["alerts_fired"] >= 1

        # Wait briefly for async webhook delivery
        import time
        time.sleep(0.3)
        assert len(webhook_log) >= 1
        wh = webhook_log[0]
        assert wh["agent_id"] == "integration-agent"
        assert wh["severity"] in ("critical", "rollback")

        # ── Step 6: Agent is now unsafe ──────────────────────────────────
        status = client.get("/status").json()
        assert status["is_safe"] is False

        rollbacks = client.get("/rollbacks").json()
        assert len(rollbacks) >= 1
        rb = rollbacks[0]
        assert rb["resolved"] is False
        assert rb["trigger_rule"] == "fraud_danger"

        # Dashboard reflects the danger state
        dash = client.get("/dashboard").json()
        assert dash["is_safe"] is False
        assert dash["governance_signals"]["rollback_events"] >= 1

        # Alerts are visible
        alerts = client.get("/alerts").json()
        assert len(alerts) >= 1

        # ── Step 7: PM resolves the rollback ─────────────────────────────
        rb_id = rb["rollback_id"]
        resolve = client.post(
            f"/rollbacks/{rb_id}/resolve",
            json={"resolved_by": "pm@company.com"},
        ).json()
        assert resolve["status"] == "resolved"

        # ── Step 8: Agent is safe again ──────────────────────────────────
        status = client.get("/status").json()
        assert status["is_safe"] is True

        # Rollback record now shows resolved state
        rollbacks = client.get("/rollbacks").json()
        resolved = next(r for r in rollbacks if r["rollback_id"] == rb_id)
        assert resolved["resolved"] is True
        assert resolved["resolved_by"] == "pm@company.com"
        assert resolved["resolved_at"] is not None

        # ── Step 9: Audit trail is intact ────────────────────────────────
        audit = client.get("/audit?limit=200").json()
        actions = [e["action"] for e in audit]
        assert "decision.ingested" in actions
        assert "baseline.computed" in actions
        assert "drift.detected" in actions
        assert "rollback.resolved" in actions

        verify = client.get("/audit/verify").json()
        assert verify["chain_valid"] is True

    def test_batch_ingest_and_ground_truth_flow(self, full_stack):
        client, _ = full_stack

        # Batch ingest
        events = [
            {
                "case_id": f"batch-{i}",
                "case_category": "fraud_claim",
                "decision": "escalate",
                "resolution_time_ms": 300,
            }
            for i in range(5)
        ]
        resp = client.post("/events/batch", json=events)
        assert resp.status_code == 201
        results = resp.json()
        assert len(results) == 5

        # Add ground truth to first event
        event_id = results[0]["event_id"]
        gt_resp = client.post(
            f"/events/{event_id}/ground-truth",
            json={"ground_truth": "escalate"},
        )
        assert gt_resp.status_code == 200

    def test_acknowledge_alert_flow(self, full_stack):
        client, _ = full_stack

        _inject_events(client, 15, "fraud_claim", "escalate", hours_ago_start=48)
        client.post("/baseline/compute", json={})
        _inject_events(client, 20, "fraud_claim", "resolve", hours_ago_start=1)
        client.get("/drift")

        alerts = client.get("/alerts").json()
        assert len(alerts) >= 1
        alert_id = alerts[0]["alert_id"]
        assert alerts[0]["acknowledged"] is False

        client.post(f"/alerts/{alert_id}/acknowledge")
        alerts_after = client.get("/alerts").json()
        acked = next(a for a in alerts_after if a["alert_id"] == alert_id)
        assert acked["acknowledged"] is True

    def test_resolve_all_rollbacks(self, full_stack):
        client, _ = full_stack

        _inject_events(client, 10, "fraud_claim", "resolve", hours_ago_start=1)
        client.get("/drift")  # triggers rollback

        assert client.get("/status").json()["is_safe"] is False

        resp = client.post("/rollbacks/resolve-all", json={"resolved_by": "oncall"})
        assert resp.json()["count"] >= 1
        assert client.get("/status").json()["is_safe"] is True
