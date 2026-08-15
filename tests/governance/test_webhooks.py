"""Tests for webhook alert dispatcher."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ai_governance.webhooks import WebhookDelivery, WebhookDispatcher, WebhookTarget


class _CaptureHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that captures POST bodies for assertions."""

    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _CaptureHandler.received.append({
            "path": self.path,
            "body": json.loads(body),
            "headers": dict(self.headers),
        })
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def capture_server():
    """Start a local HTTP server that captures POST bodies, yield its URL."""
    _CaptureHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestWebhookDispatcher:
    def test_dispatch_sends_to_target(self, capture_server):
        dispatcher = WebhookDispatcher([
            WebhookTarget(url=f"{capture_server}/alerts"),
        ])
        results = dispatcher.dispatch({
            "severity": "critical",
            "rule_name": "fraud_drop",
            "message": "test alert",
        })
        assert len(results) == 1
        assert results[0].status == "sent"
        assert results[0].status_code == 200
        assert len(_CaptureHandler.received) == 1
        assert _CaptureHandler.received[0]["body"]["rule_name"] == "fraud_drop"

    def test_dispatch_respects_severity_filter(self, capture_server):
        dispatcher = WebhookDispatcher([
            WebhookTarget(
                url=f"{capture_server}/critical-only",
                severity_filter={"critical", "rollback"},
            ),
        ])
        # info severity → should be filtered out
        results = dispatcher.dispatch({"severity": "info", "message": "low prio"})
        assert len(results) == 0
        assert len(_CaptureHandler.received) == 0

        # critical → should be sent
        results = dispatcher.dispatch({"severity": "critical", "message": "high prio"})
        assert len(results) == 1
        assert results[0].status == "sent"

    def test_dispatch_to_unreachable_url_returns_failed(self):
        dispatcher = WebhookDispatcher([
            WebhookTarget(
                url="http://127.0.0.1:1/unreachable",
                timeout_seconds=0.5,
            ),
        ])
        results = dispatcher.dispatch({"severity": "warn", "message": "test"})
        assert len(results) == 1
        assert results[0].status == "failed"
        assert results[0].error is not None

    def test_dispatch_to_multiple_targets(self, capture_server):
        dispatcher = WebhookDispatcher([
            WebhookTarget(url=f"{capture_server}/target1"),
            WebhookTarget(url=f"{capture_server}/target2"),
        ])
        results = dispatcher.dispatch({"severity": "warn", "message": "multi"})
        assert len(results) == 2
        assert all(r.status == "sent" for r in results)
        assert len(_CaptureHandler.received) == 2

    def test_delivery_log_accumulates(self, capture_server):
        dispatcher = WebhookDispatcher([
            WebhookTarget(url=f"{capture_server}/log-test"),
        ])
        dispatcher.dispatch({"severity": "info", "message": "first"})
        dispatcher.dispatch({"severity": "warn", "message": "second"})
        log = dispatcher.delivery_log
        assert len(log) == 2
        assert log[0].status == "sent"

    def test_custom_headers_sent(self, capture_server):
        dispatcher = WebhookDispatcher([
            WebhookTarget(
                url=f"{capture_server}/headers",
                headers={"X-Governance-Token": "secret-123"},
            ),
        ])
        dispatcher.dispatch({"severity": "info", "message": "with-header"})
        assert len(_CaptureHandler.received) == 1
        assert _CaptureHandler.received[0]["headers"]["X-Governance-Token"] == "secret-123"

    def test_add_target_dynamically(self, capture_server):
        dispatcher = WebhookDispatcher()
        assert len(dispatcher.targets) == 0
        dispatcher.add_target(WebhookTarget(url=f"{capture_server}/dynamic"))
        assert len(dispatcher.targets) == 1
        results = dispatcher.dispatch({"severity": "info", "message": "dynamic"})
        assert len(results) == 1
        assert results[0].status == "sent"

    def test_dispatch_async_does_not_block(self, capture_server):
        dispatcher = WebhookDispatcher([
            WebhookTarget(url=f"{capture_server}/async"),
        ])
        dispatcher.dispatch_async({"severity": "warn", "message": "async"})
        import time
        time.sleep(0.3)
        assert len(_CaptureHandler.received) == 1

    def test_queue_backpressure_drops_when_full(self):
        dispatcher = WebhookDispatcher(queue_size=2, worker_count=0)
        assert dispatcher.dispatch_async({"severity": "info", "message": "1"}) is True
        assert dispatcher.dispatch_async({"severity": "info", "message": "2"}) is True
        assert dispatcher.dispatch_async({"severity": "info", "message": "3"}) is False
        assert dispatcher.dropped_count == 1

    def test_queue_depth_tracks_pending(self):
        dispatcher = WebhookDispatcher(queue_size=10, worker_count=0)
        assert dispatcher.queue_depth == 0
        dispatcher.dispatch_async({"severity": "info", "message": "pending"})
        assert dispatcher.queue_depth == 1

    def test_worker_processes_queue(self, capture_server):
        dispatcher = WebhookDispatcher(
            [WebhookTarget(url=f"{capture_server}/worker")],
            queue_size=10,
            worker_count=1,
        )
        dispatcher.dispatch_async({"severity": "info", "message": "queued"})
        import time
        time.sleep(0.5)
        assert len(_CaptureHandler.received) == 1
        assert _CaptureHandler.received[0]["body"]["message"] == "queued"

    def test_shutdown_drains_queue(self, capture_server):
        dispatcher = WebhookDispatcher(
            [WebhookTarget(url=f"{capture_server}/drain")],
            queue_size=10,
            worker_count=1,
        )
        for i in range(3):
            dispatcher.dispatch_async({"severity": "info", "message": f"drain-{i}"})
        dispatcher.shutdown(timeout=5.0)
        assert len(_CaptureHandler.received) == 3


class TestWebhookRetry:
    def test_retry_on_failure_records_attempts(self):
        dispatcher = WebhookDispatcher([
            WebhookTarget(
                url="http://127.0.0.1:1/unreachable",
                timeout_seconds=0.2,
                max_retries=3,
                backoff_base=0.05,
            ),
        ])
        results = dispatcher.dispatch({"severity": "critical", "message": "retry-test"})
        assert len(results) == 1
        assert results[0].status == "failed"
        assert results[0].attempts == 3
        assert "3 attempts" in results[0].error

    def test_no_retry_when_max_retries_is_1(self):
        dispatcher = WebhookDispatcher([
            WebhookTarget(
                url="http://127.0.0.1:1/unreachable",
                timeout_seconds=0.2,
                max_retries=1,
                backoff_base=0.05,
            ),
        ])
        results = dispatcher.dispatch({"severity": "warn", "message": "once"})
        assert results[0].attempts == 1
        assert results[0].status == "failed"

    def test_successful_on_first_attempt_records_one_attempt(self, capture_server):
        dispatcher = WebhookDispatcher([
            WebhookTarget(
                url=f"{capture_server}/retry-ok",
                max_retries=3,
                backoff_base=0.05,
            ),
        ])
        results = dispatcher.dispatch({"severity": "warn", "message": "ok"})
        assert results[0].status == "sent"
        assert results[0].attempts == 1

    def test_delivery_log_records_retry_outcome(self):
        dispatcher = WebhookDispatcher([
            WebhookTarget(
                url="http://127.0.0.1:1/unreachable",
                timeout_seconds=0.2,
                max_retries=2,
                backoff_base=0.05,
            ),
        ])
        dispatcher.dispatch({"severity": "critical", "message": "log-test"})
        log = dispatcher.delivery_log
        assert len(log) == 1
        assert log[0].attempts == 2
        assert log[0].status == "failed"
