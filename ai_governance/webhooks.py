"""Webhook alert dispatcher.

Sends alert payloads to configured HTTP endpoints on each alert fire.
Retries with exponential backoff on transient failures.
Permanent failures are logged and never block the alert pipeline.

Supports HMAC-SHA256 request signing when a target has a secret configured.
"""

from __future__ import annotations

import collections
import hashlib
import hmac
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 1.0  # seconds


@dataclass
class WebhookTarget:
    url: str
    headers: dict = field(default_factory=dict)
    timeout_seconds: float = 5.0
    severity_filter: Optional[set[str]] = None  # None = all severities
    max_retries: int = _DEFAULT_MAX_RETRIES
    backoff_base: float = _DEFAULT_BACKOFF_BASE
    signing_secret: Optional[str] = None


@dataclass
class WebhookDelivery:
    url: str
    status: str  # "sent" | "failed"
    status_code: Optional[int] = None
    error: Optional[str] = None
    attempts: int = 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WebhookDispatcher:
    """Fire-and-forget webhook delivery for governance alerts.

    Dispatches to all registered targets. Retries failed deliveries
    with exponential backoff (1s, 2s, 4s by default).
    """

    _MAX_DELIVERY_LOG = 10000

    def __init__(
        self,
        targets: list[WebhookTarget] = None,
        *,
        queue_size: int = 1000,
        worker_count: int = 2,
        max_log_size: int = _MAX_DELIVERY_LOG,
    ) -> None:
        self._targets = list(targets or [])
        self._delivery_log: collections.deque[WebhookDelivery] = collections.deque(maxlen=max_log_size)
        self._lock = threading.Lock()
        self._queue: queue.Queue[Optional[dict]] = queue.Queue(maxsize=queue_size)
        self._queue_size = queue_size
        self._dropped = 0
        self._workers: list[threading.Thread] = []
        for i in range(worker_count):
            w = threading.Thread(target=self._worker_loop, daemon=True, name=f"webhook-worker-{i}")
            w.start()
            self._workers.append(w)

    def add_target(self, target: WebhookTarget) -> None:
        self._targets.append(target)

    @property
    def targets(self) -> list[WebhookTarget]:
        return list(self._targets)

    @property
    def delivery_log(self) -> list[WebhookDelivery]:
        with self._lock:
            return list(self._delivery_log)

    def dispatch(self, alert_payload: dict) -> list[WebhookDelivery]:
        """Send alert_payload to all matching targets. Returns delivery results."""
        severity = alert_payload.get("severity", "")
        results: list[WebhookDelivery] = []

        for target in self._targets:
            if target.severity_filter and severity not in target.severity_filter:
                continue
            delivery = self._send_with_retry(target, alert_payload)
            results.append(delivery)
            with self._lock:
                self._delivery_log.append(delivery)

        return results

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def dispatch_async(self, alert_payload: dict) -> bool:
        """Enqueue payload for async delivery. Returns False if queue is full (backpressure)."""
        try:
            self._queue.put_nowait(alert_payload)
            return True
        except queue.Full:
            self._dropped += 1
            logger.warning("Webhook queue full (max %d), dropping payload", self._queue_size)
            return False

    def shutdown(self, timeout: float = 5.0) -> None:
        """Signal workers to stop and wait for drain."""
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        for w in self._workers:
            w.join(timeout=timeout)

    def _worker_loop(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                break
            try:
                self.dispatch(payload)
            except Exception:
                logger.exception("Webhook worker error")
            finally:
                self._queue.task_done()

    def _send_with_retry(self, target: WebhookTarget, payload: dict) -> WebhookDelivery:
        last_error = None
        for attempt in range(1, target.max_retries + 1):
            delivery = self._send(target, payload)
            if delivery.status == "sent":
                delivery.attempts = attempt
                return delivery
            last_error = delivery.error
            if attempt < target.max_retries:
                backoff = target.backoff_base * (2 ** (attempt - 1))
                time.sleep(backoff)

        return WebhookDelivery(
            url=target.url,
            status="failed",
            error=f"Failed after {target.max_retries} attempts: {last_error}",
            attempts=target.max_retries,
        )

    def _send(self, target: WebhookTarget, payload: dict) -> WebhookDelivery:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **target.headers}
        if target.signing_secret:
            sig = compute_signature(body, target.signing_secret)
            headers["X-Governance-Signature"] = sig
        req = Request(target.url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=target.timeout_seconds) as resp:
                return WebhookDelivery(
                    url=target.url,
                    status="sent",
                    status_code=resp.status,
                )
        except URLError as exc:
            logger.warning("Webhook delivery to %s failed: %s", target.url, exc)
            return WebhookDelivery(
                url=target.url,
                status="failed",
                error=str(exc),
            )
        except Exception as exc:
            logger.warning("Webhook delivery to %s failed: %s", target.url, exc)
            return WebhookDelivery(
                url=target.url,
                status="failed",
                error=str(exc),
            )


def compute_signature(body: bytes, secret: str) -> str:
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def verify_signature(body: bytes, secret: str, signature: str) -> bool:
    expected = compute_signature(body, secret)
    return hmac.compare_digest(expected, signature)
