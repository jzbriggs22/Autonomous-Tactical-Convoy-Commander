"""Webhook alert dispatcher.

Sends alert payloads to configured HTTP endpoints on each alert fire.
Retries with exponential backoff on transient failures.
Permanent failures are logged and never block the alert pipeline.
"""

from __future__ import annotations

import json
import logging
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

    def __init__(self, targets: list[WebhookTarget] = None) -> None:
        self._targets = list(targets or [])
        self._delivery_log: list[WebhookDelivery] = []
        self._lock = threading.Lock()

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

    def dispatch_async(self, alert_payload: dict) -> None:
        """Non-blocking dispatch in a daemon thread."""
        t = threading.Thread(target=self.dispatch, args=(alert_payload,), daemon=True)
        t.start()

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
