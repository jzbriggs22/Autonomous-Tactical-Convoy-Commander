"""API key authentication and rate limiting middleware.

Keys are loaded from GOVERNANCE_API_KEYS env var (comma-separated key:role pairs)
or passed programmatically via configure(). Roles: "admin" (full access) or
"read" (GET-only + /events for ingestion).

Rate limiting uses a per-key token bucket with configurable burst and refill rate.
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import json as _json

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response


@dataclass
class APIKey:
    key: str
    role: str  # "admin" or "read"


@dataclass
class TokenBucket:
    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_INGESTION_PATHS = {"/events", "/events/structured", "/events/batch"}
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

_keys: dict[str, APIKey] = {}
_buckets: dict[str, TokenBucket] = {}
_lock = threading.Lock()
_enabled = False
_burst = 60
_refill_rate = 10.0  # tokens/sec


def configure(
    keys: list[APIKey] | None = None,
    *,
    enabled: bool = True,
    burst: int = 60,
    refill_rate: float = 10.0,
) -> None:
    global _keys, _buckets, _enabled, _burst, _refill_rate
    _enabled = enabled
    _burst = burst
    _refill_rate = refill_rate
    with _lock:
        _keys.clear()
        _buckets.clear()
        if keys:
            for k in keys:
                _keys[k.key] = k
                _buckets[k.key] = TokenBucket(
                    capacity=burst, tokens=burst, refill_rate=refill_rate,
                )


def configure_from_env() -> None:
    raw = os.environ.get("GOVERNANCE_API_KEYS", "")
    if not raw.strip():
        configure(enabled=False)
        return
    keys = []
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        key, role = pair.split(":", 1)
        if role not in ("admin", "read"):
            continue
        keys.append(APIKey(key=key.strip(), role=role.strip()))
    configure(keys, enabled=bool(keys))


def reset() -> None:
    global _enabled
    _enabled = False
    with _lock:
        _keys.clear()
        _buckets.clear()


def _get_bucket(key_str: str) -> TokenBucket:
    with _lock:
        if key_str not in _buckets:
            _buckets[key_str] = TokenBucket(
                capacity=_burst, tokens=_burst, refill_rate=_refill_rate,
            )
        return _buckets[key_str]


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if not _enabled:
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key. Provide X-API-Key header or api_key query param."},
            )

        with _lock:
            key_obj = _keys.get(api_key)
        if key_obj is None:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        if request.method in _WRITE_METHODS and path not in _INGESTION_PATHS:
            if key_obj.role != "admin":
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Admin role required for this operation."},
                )

        bucket = _get_bucket(api_key)
        if not bucket.consume():
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
            )

        return await call_next(request)
