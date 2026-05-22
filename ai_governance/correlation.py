"""Correlation ID middleware for request tracing.

Assigns a unique ID to every request. If the caller provides
X-Correlation-ID, it's reused; otherwise a new UUID is generated.
The ID is returned in the response header and available via
get_correlation_id() during request processing.
"""

from __future__ import annotations

import contextvars
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

HEADER_NAME = "X-Correlation-ID"

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def get_correlation_id() -> str:
    return _correlation_id.get()


class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        cid = request.headers.get(HEADER_NAME) or str(uuid.uuid4())
        token = _correlation_id.set(cid)
        try:
            response = await call_next(request)
            response.headers[HEADER_NAME] = cid
            return response
        finally:
            _correlation_id.reset(token)
