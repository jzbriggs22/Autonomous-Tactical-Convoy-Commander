"""OpenTelemetry tracing for governance operations.

Sets up a tracer with an in-memory or OTLP exporter. The in-memory exporter
is used by default (and in tests) so no external collector is required.

Wire into FastAPI via the ASGI middleware or call span() directly:

    with tracing.span("drift.detect", agent_id="cs-agent-v1") as s:
        result = detector.detect()
        s.set_attribute("violations", len(result.violations))
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, StatusCode

_SERVICE_NAME = "ai-governance"

# In-memory exporter available for tests
_memory_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_memory_exporter))

_otlp_configured = False


def configure_otlp(endpoint: str | None = None) -> None:
    """Add an OTLP gRPC exporter (e.g. Jaeger, Grafana Tempo).

    Call once at startup if OTEL_EXPORTER_OTLP_ENDPOINT is set.
    Falls back to in-memory silently if opentelemetry-exporter-otlp is not installed.
    """
    global _otlp_configured
    ep = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not ep:
        return
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        exporter = OTLPSpanExporter(endpoint=ep)
        _provider.add_span_processor(BatchSpanProcessor(exporter))
        _otlp_configured = True
    except ImportError:
        pass


def _tracer() -> trace.Tracer:
    trace.set_tracer_provider(_provider)
    return trace.get_tracer(_SERVICE_NAME)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Context manager that creates a named span with optional attributes."""
    with _tracer().start_as_current_span(name) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v)
        try:
            yield s
        except Exception as exc:
            s.set_status(StatusCode.ERROR, str(exc))
            raise


def get_finished_spans() -> list:
    """Return all finished spans recorded by the in-memory exporter."""
    return list(_memory_exporter.get_finished_spans())


def clear_spans() -> None:
    """Clear in-memory spans — used between tests."""
    _memory_exporter.clear()


def span_names() -> list[str]:
    """Convenience: return just the names of finished spans."""
    return [s.name for s in get_finished_spans()]
