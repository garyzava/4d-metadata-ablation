"""OpenTelemetry tracing utilities for Gyzasql."""

from __future__ import annotations

import functools
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter

_TRACER_NAME = "gyzasql"
_initialized = False


def init_tracing(exporter: SpanExporter | None = None) -> TracerProvider:
    """Initialize OpenTelemetry tracing with the given exporter.

    Defaults to ConsoleSpanExporter if GYZASQL_TRACE=1 and no exporter given.
    """
    global _initialized  # noqa: PLW0603
    provider = TracerProvider()
    if exporter:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    elif os.environ.get("GYZASQL_TRACE") == "1":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True
    return provider


def get_tracer() -> trace.Tracer:
    """Return a named tracer for Gyzasql instrumentation."""
    return trace.get_tracer(_TRACER_NAME)


def traced(name: str | None = None, attributes: dict[str, Any] | None = None):
    """Decorator that wraps a function in an OpenTelemetry span.

    Args:
        name: Span name. Defaults to the function name.
        attributes: Static span attributes to set.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(exc))
                    raise

        return wrapper

    return decorator
