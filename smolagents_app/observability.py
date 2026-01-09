"""Langfuse/OpenTelemetry setup for smolagents."""

import base64
import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

logger = logging.getLogger(__name__)


def setup_langfuse_observability() -> Optional[trace.Tracer]:
    """Set up Langfuse observability with OpenTelemetry."""
    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not langfuse_public_key or not langfuse_secret_key:
        logger.warning("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not found. Observability will be limited.")
        return None

    os.environ["LANGFUSE_HOST"] = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    langfuse_auth = base64.b64encode(
        f"{langfuse_public_key}:{langfuse_secret_key}".encode()
    ).decode()

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_HOST") + "/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(trace_provider)

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    return trace.get_tracer(__name__)
