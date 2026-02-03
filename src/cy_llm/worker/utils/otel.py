from __future__ import annotations

import os


def init_tracing(service_name: str) -> None:
	endpoint = os.getenv("CY_LLM_OTEL_ENDPOINT", "")
	if not endpoint:
		return

	try:
		from opentelemetry import trace  # type: ignore
		from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # type: ignore
		from opentelemetry.sdk.trace import TracerProvider  # type: ignore
		from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
		from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore
	except ImportError:
		return

	resource = Resource(attributes={SERVICE_NAME: service_name})
	provider = TracerProvider(resource=resource)
	span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
	provider.add_span_processor(BatchSpanProcessor(span_exporter))
	trace.set_tracer_provider(provider)
