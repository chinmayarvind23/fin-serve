"""Sampled OpenTelemetry spans export off the token path without recording prompt content."""

import json
import math
import threading
from collections.abc import Sequence
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.trace import Tracer


class JsonSpanExporter(SpanExporter):
    """Local evidence exporter; the same SDK boundary accepts an OTLP exporter in deployments."""

    def __init__(self, path: Path) -> None:
        """Exclusive external output avoids overwriting trace evidence or checking it into Git."""
        resolved = path.resolve()
        repository = Path(__file__).resolve().parents[3]
        if resolved == repository or repository in resolved.parents:
            raise ValueError("trace evidence must be outside the source repository")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.stream = resolved.open("x", encoding="utf-8")
        self.lock = threading.Lock()
        self.failures = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Use a fixed metadata allowlist so future instrumentation cannot export user content."""
        allowed = {
            "finserve.outcome",
            "gen_ai.request.model",
            "gen_ai.usage.output_tokens",
            "finserve.prompt_characters",
            "finserve.max_tokens",
        }
        try:
            with self.lock:
                for span in spans:
                    context = span.context
                    if context is None:
                        continue
                    self.stream.write(
                        json.dumps(
                            {
                                "name": span.name
                                if span.name
                                in {"finserve.inference", "finserve.route", "finserve.engine"}
                                else "finserve.span",
                                "trace_id": f"{context.trace_id:032x}",
                                "span_id": f"{context.span_id:016x}",
                                "parent_span_id": f"{span.parent.span_id:016x}"
                                if span.parent
                                else None,
                                "start_time_ns": span.start_time,
                                "end_time_ns": span.end_time,
                                "status": span.status.status_code.name,
                                "attributes": {
                                    key: value
                                    for key, value in (span.attributes or {}).items()
                                    if key in allowed
                                },
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                self.stream.flush()
            return SpanExportResult.SUCCESS
        except (OSError, ValueError):
            self.failures += 1
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Serialize final close with any in-progress batch flush."""
        with self.lock:
            self.stream.close()


class TraceRuntime:
    """A private provider avoids process-global configuration collisions across apps/tests."""

    def __init__(self, exporter: SpanExporter, sample_ratio: float = 0.01) -> None:
        """Bound the queue; the local file sink has no hard I/O/shutdown deadline."""
        if not math.isfinite(sample_ratio) or not 0 <= sample_ratio <= 1:
            exporter.shutdown()
            raise ValueError("sampling ratio must be finite and within [0,1]")
        self.provider = TracerProvider(
            resource=Resource.create({"service.name": "finserve"}),
            sampler=TraceIdRatioBased(sample_ratio),
        )
        self.provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                max_queue_size=2048,
                max_export_batch_size=128,
                schedule_delay_millis=500,
                export_timeout_millis=3000,
            )
        )
        self.tracer: Tracer = self.provider.get_tracer("finserve.inference", "1.0")

    def close(self) -> None:
        """Lifespan shutdown drains queued spans outside the request event loop."""
        self.provider.shutdown()
