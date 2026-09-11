"""Sampled OpenTelemetry spans export off the token path without recording prompt content."""

import json
import math
import os
import threading
import time
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, Status, Tracer, TraceState

_COMPONENTS = frozenset({"gateway", "route", "engine"})
_NAMES = frozenset({"finserve.inference", "finserve.route", "finserve.engine"})
_COUNTS = frozenset(
    {"gen_ai.usage.output_tokens", "finserve.prompt_characters", "finserve.max_tokens"}
)
_OUTCOMES = frozenset({"success", "failed", "cancelled", "timeout", "engine_failed"})


def _context(value: SpanContext | None) -> SpanContext | None:
    """Preserve causal identity while removing arbitrary vendor trace-state values."""
    if value is None:
        return None
    return SpanContext(
        value.trace_id, value.span_id, value.is_remote, value.trace_flags, TraceState()
    )


def sanitized_span(span: ReadableSpan, component: str = "gateway") -> ReadableSpan:
    """Rebuild all exportable fields: events, links, resource and scope can contain user data."""
    attributes: dict[str, str | int] = {}
    for key, value in (span.attributes or {}).items():
        if key in _COUNTS and type(value) is int and 0 <= value < 2**63:
            attributes[key] = value
        elif key == "finserve.outcome" and isinstance(value, str) and value in _OUTCOMES:
            attributes[key] = value
        elif (
            key == "gen_ai.request.model"
            and isinstance(value, str)
            and len(value) <= 256
            and all(32 <= ord(character) <= 126 for character in value)
        ):
            attributes[key] = value
    return ReadableSpan(
        name=span.name if span.name in _NAMES else "finserve.span",
        context=_context(span.context),
        parent=_context(span.parent),
        resource=Resource({"service.name": "finserve." + component}),
        attributes=attributes,
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=InstrumentationScope("finserve", "1.0"),
    )


class SanitizingSpanExporter(SpanExporter):
    """Apply the privacy boundary to every sink, including operator-provided exporters."""

    def __init__(self, exporter: SpanExporter, component: str) -> None:
        """Keep failure accounting local without recording exceptions or rejected values."""
        self.exporter = exporter
        self.component = component
        self.failures = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """A telemetry failure cannot propagate secret-bearing exception details into SDK logs."""
        try:
            result = self.exporter.export([sanitized_span(span, self.component) for span in spans])
        except Exception:
            result = SpanExportResult.FAILURE
        if result is SpanExportResult.FAILURE:
            self.failures += 1
        return result

    def shutdown(self) -> None:
        """The runtime owns the sink and drains it on its caller's shutdown thread."""
        try:
            self.exporter.shutdown()
        except Exception:
            self.failures += 1


def _otlp_exporter(endpoint: str, authorization: str | None, timeout: float) -> SpanExporter:
    """Use the official protobuf exporter with an explicitly constrained HTTP transport."""
    try:
        import requests
        from opentelemetry.exporter.otlp.proto.http import Compression
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceResponse,
        )
    except ImportError:
        raise RuntimeError("OTLP tracing requires the telemetry optional dependency") from None

    def acknowledged(response: requests.Response, deadline: float) -> bool:
        """Bound protobuf replies and detect rejected spans without exporting error text."""
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("encoded collector response")
        declared = response.headers.get("Content-Length")
        if declared is not None and not 0 <= int(declared) <= 16384:
            raise ValueError("collector response too large")
        body = bytearray()
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError
            # read1 performs one bounded body read, unlike fill-to-size read().
            chunk = response.raw.read1(16385 - len(body), decode_content=False)
            if time.monotonic() >= deadline:
                raise TimeoutError
            body.extend(chunk)
            if len(body) > 16384:
                raise ValueError("collector response too large")
            if not chunk:
                break
        acknowledgement = ExportTraceServiceResponse.FromString(bytes(body))
        return acknowledgement.partial_success.rejected_spans == 0

    class CollectorSession(requests.Session):
        """No ambient proxies, redirects, unbounded responses or remote error text in SDK logs."""

        def post(
            self, url: str | bytes, data: Any = None, json: Any = None, **kwargs: Any
        ) -> requests.Response:
            """Respect SDK timeout/retry accounting; native DNS is not interruptible."""
            if url != endpoint:
                raise requests.exceptions.Timeout("OTLP endpoint mismatch")
            budget = min(timeout, float(kwargs.get("timeout", timeout)))
            if not math.isfinite(budget) or budget <= 0:
                raise requests.exceptions.Timeout("OTLP export deadline exhausted")
            deadline = time.monotonic() + budget
            try:
                with super().request(
                    "POST",
                    endpoint,
                    data=data,
                    allow_redirects=False,
                    stream=True,
                    timeout=budget,
                    verify=True,
                    cert=None,
                ) as response:
                    # Non-success bodies are discarded; successful replies need a bounded ACK.
                    status = response.status_code
                    if 200 <= status < 300 and not acknowledged(response, deadline):
                        status = 400
            except Exception:
                raise requests.exceptions.Timeout("OTLP transport failed") from None
            safe = requests.Response()
            safe.status_code = 400 if 300 <= status < 400 else status
            safe.reason = "collector response"
            safe.raw = BytesIO(b"")
            return safe

    session = CollectorSession()
    session.trust_env = False
    headers = {"Content-Type": "application/x-protobuf", "Accept-Encoding": "identity"}
    if authorization:
        headers["Authorization"] = authorization
    try:
        return OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout,
            compression=Compression.NoCompression,
            session=session,
        )
    except Exception:
        session.close()
        raise RuntimeError("OTLP exporter initialization failed") from None


def _endpoint(value: str) -> str:
    """Collector locations are configuration only, with credentials carried separately."""
    try:
        parsed = urlsplit(value)
        valid = (
            0 < len(value) <= 2048
            and all(33 <= ord(character) <= 126 for character in value)
            and parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and "?" not in value
            and "#" not in value
            and "\\" not in value
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and (parsed.scheme == "https" or parsed.hostname in {"localhost", "127.0.0.1", "::1"})
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Invalid OTLP collector endpoint")
    return value


def from_env(*, component: str = "gateway") -> "TraceRuntime | None":
    """Unconfigured tracing has no file, thread, optional import or global-provider effects."""
    if component not in _COMPONENTS:
        raise ValueError("Invalid tracing component")
    path = os.getenv("FINSERVE_TRACE_PATH")
    endpoint = os.getenv("FINSERVE_OTLP_ENDPOINT")
    if not path and not endpoint:
        return None
    if path and endpoint:
        raise ValueError("Configure exactly one tracing output")
    try:
        ratio = float(os.getenv("FINSERVE_TRACE_SAMPLE_RATIO", "0.01"))
        timeout = float(os.getenv("FINSERVE_OTLP_TIMEOUT_SECONDS", "3"))
    except ValueError:
        raise ValueError("Invalid tracing numeric configuration") from None
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("sampling ratio must be finite and within [0,1]")
    if not math.isfinite(timeout) or not 0.1 <= timeout <= 10:
        raise ValueError("OTLP timeout must be finite and within [0.1,10] seconds")
    if endpoint:
        authorization = os.getenv("FINSERVE_OTLP_AUTHORIZATION")
        if authorization is not None and (
            not 0 < len(authorization) <= 4096
            or any(not 32 <= ord(character) <= 126 for character in authorization)
        ):
            raise ValueError("Invalid OTLP authorization")
        exporter = _otlp_exporter(_endpoint(endpoint), authorization, timeout)
    else:
        output = Path(path or "")
        if component != "gateway":
            output = output.with_name(
                f"{output.stem}.{component}.{os.getpid()}.{uuid4().hex}{output.suffix}"
            )
        exporter = JsonSpanExporter(output)
    return TraceRuntime(exporter, ratio, component=component)


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
        try:
            with self.lock:
                for original in spans:
                    span = sanitized_span(original)
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
                                    key: value for key, value in (span.attributes or {}).items()
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

    def __init__(
        self, exporter: SpanExporter, sample_ratio: float = 0.01, *, component: str = "gateway"
    ) -> None:
        """Bound the queue; the local file sink has no hard I/O/shutdown deadline."""
        if component not in _COMPONENTS:
            exporter.shutdown()
            raise ValueError("Invalid tracing component")
        if not math.isfinite(sample_ratio) or not 0 <= sample_ratio <= 1:
            exporter.shutdown()
            raise ValueError("sampling ratio must be finite and within [0,1]")
        self.provider = TracerProvider(
            resource=Resource({"service.name": "finserve." + component}),
            sampler=TraceIdRatioBased(sample_ratio),
        )
        self.exporter = SanitizingSpanExporter(exporter, component)
        self.provider.add_span_processor(
            BatchSpanProcessor(
                self.exporter,
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
