"""Decode real OTLP HTTP traffic and probe privacy and nonblocking failure boundaries."""

import asyncio
import builtins
import importlib
import json
import threading
import time
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, SpanContext, Status, StatusCode, TraceFlags, TraceState

from finserve.telemetry.tracing import (
    JsonSpanExporter,
    SanitizingSpanExporter,
    TraceRuntime,
    from_env,
)


@pytest.fixture(autouse=True)
def isolated_trace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never inherit a real collector endpoint or export an operator credential."""
    for name in (
        "FINSERVE_TRACE_PATH",
        "FINSERVE_OTLP_ENDPOINT",
        "FINSERVE_OTLP_AUTHORIZATION",
        "FINSERVE_TRACE_SAMPLE_RATIO",
        "FINSERVE_OTLP_TIMEOUT_SECONDS",
        "FINSERVE_OTLP_PROTOCOL",
    ):
        monkeypatch.delenv(name, raising=False)


class Collector:
    """Retain exact local protobuf requests without a hosted telemetry dependency."""

    def __init__(self) -> None:
        """A release event makes blocked response-header tests deterministic."""
        self.requests: list[tuple[str, dict[str, str], bytes]] = []
        self.status = 200
        self.hold = False
        self.body = b""
        self.declared_size: int | None = None
        self.omit_length = False
        self.trickle = False
        self.encoding: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.url = ""


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"{}",
        b"[]",
        b'{"name":"otel-ingestion-job"}',
        b'{"id":"1","id":"2","name":"otel-ingestion-job","data":{}}',
        b'{"id":"1","name":"otel-ingestion-job","data":{"name":"otel-ingestion-job"},"partialSuccess":{"rejectedSpans":1}}',
    ],
    ids=["empty", "empty-object", "array", "missing-job", "duplicate", "partial-rejection"],
)
async def test_langfuse_ack_rejects_ambiguous_success(
    collector: "Collector",
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    body: bytes,
) -> None:
    """Langfuse's explicit protocol cannot turn an arbitrary JSON HTTP200 into acceptance."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    collector.body = body
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", collector.url)
    monkeypatch.setenv("FINSERVE_OTLP_PROTOCOL", "langfuse-v4")
    monkeypatch.setenv("FINSERVE_OTLP_TIMEOUT_SECONDS", "0.1")
    runtime = from_env()
    assert runtime is not None
    try:
        assert (
            await asyncio.to_thread(runtime.exporter.export, [unsafe_span()])
            is SpanExportResult.FAILURE
        )
        assert runtime.exporter.failures == 1
    finally:
        await asyncio.to_thread(runtime.close)
    assert "rejectedSpans" not in caplog.text


@pytest.fixture
def collector() -> Iterator[Collector]:
    """Bind an ephemeral loopback collector and always drain server-owned threads."""
    capture = Collector()

    class Handler(BaseHTTPRequestHandler):
        """Exercise the official exporter's HTTP transport, including adversarial replies."""

        def do_POST(self) -> None:
            """Keep response body/reason deliberately private to catch SDK log leakage."""
            body = self.rfile.read(int(self.headers["Content-Length"]))
            capture.requests.append((self.path, dict(self.headers), body))
            capture.entered.set()
            if capture.hold:
                capture.release.wait(2)
            self.send_response(capture.status, "private-collector-reason")
            self.send_header("Location", capture.url + "/redirect-private")
            if not capture.omit_length:
                self.send_header(
                    "Content-Length",
                    str(
                        len(capture.body)
                        if capture.declared_size is None
                        else capture.declared_size
                    ),
                )
            if capture.encoding:
                self.send_header("Content-Encoding", capture.encoding)
            self.end_headers()
            try:
                if capture.trickle:
                    for byte in capture.body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        if capture.release.wait(0.04):
                            break
                else:
                    self.wfile.write(capture.body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, format: str, *args: Any) -> None:
            """Do not write request headers or transport details into test output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    capture.url = f"http://127.0.0.1:{server.server_port}/v1/traces"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield capture
    finally:
        capture.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def unsafe_span() -> ReadableSpan:
    """Put a marker in every non-allowlisted SDK field, including nested event/link metadata."""
    context = SpanContext(123, 456, False, TraceFlags(1), TraceState([("private", "secret")]))
    return ReadableSpan(
        "private-prompt-name",
        context=context,
        parent=context,
        resource=Resource({"private.resource": "secret", "service.name": "private-name"}),
        attributes={
            "prompt": "private-prompt",
            "gen_ai.request.model": "qwen-test",
            "finserve.outcome": "success",
            "gen_ai.usage.output_tokens": 7,
            "finserve.prompt_characters": 10,
            "finserve.max_tokens": 20,
        },
        events=[Event("private-event", attributes={"private": "secret"})],
        links=[Link(context, {"private": "secret"})],
        status=Status(StatusCode.ERROR, "private-status"),
        start_time=1,
        end_time=100,
        instrumentation_scope=InstrumentationScope(
            "private-scope", attributes={"private": "secret"}
        ),
    )


@pytest.mark.parametrize("protocol", ["otlp", "langfuse-v4"])
async def test_real_otlp_payload_privacy(
    collector: Collector,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    protocol: str,
) -> None:
    """A real protobuf collector receives causal IDs/counts but no prompts or ambient metadata."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    protobuf = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", collector.url)
    monkeypatch.setenv("FINSERVE_OTLP_PROTOCOL", protocol)
    if protocol == "langfuse-v4":
        collector.body = json.dumps(
            {"id": "1", "name": "otel-ingestion-job", "data": {"name": "otel-ingestion-job"}}
        ).encode()
    monkeypatch.setenv("FINSERVE_OTLP_AUTHORIZATION", "Bearer synthetic-secret")
    monkeypatch.setenv("FINSERVE_TRACE_SAMPLE_RATIO", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "private=ambient-secret")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    runtime = from_env(component="route")
    assert runtime is not None
    try:
        assert (
            await asyncio.to_thread(runtime.exporter.export, [unsafe_span()])
            is SpanExportResult.SUCCESS
        )
    finally:
        await asyncio.to_thread(runtime.close)
    assert len(collector.requests) == 1
    path, headers, payload = collector.requests[0]
    assert path == "/v1/traces"
    assert headers["Authorization"] == "Bearer synthetic-secret"
    assert headers["Accept-Encoding"] == "identity"
    assert headers.get("x-langfuse-ingestion-version") == (
        "4" if protocol == "langfuse-v4" else None
    )
    assert "private" not in headers
    assert b"private" not in payload and b"secret" not in payload
    decoded = protobuf.ExportTraceServiceRequest.FromString(payload)
    resource = decoded.resource_spans[0]
    assert resource.resource.attributes[0].value.string_value == "finserve.route"
    scope = resource.scope_spans[0]
    assert scope.scope.name == "finserve" and not scope.scope.attributes
    span = scope.spans[0]
    assert span.name == "finserve.span"
    assert int.from_bytes(span.trace_id) == 123 and int.from_bytes(span.parent_span_id) == 456
    assert not span.trace_state and not span.events and not span.links and not span.status.message
    assert span.start_time_unix_nano == 1 and span.end_time_unix_nano == 100
    assert {item.key for item in span.attributes} == {
        "gen_ai.request.model",
        "finserve.outcome",
        "gen_ai.usage.output_tokens",
        "finserve.prompt_characters",
        "finserve.max_tokens",
    }
    assert "private" not in caplog.text and "synthetic-secret" not in caplog.text


@pytest.mark.parametrize("status", [302, 401, 429, 503])
async def test_collector_rejection_is_bounded_and_redacted(
    status: int,
    collector: Collector,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Redirects cannot forward credentials; failure bodies/reasons never enter SDK logs."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    collector.status = status
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", collector.url)
    monkeypatch.setenv("FINSERVE_OTLP_TIMEOUT_SECONDS", "0.1")
    runtime = from_env()
    assert runtime is not None
    try:
        result = await asyncio.to_thread(runtime.exporter.export, [unsafe_span()])
        assert result is SpanExportResult.FAILURE and runtime.exporter.failures == 1
    finally:
        await asyncio.to_thread(runtime.close)
    assert len(collector.requests) == 1
    assert "private" not in caplog.text


async def test_export_timeout_and_shutdown_leave_loop_responsive(
    collector: Collector, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A blocked collector delays only SDK export/shutdown threads, not concurrent async work."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    collector.hold = True
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", collector.url)
    monkeypatch.setenv("FINSERVE_OTLP_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("FINSERVE_TRACE_SAMPLE_RATIO", "1")
    runtime = from_env()
    assert runtime is not None
    runtime.tracer.start_span("finserve.inference").end()
    started = time.perf_counter()
    closing = asyncio.create_task(asyncio.to_thread(runtime.close))
    ticks = 0
    while not closing.done():
        ticks += 1
        await asyncio.sleep(0.01)
    await closing
    assert collector.entered.is_set()
    assert ticks >= 3 and time.perf_counter() - started < 2
    assert runtime.exporter.failures == 1
    assert "private" not in caplog.text


@pytest.mark.parametrize(
    "kind", ["partial", "invalid", "oversize", "actual-oversize", "trickle", "encoded"]
)
async def test_otlp_acknowledgement_failures(
    kind: str,
    collector: Collector,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """HTTP200 is insufficient: dropped spans, invalid protobuf and unbounded bodies fail."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    proto = importlib.import_module("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    if kind == "partial":
        acknowledgement = proto.ExportTraceServiceResponse()
        acknowledgement.partial_success.rejected_spans = 1
        acknowledgement.partial_success.error_message = "private-rejection-reason"
        collector.body = acknowledgement.SerializeToString()
    elif kind == "invalid":
        collector.body = b"private-invalid-protobuf"
    elif kind == "oversize":
        collector.declared_size = 999999999
    elif kind == "actual-oversize":
        acknowledgement = proto.ExportTraceServiceResponse()
        acknowledgement.partial_success.error_message = "x" * 17000
        collector.body = acknowledgement.SerializeToString()
        collector.omit_length = True
    elif kind == "encoded":
        collector.encoding = "gzip"
    else:
        collector.body = b"private-body" * 50
        collector.trickle = True
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", collector.url)
    monkeypatch.setenv("FINSERVE_OTLP_TIMEOUT_SECONDS", "0.15")
    runtime = from_env()
    assert runtime is not None
    try:
        started = time.perf_counter()
        result = await asyncio.to_thread(runtime.exporter.export, [unsafe_span()])
        assert result is SpanExportResult.FAILURE and runtime.exporter.failures == 1
        assert time.perf_counter() - started < 1
    finally:
        await asyncio.to_thread(runtime.close)
    assert len(collector.requests) == 1 and "private" not in caplog.text


def test_no_config_and_unique_component_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled config ignores unused options; actor restarts cannot collide with gateway files."""
    monkeypatch.setenv("FINSERVE_OTLP_TIMEOUT_SECONDS", "invalid-unused")
    assert from_env() is None
    monkeypatch.delenv("FINSERVE_OTLP_TIMEOUT_SECONDS")
    output = tmp_path / "traces.jsonl"
    monkeypatch.setenv("FINSERVE_TRACE_PATH", str(output))
    for component in ("gateway", "engine", "engine", "route"):
        runtime = from_env(component=component)
        assert runtime is not None
        runtime.close()
    assert output.exists()
    assert len(list(tmp_path.glob("traces.engine.*.jsonl"))) == 2
    assert len(list(tmp_path.glob("traces.route.*.jsonl"))) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("FINSERVE_OTLP_ENDPOINT", "http://public.example/v1/traces"),
        ("FINSERVE_OTLP_ENDPOINT", "https://secret@collector.example/v1/traces"),
        ("FINSERVE_OTLP_ENDPOINT", "https://collector.example/v1/traces?key=secret"),
        ("FINSERVE_OTLP_ENDPOINT", "https://collector.example:999999/v1/traces"),
        ("FINSERVE_OTLP_ENDPOINT", "https://collector.example/\nsecret"),
        ("FINSERVE_OTLP_ENDPOINT", "https://collector.example/#secret"),
        ("FINSERVE_OTLP_TIMEOUT_SECONDS", "nan"),
        ("FINSERVE_OTLP_TIMEOUT_SECONDS", "0"),
        ("FINSERVE_OTLP_TIMEOUT_SECONDS", "11"),
        ("FINSERVE_TRACE_SAMPLE_RATIO", "inf"),
        ("FINSERVE_TRACE_SAMPLE_RATIO", "secret"),
        ("FINSERVE_OTLP_AUTHORIZATION", "Bearer secret\r\nInjected: x"),
        ("FINSERVE_OTLP_AUTHORIZATION", ""),
        ("FINSERVE_OTLP_PROTOCOL", "private-secret"),
    ],
    ids=[
        "public-http",
        "userinfo",
        "query",
        "port",
        "control",
        "fragment",
        "nan-timeout",
        "zero-timeout",
        "large-timeout",
        "inf-sample",
        "bad-sample",
        "header-injection",
        "empty-auth",
        "bad-protocol",
    ],
)
def test_bad_config_has_static_errors(
    key: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate before opening any collector/file; exception messages never echo bad secrets."""
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError) as error:
        from_env()
    assert "secret" not in str(error.value)


def test_ambiguous_output_and_component_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process selects one output and a fixed service identity before allocating resources."""
    path = tmp_path / "must-not-exist.jsonl"
    monkeypatch.setenv("FINSERVE_TRACE_PATH", str(path))
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
    with pytest.raises(ValueError, match="exactly one"):
        from_env()
    with pytest.raises(ValueError, match="component"):
        from_env(component="private-name")
    assert not path.exists()


def test_direct_json_and_custom_sink_are_sanitized(tmp_path: Path) -> None:
    """Both direct JSON usage and runtime custom sinks share the complete metadata filter."""
    exporter = JsonSpanExporter(tmp_path / "direct.jsonl")
    assert exporter.export([unsafe_span()]) is SpanExportResult.SUCCESS
    exporter.shutdown()
    raw = (tmp_path / "direct.jsonl").read_text()
    assert "private" not in raw
    assert json.loads(raw)["attributes"]["gen_ai.usage.output_tokens"] == 7

    class RejectingExporter(SpanExporter):
        """A sink exception should increment safe counters, with no escaped exception data."""

        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            """Inspect the rebuilt SDK object before deliberately failing."""
            span = spans[0]
            assert span.name == "finserve.span" and not span.events and not span.links
            assert span.status.description is None
            raise ValueError("private-sink-error")

    wrapped = SanitizingSpanExporter(RejectingExporter(), "engine")
    assert wrapped.export([unsafe_span()]) is SpanExportResult.FAILURE
    assert wrapped.failures == 1


def test_runtime_rejects_component_and_closes_sink(tmp_path: Path) -> None:
    """Direct runtime construction has the same fixed service-name contract as from_env."""
    exporter = JsonSpanExporter(tmp_path / "invalid.jsonl")
    with pytest.raises(ValueError, match="component"):
        TraceRuntime(exporter, component="private-name")
    assert exporter.stream.closed


def test_missing_optional_sdk_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuring OTLP without its extra fails before silently dropping requested telemetry."""
    original_import = builtins.__import__

    def without_exporter(name: str, *args: Any, **kwargs: Any) -> Any:
        """Simulate only the optional module being absent, retaining the installed core SDK."""
        if name.startswith("opentelemetry.exporter.otlp"):
            raise ImportError("private-import-detail")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_exporter)
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
    with pytest.raises(RuntimeError, match="telemetry optional") as failure:
        from_env()
    assert "private" not in str(failure.value)


def test_exporter_initialization_failure_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed SDK constructor cannot abandon its owned HTTP session or echo a credential."""
    module = pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    requests = importlib.import_module("requests")
    closed: list[bool] = []
    original_close = requests.Session.close

    def close(session: Any) -> None:
        """Observe real Session cleanup without opening a network connection."""
        closed.append(True)
        original_close(session)

    def fail(**kwargs: Any) -> None:
        """An arbitrary dependency error may contain sensitive configuration details."""
        raise ValueError("private-constructor-detail")

    monkeypatch.setattr(requests.Session, "close", close)
    monkeypatch.setattr(module, "OTLPSpanExporter", fail)
    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", "https://collector.example/v1/traces")
    with pytest.raises(RuntimeError, match="initialization failed") as failure:
        from_env()
    assert "private" not in str(failure.value) and closed == [True]


def test_json_failure_and_metadata_value_bounds(tmp_path: Path) -> None:
    """Bad numeric/string metadata is omitted, and an unavailable local sink records failure."""
    path = tmp_path / "filtered.jsonl"
    exporter = JsonSpanExporter(path)
    span = ReadableSpan(
        "finserve.engine",
        context=unsafe_span().context,
        attributes={
            "finserve.max_tokens": float("nan"),
            "finserve.prompt_characters": -1,
            "gen_ai.usage.output_tokens": True,
            "finserve.outcome": "private-value",
            "gen_ai.request.model": "private\nmodel",
        },
    )
    assert exporter.export([ReadableSpan("no-context"), span]) is SpanExportResult.SUCCESS
    assert json.loads(path.read_text())["attributes"] == {}
    exporter.shutdown()
    assert exporter.export([span]) is SpanExportResult.FAILURE and exporter.failures == 1
    with pytest.raises(ValueError, match="outside"):
        JsonSpanExporter(Path(__file__).resolve().parents[2] / "must-not-create-trace.jsonl")


def test_sink_shutdown_does_not_escape_private_errors() -> None:
    """Shutdown failures remain observable as a count without becoming application log text."""

    class BrokenShutdown(SpanExporter):
        """Exercise arbitrary sink teardown failure under the common wrapper."""

        def shutdown(self) -> None:
            """A third-party exporter may include sensitive configuration in exceptions."""
            raise OSError("private-shutdown-detail")

    wrapper = SanitizingSpanExporter(BrokenShutdown(), "gateway")
    wrapper.shutdown()
    assert wrapper.failures == 1
