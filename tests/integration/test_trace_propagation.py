"""Trace identity crosses serialized hops without leaking context or user-controlled metadata."""

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
from opentelemetry.trace import get_current_span, use_span

from finserve.engines.openai_adapter import OpenAICompletionEngine
from finserve.gateway import app as gateway_module
from finserve.gateway.app import create_app
from finserve.telemetry.propagation import parent_context, trace_headers, traced_stream
from finserve.telemetry.tracing import JsonSpanExporter, TraceRuntime


def test_failed_gateway_construction_closes_new_exporter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Late validation must return the exporter thread and exclusive file."""
    exporter = JsonSpanExporter(tmp_path / "startup.jsonl")
    runtime = TraceRuntime(exporter, sample_ratio=1)
    monkeypatch.setattr(gateway_module, "tracing_from_env", lambda: runtime)
    monkeypatch.setenv("FINSERVE_ENGINE", "fixture")
    monkeypatch.setenv("FINSERVE_MAX_CONCURRENCY", "0")
    with pytest.raises(ValueError):
        gateway_module.from_env()
    assert exporter.stream.closed


def test_invalid_capacity_does_not_allocate_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject malformed environment numbers before tracing can acquire background ownership."""

    def forbidden() -> TraceRuntime:
        """An invalid capacity must never reach this resource-producing factory."""
        raise AssertionError("unexpected exporter allocation")

    monkeypatch.setattr(gateway_module, "tracing_from_env", forbidden)
    monkeypatch.setenv("FINSERVE_ENGINE", "fixture")
    monkeypatch.setenv("FINSERVE_MAX_CONCURRENCY", "invalid")
    with pytest.raises(ValueError):
        gateway_module.from_env()


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "x" * 10000,
        "00-" + "0" * 32 + "-" + "1" * 16 + "-01",
        "00-" + "1" * 32 + "-" + "0" * 16 + "-01",
        "01-" + "1" * 32 + "-" + "2" * 16 + "-01",
    ],
    ids=["missing", "empty", "oversize", "zero-trace", "zero-span", "future-version"],
)
def test_invalid_parent_creates_empty_context(value: str | None) -> None:
    """Private tracing must not make malformed headers into a serving failure or valid parent."""
    assert not get_current_span(parent_context(value)).get_span_context().is_valid


async def test_generator_context_is_detached_at_yield_and_closed_on_cancellation(
    tmp_path: Path,
) -> None:
    """A consumer may resume or close from another task without inheriting producer context."""
    runtime = TraceRuntime(JsonSpanExporter(tmp_path / "owned.jsonl"), sample_ratio=1)
    seen: list[dict[str, str]] = []
    closed = asyncio.Event()

    async def source() -> AsyncGenerator[int]:
        """Observe context during execution and finalization, never through model content."""
        try:
            seen.append(trace_headers())
            yield 1
            await asyncio.Event().wait()
        finally:
            seen.append(trace_headers())
            closed.set()

    output = traced_stream(source(), runtime.tracer, "finserve.engine", None)
    assert await anext(output) == 1
    assert trace_headers() == {}
    pending = asyncio.create_task(anext(output))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert closed.is_set() and trace_headers() == {}
    assert seen[0] == seen[1] and len(seen[0]["traceparent"]) == 55
    runtime.close()
    row = json.loads((tmp_path / "owned.jsonl").read_text())
    assert row["status"] == "ERROR" and row["attributes"]["finserve.outcome"] == "cancelled"


async def test_serialized_parent_links_route_and_engine_without_baggage(tmp_path: Path) -> None:
    """Explicit parent strings survive a task boundary independently of implicit contextvars."""
    runtime = TraceRuntime(JsonSpanExporter(tmp_path / "chain.jsonl"), sample_ratio=1)
    root = runtime.tracer.start_span("finserve.inference")
    with use_span(root, end_on_exit=False):
        carrier = trace_headers()
    assert set(carrier) == {"traceparent"}
    seen: list[str] = []

    async def engine() -> AsyncGenerator[int]:
        """This is the downstream HTTP identity emitted while the engine child is active."""
        seen.append(trace_headers()["traceparent"])
        yield 1

    async def route() -> AsyncGenerator[int]:
        """Serialize the route's own identity just as the router passes actor RPC arguments."""
        parent = trace_headers()["traceparent"]
        output = traced_stream(engine(), runtime.tracer, "finserve.engine", parent)
        try:
            async for item in output:
                yield item
        finally:
            await output.aclose()

    output = traced_stream(route(), runtime.tracer, "finserve.route", carrier["traceparent"])
    assert [item async for item in output] == [1]
    assert trace_headers() == {}
    root.end()
    runtime.close()
    rows = {
        row["name"]: row
        for row in map(json.loads, (tmp_path / "chain.jsonl").read_text().splitlines())
    }
    assert len({row["trace_id"] for row in rows.values()}) == 1
    assert rows["finserve.route"]["parent_span_id"] == rows["finserve.inference"]["span_id"]
    assert rows["finserve.engine"]["parent_span_id"] == rows["finserve.route"]["span_id"]
    assert seen[0].split("-")[2] == rows["finserve.engine"]["span_id"]


@pytest.mark.parametrize("failure", ["body", "close"])
async def test_failed_generation_or_close_ends_span_without_error_text(
    tmp_path: Path, failure: str
) -> None:
    """Both execution and finalizer failures retain status without exporting exception content."""
    path = tmp_path / "failed.jsonl"
    runtime = TraceRuntime(JsonSpanExporter(path), sample_ratio=1)

    async def source() -> AsyncGenerator[int]:
        """A partial stream and a failing close exercise separate ownership exits."""
        try:
            yield 1
            raise ValueError("private-generation-error")
        finally:
            if failure == "close":
                raise ValueError("private-close-error")

    output = traced_stream(source(), runtime.tracer, "finserve.engine", None)
    assert await anext(output) == 1
    with pytest.raises(ValueError):
        if failure == "body":
            await anext(output)
        else:
            await output.aclose()
    assert trace_headers() == {}
    runtime.close()
    raw = path.read_text()
    assert "private" not in raw
    row = json.loads(raw)
    assert row["status"] == "ERROR" and row["attributes"]["finserve.outcome"] == "failed"


async def test_gateway_concurrent_streams_have_distinct_private_trace_roots(tmp_path: Path) -> None:
    """External parent/baggage cannot control gateway sampling or merge unrelated requests."""
    runtime = TraceRuntime(JsonSpanExporter(tmp_path / "http.jsonl"), sample_ratio=1)
    observed: list[dict[str, str]] = []
    barrier = asyncio.Barrier(2)

    async def upstream(request: httpx.Request) -> httpx.Response:
        """Inspect actual HTTPX outbound headers while both gateway requests are active."""
        observed.append(dict(request.headers))
        await barrier.wait()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"index":0,"text":"ok","finish_reason":"stop"}]}\n\n'
                b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    engine = OpenAICompletionEngine("http://engine/v1", transport=httpx.MockTransport(upstream))
    app = create_app(engine, tracing=runtime)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway") as client,
    ):
        responses = await asyncio.gather(
            *(
                client.post(
                    "/v1/completions",
                    json={"prompt": "private-prompt", "max_tokens": 2},
                    headers={
                        "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
                        "baggage": "secret=private-value",
                        "tracestate": "private=value",
                    },
                )
                for _ in range(2)
            )
        )
        assert all(
            response.status_code == 200 and "[DONE]" in response.text for response in responses
        )
    assert len({headers["traceparent"] for headers in observed}) == 2
    assert all("baggage" not in headers and "tracestate" not in headers for headers in observed)
    assert all(headers["traceparent"].split("-")[1] != "a" * 32 for headers in observed)
    assert trace_headers() == {}
    assert "private-prompt" not in (tmp_path / "http.jsonl").read_text()
