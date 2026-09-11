"""Inspect exported spans rather than merely asserting that an SDK object exists."""

import json
from pathlib import Path

import httpx
import pytest

from finserve.gateway.app import create_app
from finserve.telemetry.tracing import JsonSpanExporter, TraceRuntime


async def test_trace_shutdown_flushes_without_prompt_data(tmp_path: Path) -> None:
    """Completed inference produces bounded metadata and shutdown flushes its queued span."""
    path = tmp_path / "traces.jsonl"
    runtime = TraceRuntime(JsonSpanExporter(path), sample_ratio=1)
    app = create_app(tracing=runtime)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions", json={"prompt": "private-synthetic-prompt"}
            )
            assert response.status_code == 200
    raw = path.read_text()
    assert "private-synthetic-prompt" not in raw
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 1
    assert rows[0]["attributes"]["finserve.outcome"] == "success"
    assert rows[0]["attributes"]["gen_ai.usage.output_tokens"] == 25
    assert rows[0]["end_time_ns"] >= rows[0]["start_time_ns"]


def test_arbitrary_span_names_and_attributes_are_redacted(tmp_path: Path) -> None:
    """Future instrumentation cannot leak prompt-bearing names or arbitrary attributes."""
    path = tmp_path / "sanitized.jsonl"
    runtime = TraceRuntime(JsonSpanExporter(path), sample_ratio=1)
    span = runtime.tracer.start_span("private-span-name", attributes={"prompt": "private-value"})
    span.end()
    runtime.close()
    raw = path.read_text()
    assert "private" not in raw
    assert json.loads(raw)["name"] == "finserve.span"


def test_invalid_sampling_closes_exporter(tmp_path: Path) -> None:
    """Rejected runtime configuration must not leave the exclusive evidence file open."""
    exporter = JsonSpanExporter(tmp_path / "invalid.jsonl")
    with pytest.raises(ValueError):
        TraceRuntime(exporter, sample_ratio=float("nan"))
    assert exporter.stream.closed
