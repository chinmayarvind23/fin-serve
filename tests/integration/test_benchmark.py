"""Transport-level failure fixtures verify the recorder without pretending mock latency is real."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from finserve.benchmark.runner import (
    RunConfig,
    request_one,
    run_benchmark,
    validate_comparison,
    validate_evidence,
)
from finserve.benchmark.workload import default_workload


class DelayedStream(httpx.AsyncByteStream):
    """Emit real async delays in fixture bodies to exercise total request deadlines."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Content arrives before a deliberate stall, preserving partial output on timeout."""
        yield b'data: {"choices":[{"text":"partial"}]}\n\n'
        await asyncio.sleep(0.1)
        yield b"data: [DONE]\n\n"


def successful_response(request: httpx.Request) -> httpx.Response:
    """One chunk represents seventeen server tokens, exposing chunk-count mistakes."""
    return httpx.Response(
        200,
        request=request,
        content=(
            'data: {"choices":[{"text":"hello"}]}\n\n'
            'data: {"choices":[],"usage":{"completion_tokens":17},'
            '"finserve":{"server_ttft_seconds":0.02}}\n\n'
            "data: [DONE]\n\n"
        ),
    )


async def test_recorded_run_preserves_failures_and_warmup(tmp_path: Path) -> None:
    """A deterministic 503 remains one of four measured logical requests."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail the third request, after the separately recorded warmup request."""
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request) if calls == 3 else successful_response(request)

    output = tmp_path / "run"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_benchmark(
            client,
            "http://test/v1/completions",
            default_workload(),
            RunConfig(requests=4, warmup=1, concurrency=1),
            output,
        )
    assert report["offered_requests"] == 4
    assert report["successful_requests"] == 3
    assert report["generated_tokens"] == 51
    assert report["server_ttft_median_s"] == 0.02
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert sum(row["success"] is False for row in rows) == 1
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed"


async def test_timeout_keeps_partial_content() -> None:
    """The stream context closes when the total deadline expires after first content."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an async stalled body rather than buffered ASGI response timing."""
        return httpx.Response(200, request=request, stream=DelayedStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await request_one(
            client,
            "http://test",
            default_workload().items[0],
            0,
            0,
            RunConfig(timeout_s=0.01),
            "measured",
        )
    assert not row.success
    assert row.output == "partial"
    assert row.first_content_s is not None
    assert row.error == "TimeoutError"


@pytest.mark.parametrize(
    "body",
    [
        'data: {"choices":[{"text":"partial"}]}\n\n',
        'data: {"error":{"message":"engine failed"}}\n\n',
        'data: {"usage":{"completion_tokens":true}}\n\ndata: [DONE]\n\n',
        "data: [DONE]\n\n",
        "data: " + "a" * 1_000_001,
        'data: {"choices":[{"text":"x"}],"usage":{"completion_tokens":10000}}\n\ndata: [DONE]\n\n',
        'data: {"choices":[{"text":"x","finish_reason":"invented"}]}\n\ndata: [DONE]\n\n',
        'data: {"choices":[{"text":"x"}],"usage":{"completion_tokens":2}}\n\n'
        'data: {"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n',
    ],
)
async def test_protocol_errors_are_failed_requests(body: str) -> None:
    """Truncation, explicit engine errors and invalid token counts cannot count as success."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Inject protocol violations through the same HTTP transport interface."""
        return httpx.Response(200, request=request, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await request_one(
            client, "http://test", default_workload().items[0], 0, 0, RunConfig(), "measured"
        )
    assert row.success is False
    assert row.error is not None


async def test_open_loop_overflow_retains_every_offered_request(tmp_path: Path) -> None:
    """At most one worker plus one waiting request survives a deliberately excessive rate."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Stall enough to make the bounded client queue overflow reproducibly."""
        return httpx.Response(200, request=request, stream=DelayedStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_benchmark(
            client,
            "http://test",
            default_workload(),
            RunConfig(requests=20, concurrency=1, warmup=0, mode="open", rate=10000),
            tmp_path / "open",
        )
    rows = [
        json.loads(line) for line in (tmp_path / "open/requests.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 20
    assert report["offered_requests"] == 20
    assert any(row["error"] == "client_queue_full" for row in rows)


async def test_comparison_requires_same_workload(tmp_path: Path) -> None:
    """Changing output length after baseline must invalidate comparison."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(successful_response)) as client:
        config = RunConfig(
            requests=1,
            warmup=0,
            hardware="fixture CPU",
            revision="test",
            model_revision="fixture-v1",
            tokenizer_revision="fixture-v1",
            engine="mock-http-fixture",
            engine_config="default",
        )
        await run_benchmark(client, "http://test", default_workload(24), config, tmp_path / "b")
        await run_benchmark(client, "http://test", default_workload(25), config, tmp_path / "c")
    with pytest.raises(ValueError, match="workload mismatch"):
        validate_comparison(tmp_path / "b", tmp_path / "c")
    validate_evidence(tmp_path / "b")
    summary = tmp_path / "b/summary.json"
    payload = json.loads(summary.read_text())
    payload["requests_per_second"] *= 10
    summary.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary disagrees"):
        validate_evidence(tmp_path / "b")


async def test_interruption_persists_inflight_and_unsent_requests(tmp_path: Path) -> None:
    """Cancellation cannot erase failed observations or make an incomplete run comparable."""
    started = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        """Hold the first stream open until the run is deliberately cancelled."""
        started.set()
        return httpx.Response(200, request=request, stream=DelayedStream())

    output = tmp_path / "cancelled"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(
            run_benchmark(
                client,
                "http://test",
                default_workload(),
                RunConfig(requests=5, warmup=0, concurrency=1),
                output,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert any(row["output"] == "partial" and row["error"] == "CancelledError" for row in rows)
    assert json.loads((output / "manifest.json").read_text())["status"] == "interrupted"


async def test_manifest_and_errors_redact_credentials_and_invalid_content(tmp_path: Path) -> None:
    """Credential-bearing URLs and validation text must not leak through metadata/errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        """An invalid usage count would appear in a raw Pydantic ValidationError string."""
        return httpx.Response(
            200,
            request=request,
            content=('data: {"usage":{"completion_tokens":"secret-output-value"}}\n\n'),
        )

    output = tmp_path / "redaction"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await run_benchmark(
            client,
            "http://alice:secret-password@test/?key=secret-query",
            default_workload(),
            RunConfig(requests=1, warmup=0),
            output,
        )
    manifest_text = (output / "manifest.json").read_text()
    raw_text = (output / "requests.jsonl").read_text()
    assert "secret-password" not in manifest_text
    assert "secret-query" not in manifest_text
    assert "alice" not in manifest_text
    assert "secret-output-value" not in raw_text


@pytest.mark.parametrize(
    "artifact", ["missing_raw", "missing_request", "changed_hash", "changed_window"]
)
async def test_incomplete_or_changed_evidence_is_rejected(tmp_path: Path, artifact: str) -> None:
    """Raw populations and workload/window identity must agree before a comparison."""
    output = tmp_path / artifact
    config = RunConfig(
        requests=2,
        warmup=0,
        hardware="fixture",
        revision="fixture",
        model_revision="fixture",
        tokenizer_revision="fixture",
        engine="fixture",
        engine_config="fixture",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(successful_response)) as client:
        await run_benchmark(client, "http://test", default_workload(), config, output)
    if artifact == "missing_raw":
        (output / "requests.jsonl").unlink()
    elif artifact == "missing_request":
        raw = output / "requests.jsonl"
        raw.write_text(raw.read_text().splitlines()[0] + "\n")
    else:
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        if artifact == "changed_hash":
            manifest["workload_hash"] = "changed"
        else:
            manifest["measured_seconds"] *= 2
        path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_evidence(output)
