"""Reject ambiguous internal streams before the public gateway records successful completion."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing

import httpx
import pytest

from finserve.contracts.inference import InferenceRequest
from finserve.engines.openai_adapter import EngineProtocolError, EngineUnavailableError
from finserve.engines.ray_http import RayHTTPEngine


def frame(text: str = "", count: int = 0, finish: str | None = None, replica: str = "a") -> bytes:
    """Use explicit engine accounting rather than treating HTTP chunks as tokens."""
    return (
        json.dumps(
            {
                "replica_id": replica,
                "token": {"text": text, "generated_tokens": count, "finish_reason": finish},
            }
        ).encode()
        + b"\n"
    )


class Wire(httpx.AsyncByteStream):
    """Expose socket closure and optional hangs without depending on a Ray runtime."""

    def __init__(self, chunks: list[bytes], *, hang: bool = False) -> None:
        """Keep transport behavior independent from the adapter under test."""
        self.chunks = chunks
        self.hang = hang
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield arbitrary network fragments and optionally leave the connection unfinished."""
        for chunk in self.chunks:
            yield chunk
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        """Record closure even when the consumer cancels before a terminal frame."""
        self.closed = True


def engine(wire: Wire, *, status: int = 200, maximum: int = 4096) -> RayHTTPEngine:
    """Assert fixed-destination authentication at the HTTP boundary on every request."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Capture an actual HTTPX request with the configured private routing credential."""
        assert str(request.url) == "http://ray/"
        assert request.headers["authorization"] == "Bearer test-ray-key"
        return httpx.Response(status, stream=wire, headers={"content-type": "application/x-ndjson"})

    return RayHTTPEngine(
        "http://ray/",
        api_key="test-ray-key",
        transport=httpx.MockTransport(respond),
        maximum_response_bytes=maximum,
    )


async def test_fragmented_stream_and_authoritative_usage() -> None:
    """UTF-8 and line boundaries may cross chunks without changing visible output or counts."""
    content = frame("£profit") + frame(count=3, finish="stop")
    wire = Wire([content[:13], content[13:37], content[37:]])
    adapter = engine(wire)
    try:
        tokens = [token async for token in adapter.stream(InferenceRequest(prompt="test"))]
        assert "".join(token.text for token in tokens) == "£profit"
        assert sum(token.generated_tokens for token in tokens) == 3
        assert tokens[-1].finish_reason == "stop" and wire.closed
    finally:
        await adapter.close()


@pytest.mark.parametrize(
    "chunks",
    [
        [frame("partial")],
        [b"invalid\n"],
        [frame("text").replace(b"text", b"\xff")],
        [b'{"replica_id":"a","replica_id":"b","token":{"text":"x"}}\n'],
        [frame(count=1, finish="stop").rstrip(b"\n")],
        [frame("text", count=1)],
        [frame("text"), frame(count=0, finish="stop")],
        [frame(count=100, finish="length")],
        [frame("text"), frame(count=1, finish="stop", replica="b")],
        [frame(count=1, finish="stop"), frame("late")],
        [frame("terminal text", count=1, finish="stop")],
    ],
)
async def test_invalid_stream_never_yields_terminal_success(chunks: list[bytes]) -> None:
    """Malformed, switched, duplicated and incomplete results fail with static protocol errors."""
    wire = Wire(chunks)
    adapter = engine(wire)
    try:
        with pytest.raises(EngineProtocolError):
            async for token in adapter.stream(InferenceRequest(prompt="test")):
                assert token.finish_reason is None
        assert wire.closed
    finally:
        await adapter.close()


async def test_terminal_frame_without_completed_http_body_times_out() -> None:
    """Receiving final accounting alone cannot hide a failed or stalled transport termination."""
    wire = Wire([frame(count=1, finish="stop")], hang=True)
    adapter = engine(wire)
    try:
        with pytest.raises(TimeoutError):
            async for _ in adapter.stream(InferenceRequest(prompt="test", timeout_seconds=0.05)):
                pytest.fail("terminal must wait for complete HTTP response")
        assert wire.closed
    finally:
        await adapter.close()


async def test_consumer_closure_closes_upstream() -> None:
    """Abandoning a visible partial result closes the HTTP request and triggers Ray cleanup."""
    wire = Wire([frame("partial")], hang=True)
    adapter = engine(wire)
    try:
        async with aclosing(adapter.stream(InferenceRequest(prompt="test"))) as output:
            assert (await anext(output)).text == "partial"
        assert wire.closed
    finally:
        await adapter.close()


async def test_unterminated_oversized_line_is_bounded() -> None:
    """Enforce the decoded byte budget before the line decoder buffers attacker-sized input."""
    wire = Wire([b"x" * 100])
    adapter = engine(wire, maximum=50)
    try:
        with pytest.raises(EngineProtocolError, match="byte limit"):
            _ = [token async for token in adapter.stream(InferenceRequest(prompt="test"))]
        assert wire.closed
    finally:
        await adapter.close()


async def test_redirect_is_not_followed() -> None:
    """A routing service cannot send the internal credential to another origin through redirects."""
    wire = Wire([])
    adapter = engine(wire, status=302)
    try:
        with pytest.raises(EngineUnavailableError):
            _ = [token async for token in adapter.stream(InferenceRequest(prompt="test"))]
        assert wire.closed
    finally:
        await adapter.close()


@pytest.mark.parametrize("url", ["ftp://ray", "http://user:pass@ray", "http://ray/?x=1"])
def test_invalid_endpoint(url: str) -> None:
    """Reject destination credentials and query selectors during startup."""
    with pytest.raises(ValueError):
        RayHTTPEngine(url)


@pytest.mark.parametrize("maximum", [0, -1, True, float("nan"), float("inf"), 8 * 1024 * 1024])
def test_response_budget_is_a_bounded_integer(maximum: int) -> None:
    """Trusted configuration cannot accidentally disable the byte limit through NaN or infinity."""
    with pytest.raises(ValueError):
        RayHTTPEngine("http://ray/", maximum_response_bytes=maximum)
