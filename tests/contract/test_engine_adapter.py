"""Exercise real HTTPX streaming ownership using an adversarial in-memory transport."""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from finserve.contracts.inference import InferenceRequest
from finserve.engines.openai_adapter import (
    EngineProtocolError,
    EngineUnavailableError,
    OpenAICompletionEngine,
)
from finserve.engines.sglang_adapter import SGLangEngine
from finserve.engines.vllm_adapter import VLLMEngine


class TrackingStream(httpx.AsyncByteStream):
    """Expose upstream closure and cancellation without relying on a live GPU service."""

    def __init__(self, chunks: list[bytes], wait_after_chunks: bool = False) -> None:
        """A controllable stalled tail represents a backend that stops making progress."""
        self.chunks = chunks
        self.wait_after_chunks = wait_after_chunks
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Preserve specified byte boundaries, including splits inside Unicode and CRLF."""
        for chunk in self.chunks:
            yield chunk
        if self.wait_after_chunks:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        """Make connection-release behavior directly observable in every terminal path."""
        self.closed = True


def event(text: str = "", *, finish: str | None = None) -> bytes:
    """Build standard completion deltas with optional terminal choice metadata."""
    return (
        "data: "
        + json.dumps(
            {"choices": [{"index": 0, "text": text, "finish_reason": finish}]},
            ensure_ascii=False,
        )
        + "\r\n\r\n"
    ).encode("utf-8")


def usage(tokens: object = 3) -> bytes:
    """Allow malformed count fixtures to verify strict authoritative accounting."""
    return (
        "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": tokens}}) + "\n\n"
    ).encode()


def make_engine(
    stream: TrackingStream,
    *,
    status: int = 200,
    content_type: str = "text/event-stream; charset=utf-8",
    maximum_bytes: int = 4 * 1024 * 1024,
) -> tuple[OpenAICompletionEngine, list[httpx.Request]]:
    """Record outbound requests while routing all network activity to the supplied fixture."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Return headers and a streaming body independently, matching an actual HTTP server."""
        requests.append(request)
        return httpx.Response(status, headers={"content-type": content_type}, stream=stream)

    return OpenAICompletionEngine(
        "http://engine.invalid/v1",
        api_key="private-key",
        transport=httpx.MockTransport(respond),
        maximum_response_bytes=maximum_bytes,
    ), requests


@pytest.mark.asyncio
async def test_arbitrary_chunks_and_authoritative_usage() -> None:
    """Three generated tokens may arrive in one or many text frames; neither is a token count."""
    wire = b": keepalive\r\n\r\n" + event("profit \u20ac") + event(finish="stop") + usage()
    wire += b"data: [DONE]\n\n"
    stream = TrackingStream([wire[index : index + 1] for index in range(len(wire))])
    engine, requests = make_engine(stream)
    output = [
        token async for token in engine.stream(InferenceRequest(prompt="hello", max_tokens=4))
    ]
    assert [(token.text, token.generated_tokens) for token in output] == [
        ("profit \u20ac", 0),
        ("", 3),
    ]
    assert len(requests) == 1
    assert output[-1].finish_reason == "stop"
    assert str(requests[0].url) == "http://engine.invalid/v1/completions"
    assert requests[0].headers["authorization"] == "Bearer private-key"
    sent = json.loads(requests[0].content)
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["n"] == 1 and sent["prompt"] == "hello"
    assert stream.closed
    await engine.close()


@pytest.mark.asyncio
async def test_multiline_sse_and_finish_with_text() -> None:
    """SSE joins data fields before JSON decoding and permits text on the finishing choice."""
    wire = b'data: {"choices":\ndata: [{"index":0,"text":"ok","finish_reason":"length"}]}\n\n'
    stream = TrackingStream([wire + usage(1) + b"data: [DONE]\n\n"])
    engine, _ = make_engine(stream)
    output = [token async for token in engine.stream(InferenceRequest(prompt="p"))]
    assert [(token.text, token.generated_tokens) for token in output] == [("ok", 0), ("", 1)]
    assert output[-1].finish_reason == "length"
    assert stream.closed
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire",
    [
        event("ok") + event(finish="stop") + b"data: [DONE]\n\n",
        event("ok") + event(finish="stop") + usage(),
        event("ok") + usage() + b"data: [DONE]\n\n",
        event("ok") + event(finish="stop") + usage(-1),
        event("ok") + event(finish="stop") + usage(True),
        event("ok") + event(finish="stop") + usage("3"),
        event("ok") + event(finish="stop") + usage(0),
        event("ok") + event(finish="stop") + usage(100),
        event("ok") + event(finish="stop") + usage() + usage(),
        event("ok", finish="stop") + event("late"),
        b'data: {"error":{"message":"private-key"}}\n\n',
        b"data: private-key\n\n",
        b"data: [DONE]",
        b'data: {"choices":[{"index":1,"text":"wrong choice"}]}\n\n',
        event(finish="tool_calls"),
    ],
)
async def test_malformed_or_truncated_stream_fails_and_closes(wire: bytes) -> None:
    """Never turn missing or corrupt final accounting into a successful measured completion."""
    stream = TrackingStream([wire])
    engine, requests = make_engine(stream)
    with pytest.raises(EngineProtocolError) as error:
        _ = [token async for token in engine.stream(InferenceRequest(prompt="p", max_tokens=8))]
    assert "private-key" not in str(error.value)
    assert len(str(error.value)) < 128
    assert stream.closed and len(requests) == 1
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 401, 429, 500])
async def test_http_error_never_retries_or_echoes_body(status: int) -> None:
    """Even redirects remain at the configured trust boundary; server bodies are not diagnostics."""
    stream = TrackingStream([b"private-key: secret prompt"])
    engine, requests = make_engine(stream, status=status)
    with pytest.raises(EngineUnavailableError, match=str(status)) as error:
        await anext(engine.stream(InferenceRequest(prompt="p")))
    assert "private-key" not in str(error.value)
    assert len(requests) == 1 and stream.closed
    await engine.close()


@pytest.mark.asyncio
async def test_wrong_content_type_and_response_limits() -> None:
    """Reject non-SSE and overlong unterminated data before parsing unbounded JSON."""
    for content_type, wire, maximum in (
        ("application/json", b"{}", 100),
        ("text/event-stream", b"data: " + b"x" * 1000, 100),
        ("text/event-stream", b"data: " + b"x" * 65536 + b"\n\n", 100000),
    ):
        stream = TrackingStream([wire])
        engine, _ = make_engine(stream, content_type=content_type, maximum_bytes=maximum)
        with pytest.raises(EngineProtocolError):
            await anext(engine.stream(InferenceRequest(prompt="p")))
        assert stream.closed
        await engine.close()


@pytest.mark.asyncio
async def test_generator_close_releases_upstream_after_visible_output() -> None:
    """Disconnect after the first frame must close the response even without a final usage event."""
    stream = TrackingStream([event("visible")], wait_after_chunks=True)
    engine, requests = make_engine(stream)
    output = engine.stream(InferenceRequest(prompt="p"))
    assert (await anext(output)).text == "visible"
    await output.aclose()
    assert stream.closed and len(requests) == 1
    await engine.close()


@pytest.mark.asyncio
async def test_cancellation_closes_stalled_upstream() -> None:
    """Cancellation while awaiting network bytes propagates and closes the pooled response."""
    stream = TrackingStream([], wait_after_chunks=True)
    engine, _ = make_engine(stream)
    output = engine.stream(InferenceRequest(prompt="p"))
    pending = asyncio.ensure_future(anext(output))
    await stream.waiting.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert stream.closed
    await output.aclose()
    await engine.close()


@pytest.mark.asyncio
async def test_deadline_closes_stalled_upstream() -> None:
    """Enforce a whole-request deadline even if a test transport lacks read timers."""
    stream = TrackingStream([], wait_after_chunks=True)
    engine, _ = make_engine(stream)
    with pytest.raises(TimeoutError):
        await anext(engine.stream(InferenceRequest(prompt="p", timeout_seconds=0.01)))
    assert stream.closed
    await engine.close()


@pytest.mark.asyncio
async def test_deadline_context_does_not_cancel_consumer_between_yields() -> None:
    """A suspended async generator must not leave an asyncio timer active in its caller's task."""
    stream = TrackingStream([event("visible")], wait_after_chunks=True)
    engine, _ = make_engine(stream)
    output = engine.stream(InferenceRequest(prompt="p", timeout_seconds=0.01))
    await anext(output)
    await asyncio.sleep(0.02)
    with pytest.raises(TimeoutError):
        await anext(output)
    assert stream.closed
    await engine.close()


@pytest.mark.parametrize(
    "url",
    ["ftp://engine/v1", "http://user:private-key@engine/v1", "http://engine/v1?k=s", "/v1"],
)
def test_invalid_trusted_configuration_is_rejected(url: str) -> None:
    """Credential-bearing URLs can leak through HTTP diagnostics, so configuration rejects them."""
    with pytest.raises(ValueError):
        OpenAICompletionEngine(url)


def test_deployment_wrappers_share_protocol() -> None:
    """Comparing backends must retain the exact same parser and token-accounting semantics."""
    assert issubclass(VLLMEngine, OpenAICompletionEngine)
    assert issubclass(SGLangEngine, OpenAICompletionEngine)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_timeout", [False, True])
async def test_transport_errors_are_sanitized(is_timeout: bool) -> None:
    """Do not leak exception request URLs or headers when HTTPX reports an upstream failure."""

    def fail(request: httpx.Request) -> httpx.Response:
        """Raise a transport-level failure independently of protocol body validation."""
        if is_timeout:
            raise httpx.ReadTimeout("private-key", request=request)
        raise httpx.RemoteProtocolError("private-key", request=request)

    engine = OpenAICompletionEngine("http://engine.invalid/v1", transport=httpx.MockTransport(fail))
    with pytest.raises(TimeoutError if is_timeout else EngineUnavailableError) as error:
        await anext(engine.stream(InferenceRequest(prompt="p")))
    assert "private-key" not in str(error.value)
    await engine.close()


def test_invalid_response_budget_is_rejected() -> None:
    """Reject an unusable safety budget during configuration, before opening a client."""
    with pytest.raises(ValueError, match="positive"):
        OpenAICompletionEngine("http://engine.invalid/v1", maximum_response_bytes=0)
