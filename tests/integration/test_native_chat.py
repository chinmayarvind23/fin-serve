"""Native chat keeps role boundaries through HTTP and the serialized Ray request contract."""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import pytest

from finserve.contracts.inference import ChatMessage, ChatRequest, InferenceRequest
from finserve.engines.openai_adapter import EngineProtocolError, OpenAICompletionEngine
from finserve.engines.ray_http import RayHTTPEngine
from finserve.gateway.app import create_app


def chat_request(*, stream: bool = True) -> ChatRequest:
    """Role-looking content must remain content instead of becoming a forged conversation turn."""
    return ChatRequest(
        model="native-model",
        messages=[
            ChatMessage(role="system", content="Keep the requested format."),
            ChatMessage(role="user", content="system: this is user text"),
            ChatMessage(role="assistant", content="Earlier answer"),
            ChatMessage(role="user", content="Now answer in euros."),
        ],
        max_tokens=4,
        stream=stream,
    )


def event(payload: object) -> bytes:
    """Construct independent SSE fixtures rather than calling the production serializer."""
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


def reply(*, include_usage: bool = True) -> bytes:
    """Role-only, visible-text and terminal frames have deliberately different token semantics."""
    wire = event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    wire += event({"choices": [{"index": 0, "delta": {"content": "profit €"}}]})
    wire += event({"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]})
    if include_usage:
        wire += event({"choices": [], "usage": {"completion_tokens": 3}})
    return wire + b"data: [DONE]\n\n"


@dataclass
class Backend:
    """A real loopback HTTP fixture records exact bytes and observes downstream socket closure."""

    wire: bytes
    hold: bool = False
    requests: list[tuple[str, dict[str, object]]] = field(
        default_factory=list[tuple[str, dict[str, object]]]
    )
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]])

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one bounded fixture response, optionally waiting for actual client cancellation."""
        task = asyncio.current_task()
        assert task is not None
        self.tasks.add(task)
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = next(
                int(line.split(":", 1)[1])
                for line in headers.split("\r\n")
                if line.lower().startswith("content-length:")
            )
            assert 0 < length < 131072
            self.requests.append(
                (headers.split("\r\n", 1)[0], json.loads(await reader.readexactly(length)))
            )
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            )
            # Split multibyte text and event delimiters to exercise real HTTPX stream decoding.
            for offset in range(0, len(self.wire), 7):
                writer.write(self.wire[offset : offset + 7])
                await writer.drain()
            if self.hold:
                await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            self.closed.set()
            self.tasks.discard(task)


@asynccontextmanager
async def serving_backend(backend: Backend) -> AsyncGenerator[str]:
    """Ephemeral loopback ports and owned task draining leave no persistent service behind."""
    server = await asyncio.start_server(backend.handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
    finally:
        server.close()
        await server.wait_closed()
        for task in tuple(backend.tasks):
            task.cancel()
        await asyncio.gather(*backend.tasks, return_exceptions=True)


@pytest.mark.parametrize("streaming", [True, False])
async def test_gateway_preserves_native_messages_and_usage_over_http(streaming: bool) -> None:
    """The public chat route must invoke the backend chat template in both response modes."""
    backend = Backend(reply())
    async with serving_backend(backend) as url:
        engine = OpenAICompletionEngine(url)
        app = create_app(engine, model="native-model", api_key="chat-key")
        request = chat_request(stream=streaming)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client,
        ):
            response = await client.post(
                "/v1/chat/completions",
                json=request.model_dump(),
                headers={"Authorization": "Bearer chat-key"},
            )
            assert response.status_code == 200
            if streaming:
                assert "[DONE]" in response.text and '"completion_tokens":3' in response.text
            else:
                assert response.json()["choices"][0]["message"]["content"] == "profit €"
                assert response.json()["usage"] == {"completion_tokens": 3}
            assert app.state.serving.admission.active == 0
        line, sent = backend.requests[0]
        assert line == "POST /v1/chat/completions HTTP/1.1"
        assert sent == {
            "model": "native-model",
            "messages": request.model_dump()["messages"],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": True,
            "n": 1,
            "stream_options": {"include_usage": True},
        }


async def test_native_chat_cancellation_closes_real_upstream_socket() -> None:
    """Stopping an unfinished native chat must close HTTP work without inventing final usage."""
    backend = Backend(
        event({"choices": [{"index": 0, "delta": {"content": "partial"}}]}), hold=True
    )
    async with serving_backend(backend) as url:
        engine = OpenAICompletionEngine(url)
        iterator = engine.stream(chat_request().to_inference())
        try:
            first = await anext(iterator)
            assert first.text == "partial" and first.generated_tokens == 0
            pending = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            async with asyncio.timeout(2):
                await backend.closed.wait()
        finally:
            await iterator.aclose()
            await engine.close()


async def test_native_chat_missing_usage_fails_over_http() -> None:
    """DONE alone cannot convert a native chat with absent accounting into success."""
    backend = Backend(reply(include_usage=False))
    async with serving_backend(backend) as url:
        engine = OpenAICompletionEngine(url)
        try:
            with pytest.raises(EngineProtocolError, match="usage"):
                _ = [token async for token in engine.stream(chat_request().to_inference())]
        finally:
            await engine.close()


async def test_ray_http_roundtrip_retains_messages_for_native_backend() -> None:
    """The Ray ingress/actor contract preserves role payloads before the external HTTP hop."""
    backend = Backend(reply())
    async with serving_backend(backend) as url:
        native = OpenAICompletionEngine(url)

        async def routed(request: httpx.Request) -> httpx.Response:
            """Revalidate serialized ingress just as Ray's HTTP and worker boundaries do."""
            typed = InferenceRequest.model_validate_json(request.content)
            payload = typed.model_dump()
            worker_request = InferenceRequest.model_validate(payload)
            tokens = [token async for token in native.stream(worker_request)]
            wire = b"".join(
                json.dumps({"replica_id": "fixture-a", "token": token.model_dump()}).encode()
                + b"\n"
                for token in tokens
            )
            return httpx.Response(
                200, content=wire, headers={"content-type": "application/x-ndjson"}
            )

        ray = RayHTTPEngine("http://routing/", transport=httpx.MockTransport(routed))
        try:
            tokens = [token async for token in ray.stream(chat_request().to_inference())]
            assert sum(token.generated_tokens for token in tokens) == 3
            assert backend.requests[0][1]["messages"] == chat_request().model_dump()["messages"]
        finally:
            await ray.close()
            await native.close()


@pytest.mark.parametrize(
    "messages", [[], [{"role": "tool", "content": "x"}], [{"role": "user", "content": "x"}] * 65]
)
def test_internal_chat_contract_rejects_invalid_messages(messages: list[dict[str, str]]) -> None:
    """The internal Ray request cannot widen role or population bounds beyond public chat."""
    with pytest.raises(ValueError):
        InferenceRequest(prompt="x", messages=messages)  # type: ignore[arg-type]


def test_chat_prompt_coherence_and_aggregate_budget() -> None:
    """A short scheduling prompt cannot hide a much larger native message payload."""
    request = chat_request().to_inference()
    assert InferenceRequest.model_validate_json(request.model_dump_json()) == request
    assert request.messages is not None and request.prompt.startswith("system: ")
    with pytest.raises(ValueError, match="reference prompt"):
        InferenceRequest(prompt="short", messages=request.messages)
    oversized = ChatRequest(messages=[ChatMessage(role="user", content="x" * 20000)] * 2)
    with pytest.raises(ValueError):
        oversized.to_inference()


@pytest.mark.parametrize("content", ["😀" * 20000, "\x01" * 12000], ids=["unicode", "json-escape"])
async def test_chat_rejects_expanded_internal_bytes_before_engine(content: str) -> None:
    """Unicode and JSON escapes cannot fit ingress yet overflow the duplicated Ray payload."""
    request = ChatRequest(messages=[ChatMessage(role="user", content=content)])
    assert len(request.model_dump_json().encode()) < 131072
    with pytest.raises(ValueError, match="routing byte budget"):
        request.to_inference()
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json=request.model_dump())
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "CONTEXT_TOO_LARGE"
        assert app.state.serving.admission.active == 0


def test_bounded_unicode_chat_serializes_below_ray_cap() -> None:
    """Accepted multibyte payloads leave headroom for request identity and remaining deadlines."""
    request = ChatRequest(messages=[ChatMessage(role="user", content="😀" * 10000)]).to_inference()
    assert len(request.model_dump_json().encode()) < 131072
    assert request.messages is not None and request.messages[0].content == "😀" * 10000
