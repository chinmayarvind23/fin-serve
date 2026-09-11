"""ASGI route checks cover image eligibility, slot ownership and transport failures."""

import asyncio
import base64
import json
import struct
import threading
import zlib
from collections.abc import AsyncGenerator

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import ClientDisconnect, Request
from starlette.types import Message, Scope

from finserve.contracts.inference import EngineToken
from finserve.contracts.vision import MAX_VISION_BODY_BYTES, VISION_MODEL, VisionRequest
from finserve.gateway.vision import VisionLease, VisionResponse, register_vision_routes
from finserve.multimodal.images import PNG_SIGNATURE, PreparedImage, prepare_png

pytest.importorskip("PIL")


def image_bytes() -> bytes:
    """One RGB pixel is sufficient for transport ownership tests, not model quality tests."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        """Supply real PNG checksums rather than bypassing validation in route tests."""
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


def payload(**updates: object) -> dict[str, object]:
    """Use a bounded genuine image and vary only each tested failure condition."""
    result: dict[str, object] = {
        "prompt": "What color?",
        "image_png_base64": base64.b64encode(image_bytes()).decode(),
    }
    result.update(updates)
    return result


class Fixture:
    """CPU fixture emits compliant authoritative accounting, with optional visible-output stall."""

    def __init__(self, stall: bool = False) -> None:
        """Record admission and cleanup separately from response text."""
        self.stall, self.calls, self.closed = stall, 0, False
        self.started, self.drained = asyncio.Event(), asyncio.Event()

    async def stream(
        self, request: VisionRequest, image: PreparedImage
    ) -> AsyncGenerator[EngineToken]:
        """A cancellation path sets drained only after the iterator really closes."""
        self.calls += 1
        self.started.set()
        try:
            yield EngineToken(text="red", generated_tokens=0)
            if self.stall:
                await asyncio.Event().wait()
            yield EngineToken(text="", generated_tokens=2, finish_reason="stop")
        finally:
            self.drained.set()

    async def close(self) -> None:
        """Let lifecycle tests observe explicit engine ownership."""
        self.closed = True


async def test_route_auth_model_caps_and_streaming() -> None:
    """Rejected requests never invoke an image engine; success preserves final usage."""
    app, engine = FastAPI(), Fixture()
    serving = register_vision_routes(app, engine, VISION_MODEL, "secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.post("/v1/vision/completions", json=payload())).status_code == 401
        client.headers["authorization"] = "Bearer secret"
        assert (
            await client.post("/v1/vision/completions", json=payload(model="text-only"))
        ).status_code == 404
        assert (
            await client.post(
                "/v1/vision/completions", json=payload(image_png_base64="https://example.org")
            )
        ).status_code == 422
        assert (
            await client.post("/v1/vision/completions", content=b"x" * (MAX_VISION_BODY_BYTES + 1))
        ).status_code == 413
        assert engine.calls == 0
        response = await client.post("/v1/vision/completions", json=payload())
        assert response.status_code == 200
        assert response.text.endswith("data: [DONE]\n\n")
        assert '"completion_tokens":2' in response.text
        assert '"finish_reason":"stop"' in response.text
        aggregate = await client.post("/v1/vision/completions", json=payload(stream=False))
        assert aggregate.json()["choices"][0]["message"]["content"] == "red"
        assert aggregate.json()["usage"]["completion_tokens"] == 2
    assert serving.admission.active == 0
    await serving.close()
    assert engine.closed


async def test_timeout_and_capacity_rejection() -> None:
    """An occupied image slot rejects immediately and a timeout closes its upstream iterator."""
    app, engine = FastAPI(), Fixture(stall=True)
    serving = register_vision_routes(app, engine, VISION_MODEL)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = asyncio.create_task(
            client.post("/v1/vision/completions", json=payload(timeout_seconds=0.1))
        )
        await engine.started.wait()
        assert (await client.post("/v1/vision/completions", json=payload())).status_code == 429
        response = await first
        assert '"code":"DEADLINE_EXCEEDED"' in response.text
        assert "[DONE]" not in response.text
    assert engine.drained.is_set()
    assert serving.admission.active == 0


@pytest.mark.parametrize("fail_body", [False, True])
async def test_asgi_send_failure_releases_slot(fail_body: bool) -> None:
    """Header and partial-body transport failures must close the owned iterator and lease."""
    app, engine = FastAPI(), Fixture(stall=True)
    serving = register_vision_routes(app, engine, VISION_MODEL)
    assert serving.admission.acquire()
    request = VisionRequest.model_validate(payload())
    iterator = serving.events(
        request, prepare_png(image_bytes()), asyncio.get_running_loop().time() + 5
    )
    response = VisionResponse(
        iterator,
        VisionLease(serving.admission),
        request.request_id,
        asyncio.get_running_loop().time() + 5,
    )
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [],
        "asgi": {"version": "3.0", "spec_version": "2.4"},
    }

    async def receive() -> Message:
        """No disconnect race is needed; the outbound socket itself fails."""
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        """Fail at a selected ownership transition rather than relying on GC cleanup."""
        if not fail_body or message["type"] == "http.response.body":
            raise OSError("closed peer")

    with pytest.raises(ClientDisconnect):
        await response(scope, receive, send)
    assert serving.admission.active == 0
    assert engine.drained.is_set() == fail_body


async def test_cancel_preprocessing_retains_capacity_until_native_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated cancellation cannot free capacity while CPU decoding still owns native work."""
    app, engine = FastAPI(), Fixture()
    serving = register_vision_routes(app, engine, VISION_MODEL)
    entered, release = threading.Event(), threading.Event()

    def blocked(raw: bytes) -> PreparedImage:
        """Create a deterministic native-thread boundary without running expensive computation."""
        entered.set()
        release.wait(5)
        return prepare_png(raw)

    monkeypatch.setattr("finserve.gateway.vision.prepare_png", blocked)
    body = json.dumps(payload()).encode()

    async def receive() -> Message:
        """Deliver the complete bounded HTTP request exactly once."""
        return {"type": "http.request", "body": body, "more_body": False}

    scope: Scope = {"type": "http", "method": "POST", "path": "/", "headers": []}
    task = asyncio.create_task(serving.infer(Request(scope, receive)))
    try:
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert serving.admission.active == 1
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert serving.admission.active == 0
    assert engine.calls == 0


async def test_slow_body_consumes_inference_deadline() -> None:
    """Body transfer time belongs to trusted ingress receipt, not a fresh decoder budget."""
    app, engine = FastAPI(), Fixture()
    serving = register_vision_routes(app, engine, VISION_MODEL)
    body = json.dumps(payload(timeout_seconds=0.01)).encode()

    async def receive() -> Message:
        """Delay a valid body beyond its declared inference budget."""
        await asyncio.sleep(0.03)
        return {"type": "http.request", "body": body, "more_body": False}

    response = await serving.infer(Request({"type": "http", "headers": []}, receive))
    assert response.status_code == 504
    assert engine.calls == 0
    assert serving.admission.active == 0


async def test_slow_upload_holds_admission_before_buffering() -> None:
    """Direct authenticated FastAPI clients cannot accumulate unbounded request bodies."""
    app, engine = FastAPI(), Fixture()
    serving = register_vision_routes(app, engine, VISION_MODEL)
    entered = asyncio.Event()

    async def receive() -> Message:
        """Hold the initial client inside body transfer with a slot already reserved."""
        entered.set()
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def forbidden_receive() -> Message:
        """An overloaded request must never read or allocate its body."""
        raise AssertionError("overloaded request read its body")

    scope: Scope = {"type": "http", "headers": []}
    first = asyncio.create_task(serving.infer(Request(scope, receive)))
    await entered.wait()
    assert serving.admission.active == 1
    assert (await serving.infer(Request(scope, forbidden_receive))).status_code == 429
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert serving.admission.active == 0


async def test_stalled_downstream_send_closes_generation() -> None:
    """A client that stops reading must not retain the only image slot indefinitely."""
    app, engine = FastAPI(), Fixture(stall=True)
    serving = register_vision_routes(app, engine, VISION_MODEL)
    assert serving.admission.acquire()
    request = VisionRequest.model_validate(payload())
    deadline = asyncio.get_running_loop().time() + 0.05
    response = VisionResponse(
        serving.events(request, prepare_png(image_bytes()), deadline),
        VisionLease(serving.admission),
        request.request_id,
        deadline,
    )

    async def receive() -> Message:
        """Represent a connected peer while the outbound body is blocked."""
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        """Allow headers, then emulate downstream backpressure forever."""
        if message["type"] == "http.response.body":
            await asyncio.Event().wait()

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "headers": [],
        "asgi": {"version": "3.0", "spec_version": "2.4"},
    }
    with pytest.raises(ClientDisconnect):
        await asyncio.wait_for(response(scope, receive, send), 1)
    assert engine.drained.is_set()
    assert serving.admission.active == 0
