"""Exercise failures in ASGI send, beyond the buffered HTTPX test transport."""

import asyncio
import json

import pytest
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from finserve.gateway.app import create_app
from finserve.gateway.body_limit import BodyLimit


def http_scope() -> Scope:
    """ASGI 2.4 reports disconnected writes through send rather than a receive task."""
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1111),
        "server": ("127.0.0.1", 80),
    }


@pytest.mark.parametrize("body_index", [0, 1, 2])
async def test_disconnected_send_releases_lease(body_index: int) -> None:
    """Header, first-token and midstream write failures must return capacity immediately."""
    app = create_app(max_concurrency=1)
    body = json.dumps({"prompt": "hello", "max_tokens": 3}).encode()
    received = False
    bodies_sent = 0

    async def receive() -> Message:
        """No synthetic disconnect is supplied: the failing write must trigger cleanup."""
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        """Fail at a selected transport boundary while the generator owns live state."""
        nonlocal bodies_sent
        if message["type"] == "http.response.body":
            bodies_sent += 1
        if (
            body_index == 0 and message["type"] == "http.response.start"
        ) or (body_index > 0 and bodies_sent == body_index):
            raise OSError("connection lost")

    with pytest.raises(ClientDisconnect):
        await app(http_scope(), receive, send)
    assert app.state.serving.admission.active == 0
    assert app.state.serving.admission.acquire()
    app.state.serving.admission.release()


async def test_chunked_body_limit_without_content_length() -> None:
    """The actual byte cap must reject chunked bodies before JSON parsing or admission."""
    called = False
    sent: list[Message] = []
    chunks = iter([b"1234", b"56789"])

    async def downstream(scope: Scope, receive: object, send: object) -> None:
        """An oversized request must never reach the downstream application."""
        nonlocal called
        called = True

    async def receive() -> Message:
        """Split overflow across events so the cumulative cap is exercised."""
        return {"type": "http.request", "body": next(chunks), "more_body": True}

    async def send(message: Message) -> None:
        """Retain the public response to verify rejection without inspecting internals."""
        sent.append(message)

    await BodyLimit(downstream, max_bytes=8)(http_scope(), receive, send)
    assert not called
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "BODY_TOO_LARGE"
