"""Runtime readiness uses strict completion framing and retains failed offered observations."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from finserve.registry.runtime_probe import completion_probe

BODY = (
    'data: {"choices":[{"index":0,"text":"hello","finish_reason":"length"}]}\n\n'
    'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
)


@pytest.mark.parametrize(
    "defect", ["none", "content_type", "finish", "usage", "done", "duplicate", "status", "created"]
)
async def test_strict_probe_protocol(defect: str) -> None:
    """A live-looking HTTP response cannot replace required finish, usage and DONE observations."""
    body = BODY
    if defect == "finish":
        body = body.replace('"finish_reason":"length"', '"finish_reason":null')
    elif defect == "usage":
        body = body.replace('data: {"choices":[],"usage":{"completion_tokens":1}}\n\n', "")
    elif defect == "done":
        body = body.replace("data: [DONE]\n\n", "")
    elif defect == "duplicate":
        body = body.replace(
            "data: [DONE]", 'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]'
        )
    response = httpx.Response(
        503 if defect == "status" else (201 if defect == "created" else 200),
        content=body,
        headers={}
        if defect == "content_type"
        else {"content-type": "text/event-stream; charset=utf-8"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        row = await completion_probe(client, "http://fixture/v1", "fixture", 1)
    assert row.offered and row.send_s is not None and row.success == (defect == "none")
    if defect == "none":
        assert (
            row.output == "hello" and row.generated_tokens == 1 and row.first_content_s is not None
        )
    else:
        assert row.error is not None


@pytest.mark.parametrize("cancel", [False, True])
async def test_probe_interruption_retains_partial_content(cancel: bool) -> None:
    """A caller cancellation and a timeout both preserve content already observed on the wire."""
    arrived = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        """Wait indefinitely after one strictly shaped nonterminal content event."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """Offer visible content before exercising the deadline or caller cancellation."""
            yield b'data: {"choices":[{"index":0,"text":"partial","finish_reason":null}]}\n\n'
            arrived.set()
            await asyncio.Event().wait()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, stream=Stream(), headers={"content-type": "text/event-stream"}
            )
        )
    ) as client:
        task = asyncio.create_task(completion_probe(client, "http://fixture/v1", "fixture", 0.05))
        await asyncio.wait_for(arrived.wait(), 1)
        if cancel:
            task.cancel()
        row = await task
    assert not row.success and row.output == "partial"
    assert row.error == ("CancelledError" if cancel else "TimeoutError")


async def test_compressed_probe_rejected_before_reading() -> None:
    """The readiness protocol disallows compressed streams before allocating decoded content."""
    reads = 0

    class Stream(httpx.AsyncByteStream):
        """An encoded response must not be iterated when its framing contract is rejected."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """Count a forbidden read independently of the reported readiness failure."""
            nonlocal reads
            reads += 1
            yield b"invalid compressed bytes"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                stream=Stream(),
                headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
            )
        )
    ) as client:
        row = await completion_probe(client, "http://fixture/v1", "fixture", 1)
    assert reads == 0 and not row.success and row.error == "EngineProtocolError"
