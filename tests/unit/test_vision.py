"""Adversarial media and real chat-wire contracts run without loading any model."""

import asyncio
import base64
import json
import struct
import zlib
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import ValidationError

from finserve.contracts.vision import MAX_PNG_BYTES, VisionRequest
from finserve.engines.openai_adapter import EngineProtocolError, EngineUnavailableError
from finserve.engines.vision_openai import ChatState, OpenAIVisionEngine
from finserve.multimodal.images import (
    PNG_SIGNATURE,
    bounded_png,
    decode_inline_png,
    prepare_png,
)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    """Create exact CRC-valid fixtures so each rejection targets its intended invariant."""
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png(width: int = 2, height: int = 2, color: int = 2, extra: bytes = b"") -> bytes:
    """Build small genuine PNGs using the standard library, independent of optional Pillow."""
    channels = 4 if color == 6 else 3
    header = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    pixels = b"".join(b"\x00" + bytes([0]) * width * channels for _ in range(height))
    return (
        PNG_SIGNATURE
        + png_chunk(b"IHDR", header)
        + extra
        + png_chunk(b"IDAT", zlib.compress(pixels))
        + png_chunk(b"IEND", b"")
    )


def request(**updates: object) -> VisionRequest:
    """Keep image fixture identity stable while varying one contract constraint."""
    data: dict[str, object] = {
        "prompt": "What color?",
        "image_png_base64": base64.b64encode(png()).decode(),
    }
    data.update(updates)
    return VisionRequest.model_validate(data)


@pytest.mark.parametrize(
    "value", ["https://example.com/image.png", "data:image/png;base64,AA==", "AA==\n", "AB=="]
)
def test_base64_rejects_urls_whitespace_and_noncanonical(value: str) -> None:
    """Destination syntax never becomes an upstream network fetch."""
    with pytest.raises(ValueError):
        decode_inline_png(value)


@pytest.mark.parametrize(
    "updates",
    [
        {"max_tokens": True},
        {"max_tokens": 257},
        {"temperature": float("nan")},
        {"images": []},
        {"modality": "text"},
    ],
)
def test_vision_contract_bounds(updates: dict[str, object]) -> None:
    """Unsupported modalities, batch/options and unbounded generation fail before admission."""
    with pytest.raises(ValidationError):
        request(**updates)


def test_valid_png_is_metadata_free_and_alpha_white() -> None:
    """The same pixel input yields byte-identical canonical PNG across repeated stage calls."""
    pytest.importorskip("PIL")
    raw = png(extra=png_chunk(b"zTXt", b"malicious metadata need not decompress"))
    first, second = prepare_png(raw), prepare_png(png())
    assert first.png == second.png
    assert first.sha256 == second.sha256
    assert first.source_sha256 != second.source_sha256
    assert prepare_png(first.png).sha256 == first.sha256
    assert (first.width, first.height) == (2, 2)
    assert first.data_url().startswith("data:image/png;base64,")
    assert prepare_png(png(color=6)).sha256 != first.sha256


@pytest.mark.parametrize(
    "raw",
    [
        b"bad",
        png(513),
        png(color=3),
        png()[:-1],
        png() + b"junk",
        png(extra=png_chunk(b"acTL", b"12345678")),
        png()[:20] + b"x" + png()[21:],
    ],
)
def test_png_structure_fails_before_decoder(raw: bytes) -> None:
    """Allocation, animation, truncation and checksum boundaries do not depend on Pillow."""
    with pytest.raises(ValueError):
        bounded_png(raw)


def test_png_byte_budget_and_decode_failure() -> None:
    """Encoded and pixel decoding corruption have separate bounded failure paths."""
    with pytest.raises(ValueError):
        decode_inline_png(base64.b64encode(b"x" * (MAX_PNG_BYTES + 1)).decode())
    pytest.importorskip("PIL")
    malformed = PNG_SIGNATURE + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
    malformed += png_chunk(b"IDAT", b"not zlib") + png_chunk(b"IEND", b"")
    with pytest.raises(ValueError):
        prepare_png(malformed)


def event(content: str | None = None, finish: str | None = None) -> str:
    """Produce actual chat delta frames, never raw completion text frames."""
    return json.dumps(
        {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]}
    )


def frames() -> bytes:
    """Include role-only, content, empty finish and usage-only frames from vLLM chat SSE."""
    values = [
        json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
        event("red"),
        event(finish="stop"),
        json.dumps({"choices": [], "usage": {"completion_tokens": 2}}),
        "[DONE]",
    ]
    return "".join("data: " + value + "\r\n\r\n" for value in values).encode()


class BytesStream(httpx.AsyncByteStream):
    """Arbitrary byte splitting and cancellable stalls exercise transport ownership."""

    def __init__(self, value: bytes, *, stall: bool = False) -> None:
        """Record closure independently from generation success."""
        self.value, self.stall, self.closed = value, stall, False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield small chunks, then optionally block after visible text."""
        for offset in range(0, len(self.value), 7):
            yield self.value[offset : offset + 7]
        if self.stall:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        """An explicit close is required after success, cancellation and malformed streams."""
        self.closed = True


async def test_actual_chat_payload_and_final_usage() -> None:
    """Verify the image goes in inline chat content, with token count independent of frames."""
    pytest.importorskip("PIL")
    stream = BytesStream(frames())
    captured: list[httpx.Request] = []

    def handle(outbound: httpx.Request) -> httpx.Response:
        """Inspect the real serialized HTTP request before delivering scripted SSE bytes."""
        captured.append(outbound)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    engine = OpenAIVisionEngine(
        "http://engine/v1", api_key="private-key", transport=httpx.MockTransport(handle)
    )
    try:
        tokens = [token async for token in engine.stream(request(), prepare_png(png()))]
        assert [token.text for token in tokens] == ["red", ""]
        assert [token.generated_tokens for token in tokens] == [0, 2]
        assert tokens[-1].finish_reason == "stop"
        body = json.loads(captured[0].content)
        assert captured[0].url.path == "/v1/chat/completions"
        assert captured[0].headers["authorization"] == "Bearer private-key"
        assert body["messages"][0]["content"][0]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        assert body["stream_options"] == {"include_usage": True}
        assert stream.closed
    finally:
        await engine.close()


@pytest.mark.parametrize(
    "bad",
    [
        '{"choices":[{"index":0,"delta":{"tool_calls":[{}]}}]}',
        '{"choices":[{"index":0,"delta":{"role":"user"}}]}',
        '{"choices":[{"index":0,"text":"raw completion"}]}',
        '{"choices":[],"usage":{"completion_tokens":2}}',
        '{"choices":[{"index":1,"delta":{}}]}',
    ],
)
def test_chat_protocol_rejects_unsupported_shapes(bad: str) -> None:
    """Extensions cannot bypass the single-choice text content and accounting contract."""
    with pytest.raises(EngineProtocolError):
        ChatState(10).consume(bad)


@pytest.mark.parametrize("limit", [True, 0, -1, float("nan"), float("inf"), 4 * 1024 * 1024 + 1])
def test_response_budget_is_finite_bounded_integer(limit: int) -> None:
    """Misconfigured caps must not disable the byte safety boundary."""
    with pytest.raises(ValueError):
        OpenAIVisionEngine("http://engine/v1", maximum_response_bytes=limit)


def test_transparency_key_and_duplicate_usage_are_rejected() -> None:
    """Stripping color-key transparency would change pixels; duplicate usage is ambiguous."""
    with pytest.raises(ValueError):
        bounded_png(png(extra=png_chunk(b"tRNS", b"\x00" * 6)))
    state = ChatState(10)
    state.consume(event(finish="stop"))
    with pytest.raises(EngineProtocolError):
        state.consume('{"choices":[],"usage":{"completion_tokens":2,"completion_tokens":3}}')


@pytest.mark.parametrize("cancel", [False, True])
async def test_close_after_partial_text_and_deadline(cancel: bool) -> None:
    """Stopping iteration closes HTTP immediately; native GPU cancellation is not claimed."""
    pytest.importorskip("PIL")
    stream = BytesStream(("data: " + event("red") + "\n\n").encode(), stall=True)
    engine = OpenAIVisionEngine(
        "http://engine/v1",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        ),
    )
    try:
        iterator = engine.stream(request(timeout_seconds=0.03), prepare_png(png()))
        assert (await anext(iterator)).text == "red"
        if cancel:
            await iterator.aclose()
        else:
            with pytest.raises(TimeoutError):
                await anext(iterator)
        assert stream.closed
    finally:
        await engine.close()


@pytest.mark.parametrize(
    "status,headers,error",
    [
        (503, {}, EngineUnavailableError),
        (200, {"content-type": "application/json"}, EngineProtocolError),
        (
            200,
            {"content-type": "text/event-stream", "content-encoding": "gzip"},
            EngineProtocolError,
        ),
    ],
)
async def test_backend_response_boundary(
    status: int, headers: dict[str, str], error: type[Exception]
) -> None:
    """Protocol errors expose static classifications, never backend response bodies."""
    pytest.importorskip("PIL")
    stream = BytesStream(b"")
    engine = OpenAIVisionEngine(
        "http://engine/v1",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, headers=headers, stream=stream)
        ),
    )
    try:
        with pytest.raises(error):
            _ = [token async for token in engine.stream(request(), prepare_png(png()))]
        assert stream.closed
    finally:
        await engine.close()
