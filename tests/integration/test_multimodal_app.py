"""Main gateway integration keeps image admission separate and closes every owned resource."""

import base64
import random
import struct
import zlib
from collections.abc import AsyncGenerator

import httpx
import pytest

from finserve.contracts.inference import EngineToken
from finserve.contracts.vision import MAX_VISION_BODY_BYTES, VisionRequest
from finserve.engines.fixture import FixtureEngine
from finserve.gateway.app import create_app, from_env
from finserve.multimodal.images import PNG_SIGNATURE, PreparedImage

pytest.importorskip("PIL")


def large_png() -> str:
    """Incompressible bounded RGB pixels exercise the distinct image upload envelope."""

    def chunk(kind: bytes, value: bytes) -> bytes:
        """CRC-valid test bytes pass the same parser as real uploaded images."""
        return (
            struct.pack(">I", len(value))
            + kind
            + value
            + struct.pack(">I", zlib.crc32(kind + value))
        )

    generator = random.Random(9)
    rows = b"".join(b"\0" + generator.randbytes(256 * 3) for _ in range(256))
    data = (
        PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(data).decode()


class VisionFixture:
    """The fixture validates integration and ownership, never pretrained model quality."""

    def __init__(self, *, fail_close: bool = False) -> None:
        """Record both request eligibility and lifespan cleanup independently."""
        self.calls, self.closed, self.fail_close = 0, False, fail_close

    async def stream(
        self, request: VisionRequest, image: PreparedImage
    ) -> AsyncGenerator[EngineToken]:
        """Only a decoded bounded image may reach this engine interface."""
        assert image.width == image.height == 256
        self.calls += 1
        yield EngineToken(text="fixture", generated_tokens=0)
        yield EngineToken(text="", generated_tokens=1, finish_reason="stop")

    async def close(self) -> None:
        """A simulated close failure must not suppress another resource's cleanup."""
        self.closed = True
        if self.fail_close:
            raise RuntimeError("fixture close failure")


class TextFixture(FixtureEngine):
    """Expose text lifecycle cleanup without altering text inference behavior."""

    def __init__(self) -> None:
        """Initialize the existing generator and an independent closure observation."""
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        """Record that shared gateway cleanup reaches the text resource."""
        self.closed = True


async def test_main_gateway_supports_large_bounded_images_and_preserves_text_cap() -> None:
    """The image route owns pre-body auth/admission while text remains limited to128KiB."""
    engine = VisionFixture()
    app = create_app(vision_engine=engine, vision_model="image-model", api_key="integration-key")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        body = {
            "model": "image-model",
            "prompt": "Describe",
            "image_png_base64": large_png(),
            "stream": False,
        }
        assert len(str(body)) > 131072
        assert (await client.post("/v1/vision/completions", json=body)).status_code == 401
        client.headers["Authorization"] = "Bearer integration-key"
        result = await client.post("/v1/vision/completions", json=body)
        assert result.status_code == 200 and result.json()["usage"]["completion_tokens"] == 1
        assert engine.calls == 1 and app.state.vision.admission.active == 0
        assert (await client.post("/v1/completions", json=body)).status_code == 413
        assert (
            await client.post("/v1/vision/completions", content=b"x" * (MAX_VISION_BODY_BYTES + 1))
        ).status_code == 413
        assert app.state.vision.admission.active == 0
    assert engine.closed


async def test_multimodal_close_failure_still_closes_text() -> None:
    """Every owned transport closes even when one modality reports a shutdown failure."""
    text, vision = TextFixture(), VisionFixture(fail_close=True)
    app = create_app(text, vision_engine=vision, api_key="integration-key")
    with pytest.raises(RuntimeError, match="fixture close failure"):
        async with app.router.lifespan_context(app):
            pass
    assert text.closed and vision.closed
    with pytest.raises(ValueError):
        create_app(vision_engine=VisionFixture())


async def test_process_configuration_registers_only_explicit_vision_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image capability is enabled by configured engine identity and never by the text backend."""
    monkeypatch.setenv("FINSERVE_ENGINE", "fixture")
    for key in ("FINSERVE_REDIS_URL", "FINSERVE_VISUAL_GRPC_TARGET", "FINSERVE_TRACE_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FINSERVE_API_KEY", "integration-key")
    monkeypatch.setenv("FINSERVE_VISION_ENGINE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("FINSERVE_VISION_MODEL", "declared-image-model")
    app = from_env()
    async with app.router.lifespan_context(app):
        assert app.state.vision.model == "declared-image-model"
