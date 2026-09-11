"""The CPU stage preserves canonical pixels and authenticates its separate HTTP boundary."""

import asyncio
import hashlib
import json
import struct
import threading
import zlib
from pathlib import Path

import httpx
import pytest

from finserve.contracts.vision import MAX_PNG_BYTES
from finserve.multimodal.images import PreparedImage, prepare_png
from finserve.multimodal.preprocess_http import PreprocessorClient, create_preprocessor_app
from finserve.multimodal.vision_benchmark import color_png, run

pytest.importorskip("PIL")
KEY = "test-preprocessing-internal-key-12345"


@pytest.mark.parametrize("mode", ["metadata", "wrong_media", "impossible_timing"])
async def test_stage_requires_canonical_media_and_possible_timing(mode: str) -> None:
    """A matching SHA alone cannot validate decoder safety or cross-process timing claims."""
    raw = color_png((255, 0, 0), 1)
    result = prepare_png(raw).png
    if mode == "metadata":
        payload = b"ignored text"
        chunk = struct.pack(">I", len(payload)) + b"tEXt" + payload
        chunk += struct.pack(">I", zlib.crc32(b"tEXt" + payload))
        result = result[:33] + chunk + result[33:]
    headers = {
        "content-type": "application/octet-stream" if mode == "wrong_media" else "image/png",
        "x-png-sha256": hashlib.sha256(result).hexdigest(),
        "x-source-sha256": hashlib.sha256(raw).hexdigest(),
        "x-preprocess-seconds": "1" if mode == "impossible_timing" else "0",
    }
    client = PreprocessorClient(
        "http://stage",
        KEY,
        httpx.MockTransport(lambda _: httpx.Response(200, headers=headers, content=result)),
    )
    try:
        with pytest.raises(RuntimeError):
            await client.prepare(raw)
    finally:
        await client.close()


async def test_benchmark_profile_mismatch_retains_declaration(tmp_path: Path) -> None:
    """A supplied model declaration cannot silently disagree with the actual request contract."""
    profile = tmp_path / "wrong-profile.json"
    profile.write_text('{"model":"different","model_revision":"untrusted"}')
    output = tmp_path / "profile-failed"
    with pytest.raises(ValueError, match="profile model"):
        await run(output, "http://unused", "http://unused", 1, profile)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert (output / "engine-profile.json").is_file()


def test_preprocessor_configuration_fails_closed() -> None:
    """Weak worker credentials and caller-style endpoint tricks fail at construction."""
    with pytest.raises(ValueError):
        create_preprocessor_app("short")
    for url in ("file:///tmp/png", "http://user:password@stage", "http://stage?next=other"):
        with pytest.raises(ValueError):
            PreprocessorClient(url, KEY)


@pytest.mark.parametrize("mode", ["oversized", "status", "invalid_timing"])
async def test_preprocessor_response_limits(mode: str) -> None:
    """Bound remote bytes and reject unsupported status/timing before trusting a stage result."""
    raw = color_png((255, 0, 0), 1)
    headers = {
        "content-type": "image/png",
        "x-png-sha256": hashlib.sha256(raw).hexdigest(),
        "x-source-sha256": hashlib.sha256(raw).hexdigest(),
        "x-preprocess-seconds": "nan",
    }
    body = b"x" * (MAX_PNG_BYTES + 1) if mode == "oversized" else raw
    client = PreprocessorClient(
        "http://stage",
        KEY,
        httpx.MockTransport(
            lambda _: httpx.Response(
                503 if mode == "status" else 200, headers=headers, content=body
            )
        ),
    )
    try:
        with pytest.raises(RuntimeError):
            await client.prepare(raw)
        with pytest.raises(ValueError):
            await client.prepare(b"x" * (MAX_PNG_BYTES + 1))
    finally:
        await client.close()


async def test_stage_cancellation_retains_slot_until_thread_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP cancellation cannot over-admit another decode while native work remains active."""
    entered, release = threading.Event(), threading.Event()

    def blocked(raw: bytes) -> PreparedImage:
        """Hold a real worker thread at a deterministic cancellation boundary."""
        entered.set()
        release.wait(5)
        return prepare_png(raw)

    monkeypatch.setattr("finserve.multimodal.preprocess_http.prepare_png", blocked)
    app = create_preprocessor_app(KEY)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://stage",
        headers={"authorization": "Bearer " + KEY},
    ) as http:
        task = asyncio.create_task(http.post("/internal/prepare", content=color_png((1, 2, 3), 1)))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            assert (await http.post("/internal/prepare", content=b"unused")).status_code == 429
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (
            await http.post("/internal/prepare", content=color_png((1, 2, 3), 1))
        ).status_code == 200


async def test_real_stage_contract_matches_local_canonical_image() -> None:
    """Exercise HTTP serialization and ASGI decoding with genuine PNG bytes and stage hashes."""
    app = create_preprocessor_app(KEY)
    client = PreprocessorClient("http://stage", KEY, httpx.ASGITransport(app=app))
    try:
        raw = color_png((255, 0, 0), 256)
        image, elapsed, worker = await client.prepare(raw)
        assert image == prepare_png(raw)
        assert elapsed >= worker >= 0
        assert (image.width, image.height) == (256, 256)
    finally:
        await client.close()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://stage"
    ) as http:
        assert (await http.post("/internal/prepare", content=raw)).status_code == 401
        http.headers["authorization"] = "Bearer " + KEY
        assert (await http.post("/internal/prepare", content=b"invalid")).status_code == 422
        assert (
            await http.post("/internal/prepare", content=b"x" * (MAX_PNG_BYTES + 1))
        ).status_code == 413


@pytest.mark.parametrize(
    "headers,body",
    [
        ({}, b"bad"),
        ({"x-png-sha256": "wrong", "x-source-sha256": "wrong"}, b"bad"),
        ({"content-encoding": "gzip"}, b""),
    ],
)
async def test_stage_rejects_untrusted_response(headers: dict[str, str], body: bytes) -> None:
    """A remote stage cannot silently substitute image bytes or defeat response budgeting."""
    client = PreprocessorClient(
        "http://stage",
        KEY,
        httpx.MockTransport(lambda _: httpx.Response(200, headers=headers, content=body)),
    )
    try:
        with pytest.raises(RuntimeError):
            await client.prepare(color_png((1, 2, 3), 1))
    finally:
        await client.close()


async def test_failed_benchmark_retains_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable real endpoint preserves failed lifecycle evidence without any model work."""
    monkeypatch.setenv("FINSERVE_PREPROCESSOR_KEY", KEY)
    output = tmp_path / "failed-run"
    with pytest.raises(RuntimeError):
        await run(output, "http://127.0.0.1:1/v1", "http://127.0.0.1:1", 1)
    assert '"status": "failed"' in (output / "manifest.json").read_text()
    assert '"error_type": "EngineUnavailableError"' in (output / "manifest.json").read_text()
    assert (output / "red-256-input.png").is_file()
    warmup = json.loads((output / "red-256-warmup.json").read_text())
    assert warmup["phase"] == "warmup" and warmup["status"] == "failed"
    assert warmup["fixture"] == "red-256"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["engine_url"] == "http://127.0.0.1:1/v1"
    for relative, sha in manifest["source_sha256"].items():
        archived = output / ("source__" + relative.replace("/", "__"))
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == sha
    assert len(manifest["repository"]["git_revision"]) == 40


async def test_benchmark_repo_path_rejected() -> None:
    """Evidence destination validation happens before provisioning any clients or files."""
    with pytest.raises(ValueError, match="outside"):
        await run(
            Path(__file__).parents[2] / "forbidden-vision-output",
            "http://unused",
            "http://unused",
            1,
        )
    with pytest.raises(ValueError, match="Repetitions"):
        await run(Path("unused"), "http://unused", "http://unused", 0)


async def test_interrupted_benchmark_closes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation cannot leave evidence claiming the run is still actively measuring."""
    import finserve.multimodal.vision_benchmark as benchmark

    async def interrupted(*args: object, **kwargs: object) -> dict[str, object]:
        """Interrupt the owned workload after manifest creation without invoking a model."""
        raise asyncio.CancelledError

    monkeypatch.setenv("FINSERVE_PREPROCESSOR_KEY", KEY)
    monkeypatch.setattr(benchmark, "sample", interrupted)
    output = tmp_path / "interrupted-run"
    with pytest.raises(asyncio.CancelledError):
        await run(output, "http://unused", "http://unused", 1)
    assert '"status": "interrupted"' in (output / "manifest.json").read_text()
