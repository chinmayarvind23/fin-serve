"""Model producers must verify fixed source bytes before publishing reusable snapshots."""

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from finserve.contracts.model_assets import (
    ModelFetchSpec,
    ModelManifest,
    SourceFile,
    VerifiedFile,
    safe_asset_path,
)
from finserve.registry.model_assets import (
    MANIFEST_NAME,
    bounded_response,
    fetch_verified_model,
    owned_disk,
    parse_hub_plan,
    plan_hub_snapshot,
    verify_file,
    verify_snapshot,
)


def specification(data: bytes = b"{}", git_blob: bool = False) -> ModelFetchSpec:
    """Use real fixture byte hashes rather than accepting a passed verification flag."""
    digest = (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if git_blob
        else hashlib.sha256(data).hexdigest()
    )
    return ModelFetchSpec(
        repository="fixture/model",
        revision="a" * 40,
        maximum_bytes=1024,
        files=(
            SourceFile(
                path="config.json",
                size_bytes=len(data),
                checksum_kind="git-blob-sha1" if git_blob else "sha256",
                checksum=digest,
            ),
        ),
    )


@pytest.mark.parametrize(
    "path",
    [
        "../config.json",
        "/config.json",
        "config.py",
        "pytorch_model.bin",
        "a/../config.json",
        "a\\config.json",
        "x:config.json",
        "config.json/",
        ".config.json",
        "finserve-model-manifest.json",
        "x?config.json",
        "CON.json",
        "nested./config.json",
    ],
)
def test_unsafe_model_paths_are_rejected(path: str) -> None:
    """No selected source path may escape staging or introduce executable pickle/Python files."""
    with pytest.raises(ValueError):
        safe_asset_path(path)


def test_frozen_file_set_size_and_checksum_contracts() -> None:
    """Manifest order does not change identity; duplicate names or contradictory digests fail."""
    first = specification()
    second = first.files[0].model_copy(update={"path": "tokenizer.json"})
    left = first.model_copy(update={"files": (first.files[0], second)})
    right = first.model_copy(update={"files": (second, first.files[0])})
    assert left.digest() == right.digest()
    for update in (
        {"maximum_bytes": 1},
        {"files": (first.files[0], first.files[0])},
        {"revision": "main"},
    ):
        with pytest.raises(ValueError):
            ModelFetchSpec.model_validate({**first.model_dump(), **update})
    with pytest.raises(ValueError):
        SourceFile.model_validate({**first.files[0].model_dump(), "checksum": "a" * 40})
    with pytest.raises(ValueError):
        ModelManifest(specification=first, files=())
    with pytest.raises(ValueError, match="verified SHA256"):
        VerifiedFile(source=first.files[0], sha256="1" * 64)


@pytest.mark.parametrize("git_blob", [False, True])
async def test_fixed_hub_plan_download_and_idempotent_byte_recheck(
    tmp_path: Path, git_blob: bool
) -> None:
    """Both Hub checksum algorithms yield independently checked SHA256 model manifests."""
    frozen = specification(git_blob=git_blob)
    selected = frozen.files[0]
    calls: list[str] = []

    def response(request: httpx.Request) -> httpx.Response:
        """Serve metadata and matching bytes at the exact frozen revision."""
        calls.append(str(request.url))
        if "/api/models/" in request.url.path:
            sibling: dict[str, object] = {
                "rfilename": selected.path,
                "size": 2,
                "blobId": selected.checksum,
            }
            if not git_blob:
                sibling["lfs"] = {"sha256": selected.checksum}
            return httpx.Response(200, json={"sha": frozen.revision, "siblings": [sibling]})
        assert request.url.path.endswith(frozen.revision + "/config.json")
        return httpx.Response(200, content=b"{}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        planned = await plan_hub_snapshot(
            client, frozen.repository, frozen.revision, (selected.path,), 1024
        )
        assert planned == frozen
        directory, manifest = await fetch_verified_model(client, planned, tmp_path)
        assert manifest.files[0].sha256 == hashlib.sha256(b"{}").hexdigest()
        assert json.loads((directory / MANIFEST_NAME).read_text()) == json.loads(
            manifest.canonical()
        )
        assert await fetch_verified_model(client, planned, tmp_path) == (directory, manifest)
        assert len(calls) == 2
        (directory / selected.path).write_bytes(b"xx")
        with pytest.raises(ValueError, match="checksum"):
            await fetch_verified_model(client, planned, tmp_path)


@pytest.mark.parametrize("body", [b"different", b"x", b"xx"])
@pytest.mark.parametrize("git_blob", [False, True])
async def test_invalid_download_never_publishes_final_model(
    tmp_path: Path, body: bytes, git_blob: bool
) -> None:
    """Oversized, truncated and equal-length wrong content all leave no reusable model directory."""
    frozen = specification(git_blob=git_blob)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    ) as client:
        with pytest.raises(ValueError):
            await fetch_verified_model(client, frozen, tmp_path)
    assert await asyncio.to_thread(lambda: list(tmp_path.iterdir())) == []


async def test_snapshot_manifest_extra_files_and_deadline_are_verified(tmp_path: Path) -> None:
    """Reusing a directory never trusts its filename, a manifest assertion or unbounded hashing."""
    frozen = specification()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"{}"))
    ) as client:
        directory, manifest = await fetch_verified_model(client, frozen, tmp_path)
    with pytest.raises(TimeoutError):
        verify_file(directory / "config.json", frozen.files[0], deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        verify_snapshot(directory, frozen, deadline=time.monotonic() - 1)
    (directory / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        verify_snapshot(directory, frozen)
    (directory / "extra.txt").unlink()
    recorded = manifest.model_dump()
    recorded["specification"]["repository"] = "fixture/other"
    (directory / MANIFEST_NAME).write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="manifest"):
        verify_snapshot(directory, frozen)


async def test_metadata_identity_limits_and_nonfinite_timeout(tmp_path: Path) -> None:
    """An absent immutable commit or oversized metadata cannot start model downloads."""
    frozen = specification()
    with pytest.raises(ValueError, match="revision"):
        parse_hub_plan(
            {"sha": "b" * 40}, frozen.repository, frozen.revision, ("config.json",), 1024
        )
    with pytest.raises(ValueError, match="duplicate"):
        parse_hub_plan(
            {"sha": frozen.revision, "siblings": [{"rfilename": "config.json"}] * 2},
            frozen.repository,
            frozen.revision,
            ("config.json",),
            1024,
        )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"012345"))
    ) as client:
        async with client.stream("GET", "https://huggingface.co/metadata") as response:
            with pytest.raises(ValueError, match="byte limit"):
                await bounded_response(response, 5)
        for value in (float("nan"), float("inf"), 0, -1):
            with pytest.raises(ValueError, match="timeout"):
                await fetch_verified_model(client, frozen, tmp_path, timeout_seconds=value)
        with pytest.raises(ValueError, match="selection"):
            await plan_hub_snapshot(client, frozen.repository, frozen.revision, (), 1024)


async def test_cancelled_download_removes_only_its_owned_staging(tmp_path: Path) -> None:
    """Cancellation preserves unrelated model volumes and never leaves a final partial snapshot."""
    started = asyncio.Event()

    async def response(_: httpx.Request) -> httpx.Response:
        """Block the owned fixture request until task cancellation reaches the download await."""
        started.set()
        await asyncio.Event().wait()
        return httpx.Response(200, content=b"{}")

    untouched = tmp_path / "unrelated.txt"
    untouched.write_text("keep")
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        task = asyncio.create_task(fetch_verified_model(client, specification(), tmp_path))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert await asyncio.to_thread(lambda: list(tmp_path.iterdir())) == [untouched]


async def test_concurrent_publish_and_empty_nested_file(tmp_path: Path) -> None:
    """Competing downloads converge on one verified directory without replacing its content."""
    frozen = specification(b"")
    frozen = ModelFetchSpec.model_validate(
        {
            **frozen.model_dump(),
            "files": [{**frozen.files[0].model_dump(), "path": "nested/config.json"}],
        }
    )
    arrived = 0
    ready = asyncio.Event()

    async def response(_: httpx.Request) -> httpx.Response:
        """Let both attempts observe no final directory before either response completes."""
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            ready.set()
        await ready.wait()
        return httpx.Response(200, content=b"")

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        first, second = await asyncio.gather(
            fetch_verified_model(client, frozen, tmp_path),
            fetch_verified_model(client, frozen, tmp_path),
        )
    assert first == second and arrived == 2
    assert (first[0] / "nested/config.json").read_bytes() == b""


def test_snapshot_missing_symlink_and_oversized_manifest_rejected(tmp_path: Path) -> None:
    """Neither missing data, symlink substitution nor an oversized record is verified."""
    frozen = specification()
    with pytest.raises(ValueError, match="file set"):
        verify_snapshot(tmp_path, frozen)
    leaf = tmp_path / "config.json"
    leaf.write_bytes(b"{}")
    with pytest.raises(ValueError, match="regular"):
        verify_file(tmp_path, frozen.files[0])
    (tmp_path / MANIFEST_NAME).write_bytes(b"x" * (2 * 1024**2 + 1))
    with pytest.raises(ValueError, match="byte limit"):
        verify_snapshot(tmp_path, frozen)
    (tmp_path / MANIFEST_NAME).unlink()
    link = tmp_path / "link.json"
    try:
        link.symlink_to(leaf)
    except OSError:
        pytest.skip("symlink creation unavailable on this host")
    with pytest.raises(ValueError, match="symlink"):
        verify_snapshot(tmp_path, frozen)
    with pytest.raises(ValueError, match="symlink"):
        verify_snapshot(link, frozen)


async def test_source_checkout_cannot_be_a_model_volume() -> None:
    """Reject an in-repository destination before constructing download requests."""
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="external"):
            await fetch_verified_model(client, specification(), Path(__file__).parents[2])


@pytest.mark.parametrize("fail", [False, True])
async def test_owned_disk_work_is_drained_before_cancellation_returns(fail: bool) -> None:
    """Cancelled producers retain disk ownership while peer tasks stay responsive."""
    started, release, finished = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def operation() -> str:
        """Hold a real worker thread until the event loop explicitly allows disk work to finish."""
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=5):
            raise TimeoutError("test worker was not released")
        finished.set()
        if fail:
            raise OSError("test disk failure during cancellation")
        return "finished"

    task = asyncio.create_task(owned_disk(operation))
    async with asyncio.timeout(5):
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set()
