"""Plan fixed Hugging Face snapshots and verify streamed model files before publication."""

import asyncio
import hashlib
import json
import os
import stat
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import httpx

from finserve.contracts.model_assets import (
    HubRevision,
    ModelFetchSpec,
    ModelManifest,
    SourceFile,
    VerifiedFile,
    safe_asset_path,
)
from finserve.http_ownership import own_response

CHUNK_BYTES = 1024 * 1024
MANIFEST_NAME = "finserve-model-manifest.json"


async def owned_disk[DiskResult](operation: Callable[[], DiskResult]) -> DiskResult:
    """Drain an owned disk worker on cancellation before its file or staging directory is closed."""
    worker = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


def sync_file(output: BinaryIO) -> None:
    """Flush and fsync in the owned disk worker rather than pausing peer producer tasks."""
    output.flush()
    os.fsync(output.fileno())


async def bounded_response(response: httpx.Response, maximum_bytes: int) -> bytes:
    """Read metadata with an incremental byte cap instead of buffering an arbitrary response."""
    response.raise_for_status()
    content = bytearray()
    async for chunk in response.aiter_bytes(CHUNK_BYTES):
        if len(content) + len(chunk) > maximum_bytes:
            raise ValueError("model metadata exceeds byte limit")
        content.extend(chunk)
    return bytes(content)


def parse_hub_plan(
    document: dict[str, Any],
    repository: str,
    revision: str,
    files: tuple[str, ...],
    maximum_bytes: int,
) -> ModelFetchSpec:
    """Translate documented Hub blob/LFS metadata into explicit source checksum contracts."""
    if document.get("sha") != revision:
        raise ValueError("Hub metadata differs from requested immutable revision")
    siblings = {item["rfilename"]: item for item in document["siblings"]}
    if len(siblings) != len(document["siblings"]):
        raise ValueError("Hub metadata has duplicate file identities")
    selected: list[SourceFile] = []
    for name in files:
        value = siblings[name]
        lfs = value.get("lfs")
        selected.append(
            SourceFile(
                path=name,
                size_bytes=value["size"],
                checksum_kind="sha256" if lfs else "git-blob-sha1",
                checksum=lfs["sha256"] if lfs else value["blobId"],
            )
        )
    return ModelFetchSpec(
        repository=repository, revision=revision, files=tuple(selected), maximum_bytes=maximum_bytes
    )


async def plan_hub_snapshot(
    client: httpx.AsyncClient,
    repository: str,
    revision: str,
    files: tuple[str, ...],
    maximum_bytes: int,
) -> ModelFetchSpec:
    """Read only the fixed Hub origin; require full commit identity before resolving data URLs."""
    # Reuse contract validation before constructing a network destination.
    probe = HubRevision(repository=repository, revision=revision)
    if not 1 <= len(files) <= 1024:
        raise ValueError("bounded explicit file selection required")
    for name in files:
        safe_asset_path(name)
    async with asyncio.timeout(60):
        async with client.stream(
            "GET",
            f"https://huggingface.co/api/models/{probe.repository}/revision/{probe.revision}",
            params={"blobs": "true"},
            follow_redirects=False,
        ) as response:
            own_response(response)
            document = json.loads(await bounded_response(response, 2 * CHUNK_BYTES))
    return parse_hub_plan(document, repository, revision, files, maximum_bytes)


def external_root(root: Path) -> Path:
    """Model weights and producer evidence must remain outside the source checkout."""
    target = root.resolve()
    repository = Path(__file__).resolve().parents[3]
    if target == repository or repository in target.parents:
        raise ValueError("external model volume required")
    target.mkdir(parents=True, exist_ok=True)
    return target


def sync_directory(directory: Path) -> None:
    """POSIX publication fsyncs directory entries; Windows provides atomic visibility only."""
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def verify_file(path: Path, source: SourceFile, *, deadline: float | None = None) -> VerifiedFile:
    """Stream both raw SHA256 and Git blob hashing with a strict expected-size bound."""
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("regular model file required")
    raw = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(source.size_bytes).encode() + b"\0")
    size = 0
    with path.open("rb") as opened:
        while chunk := opened.read(CHUNK_BYTES):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("model verification deadline exceeded")
            size += len(chunk)
            if size > source.size_bytes:
                raise ValueError("model file exceeds frozen size")
            raw.update(chunk)
            blob.update(chunk)
    expected = raw.hexdigest() if source.checksum_kind == "sha256" else blob.hexdigest()
    if size != source.size_bytes or expected != source.checksum:
        raise ValueError("model file checksum or size differs from frozen source")
    return VerifiedFile(source=source, sha256=raw.hexdigest())


def verify_snapshot(
    directory: Path, specification: ModelFetchSpec, *, deadline: float | None = None
) -> ModelManifest:
    """Recheck actual bytes, extra files and symlink parents before reusing a published snapshot."""
    if directory.is_symlink():
        raise ValueError("model snapshot cannot be a symlink")
    actual: set[str] = set()
    expected = {item.path for item in specification.files}
    for path in directory.rglob("*"):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("model verification deadline exceeded")
        if path.is_symlink():
            raise ValueError("model snapshot contains a symlink")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name not in expected | {MANIFEST_NAME}:
                raise ValueError("model snapshot contains an unexpected file")
            actual.add(name)
    if actual not in (expected, expected | {MANIFEST_NAME}):
        raise ValueError("model snapshot file set differs from frozen selection")
    verified = tuple(
        verify_file(directory / item.path, item, deadline=deadline) for item in specification.files
    )
    manifest = ModelManifest(specification=specification, files=verified)
    if (directory / MANIFEST_NAME).exists():
        with (directory / MANIFEST_NAME).open("rb") as opened:
            content = opened.read(2 * CHUNK_BYTES + 1)
        if len(content) > 2 * CHUNK_BYTES:
            raise ValueError("model manifest exceeds byte limit")
        recorded = ModelManifest.model_validate_json(content)
        if recorded.digest() != manifest.digest():
            raise ValueError("stored model manifest differs from verified bytes")
    return manifest


async def download_file(
    client: httpx.AsyncClient, specification: ModelFetchSpec, source: SourceFile, directory: Path
) -> None:
    """Stream one immutable resolve URL to an exclusive file with no unbounded download buffer."""
    path = directory / source.path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{specification.repository}/resolve/{specification.revision}/{source.path}"
    async with client.stream("GET", url, follow_redirects=True) as response:
        own_response(response)
        response.raise_for_status()
        size = 0
        with path.open("xb") as output:
            async for chunk in response.aiter_bytes(CHUNK_BYTES):
                size += len(chunk)
                if size > source.size_bytes:
                    raise ValueError("model download exceeds frozen size")
                await owned_disk(lambda chunk=chunk: output.write(chunk))
            await owned_disk(lambda: sync_file(output))


def publish_snapshot(
    staging: Path, destination: Path, manifest: ModelManifest, deadline: float
) -> None:
    """Keep bounded atomic publication in the staging lifetime; no worker may outlive cleanup."""
    with (staging / MANIFEST_NAME).open("x") as output:
        output.write(manifest.canonical())
        output.flush()
        os.fsync(output.fileno())
    for path in staging.rglob("*"):
        if path.is_dir():
            sync_directory(path)
    sync_directory(staging)
    try:
        staging.rename(destination)
    except OSError:
        if not destination.exists():
            raise
        verify_snapshot(destination, manifest.specification, deadline=deadline)
    sync_directory(destination.parent)


async def fetch_verified_model(
    client: httpx.AsyncClient,
    specification: ModelFetchSpec,
    root: Path,
    *,
    timeout_seconds: float = 3600,
) -> tuple[Path, ModelManifest]:
    """Publish only a fully verified snapshot; cancellation never leaves a partial final model."""
    if not 0 < timeout_seconds <= 14400:
        raise ValueError("finite bounded fetch timeout required")
    specification = ModelFetchSpec.model_validate_json(specification.model_dump_json())
    deadline = time.monotonic() + timeout_seconds
    destination = external_root(root) / specification.digest()
    if destination.exists():
        return destination, await owned_disk(
            lambda: verify_snapshot(destination, specification, deadline=deadline)
        )
    async with asyncio.timeout(timeout_seconds):
        with tempfile.TemporaryDirectory(
            prefix="finserve-model-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary)
            for source in specification.files:
                await download_file(client, specification, source, staging)
            manifest = await owned_disk(
                lambda: verify_snapshot(staging, specification, deadline=deadline)
            )
            await owned_disk(lambda: publish_snapshot(staging, destination, manifest, deadline))
    return destination, manifest
