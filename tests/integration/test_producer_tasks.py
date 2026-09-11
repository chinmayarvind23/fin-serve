"""Trusted producer stages replay verified receipts and retain ambiguous external build state."""

import asyncio
import hashlib
import threading
from collections.abc import AsyncIterator, Generator
from pathlib import Path

import httpx
import pytest

from finserve.benchmark.runner import RunConfig
from finserve.contracts.deployment import Revision
from finserve.contracts.model_assets import ModelFetchSpec, ModelManifest, SourceFile
from finserve.contracts.producer import QualityCollectionSpec
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.evaluation.quality import default_suite
from finserve.http_ownership import HTTPClosureError
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import (
    ModelSnapshotReceipt,
    build_stage,
    failed_after_drain,
    fetch_stage,
    quality_stage,
    start_owned,
    verified_model_receipt,
)
from finserve.registry.runtime_build import RuntimeBuildSpec, RuntimeImage


@pytest.fixture
def journal(tmp_path: Path) -> Generator[ProducerStages]:
    """Use the same real local registry/CAS namespaces across asynchronous task attempts."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        yield ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
    finally:
        registry.close()


def source_specification() -> tuple[ModelFetchSpec, bytes]:
    """A tiny original data file exercises actual download, checksum and publication code."""
    data = b'{"model_type":"fixture"}'
    return ModelFetchSpec(
        repository="owned/fixture",
        revision="a" * 40,
        maximum_bytes=1024,
        files=(
            SourceFile(
                path="config.json",
                size_bytes=len(data),
                checksum_kind="sha256",
                checksum=hashlib.sha256(data).hexdigest(),
            ),
        ),
    ), data


async def test_fetch_replays_only_verified_local_snapshot(
    journal: ProducerStages, tmp_path: Path
) -> None:
    """A completed receipt avoids repeated downloads but never skips rehashing actual files."""
    specification, data = source_specification()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        """Count actual fixture downloads independently of stage-state observations."""
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await fetch_stage(journal, "job:fetch", client, specification, tmp_path / "models")
        assert (
            await fetch_stage(journal, "job:fetch", client, specification, tmp_path / "models")
            == first
        )
        assert calls == 1 and journal.state("job:fetch").status == "completed"
        (first.directory / "config.json").write_bytes(b"tampered")
        with pytest.raises(ValueError):
            await fetch_stage(journal, "job:fetch", client, specification, tmp_path / "models")
    assert journal.state("job:fetch").status == "completed"


async def test_fetch_failure_is_explicitly_drained(journal: ProducerStages, tmp_path: Path) -> None:
    """An HTTP download failure records reconciliation before any retry can begin."""
    specification, _ = source_specification()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await fetch_stage(journal, "job:fetch", client, specification, tmp_path / "models")
    failed = journal.state("job:fetch")
    assert failed.status == "failed" and failed.reconciliation is not None
    assert not list((tmp_path / "models").iterdir())


async def test_build_receipt_replay_and_ambiguous_failure(
    journal: ProducerStages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model verification precedes building; ambiguous failure cannot silently trigger rebuild."""
    specification, data = source_specification()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))
    ) as client:
        model = await fetch_stage(journal, "job:fetch", client, specification, tmp_path / "models")
    manifest = ModelManifest.model_validate_json(journal.artifacts.get(model.manifest))
    build = RuntimeBuildSpec(source_revision="b" * 40, model_manifest_sha256=manifest.digest())
    calls = 0

    def builder(
        repository: Path, specification: RuntimeBuildSpec, model: ModelManifest, output: Path
    ) -> RuntimeImage:
        """Fixture Docker result tests orchestration, not actual image execution."""
        nonlocal calls
        calls += 1
        assert repository == tmp_path.resolve() and model.digest() == manifest.digest()
        output.mkdir(parents=True)
        return RuntimeImage(
            specification=specification,
            source_archive_sha256="c" * 64,
            image_config_digest="sha256:" + "d" * 64,
            image_manifest_digest="sha256:" + "e" * 64,
            image_local_id="sha256:" + "d" * 64,
        )

    monkeypatch.setattr("finserve.registry.producer_tasks.build_runtime", builder)
    first = await build_stage(journal, "job:build", "job:fetch", build, tmp_path, tmp_path / "work")
    assert (
        await build_stage(journal, "job:build", "job:fetch", build, tmp_path, tmp_path / "work")
        == first
    )
    assert calls == 1
    changed = build.model_copy(update={"source_revision": "f" * 40})
    with pytest.raises(RegistryConflict):
        await build_stage(journal, "job:build", "job:fetch", changed, tmp_path, tmp_path / "work")

    def broken(
        repository: Path, specification: RuntimeBuildSpec, model: ModelManifest, output: Path
    ) -> RuntimeImage:
        """A disconnected build CLI does not prove that its daemon-side build stopped."""
        raise RuntimeError("fixture daemon state unknown")

    monkeypatch.setattr("finserve.registry.producer_tasks.build_runtime", broken)
    with pytest.raises(RuntimeError):
        await build_stage(journal, "other:build", "job:fetch", build, tmp_path, tmp_path / "work")
    assert journal.state("other:build").status == "running"
    with pytest.raises(RegistryConflict):
        await build_stage(journal, "other:build", "job:fetch", build, tmp_path, tmp_path / "work")


async def test_cancel_before_action_keeps_known_attempt_owner(journal: ProducerStages) -> None:
    """Cancellation during offloaded SQL retains ownership without permitting duplicate work."""
    entered, released = threading.Event(), threading.Event()

    def clock() -> float:
        """Hold the first real transaction until cancellation is waiting on its owned thread."""
        entered.set()
        assert released.wait(5)
        return 10.0

    journal.clock = clock
    journal.declare("job:cancel", journal.artifacts.put(b"input"))
    task = asyncio.create_task(start_owned(journal, "job:cancel"))
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    released.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    state = journal.state("job:cancel")
    assert state.status == "failed" and state.error_code == "CancelledBeforeAction"
    assert state.attempt_id is not None
    with pytest.raises(RegistryConflict, match="reused"):
        journal.start("job:cancel", attempt_id=state.attempt_id)


async def test_fetch_cancel_drains_response_and_staging(
    journal: ProducerStages, tmp_path: Path
) -> None:
    """Partial downloads disappear only after their owned response and writes finish unwinding."""
    entered = asyncio.Event()
    specification, data = source_specification()

    class Stream(httpx.AsyncByteStream):
        """Hold a real async byte stream after part of its frozen file has arrived."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """The producer must cancel the network wait and safely release staging ownership."""
            yield data[:3]
            entered.set()
            await asyncio.Event().wait()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))
    ) as client:
        task = asyncio.create_task(
            fetch_stage(journal, "job:fetch", client, specification, tmp_path / "models")
        )
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert journal.state("job:fetch").status == "failed"
    assert not list((tmp_path / "models").iterdir())


def quality_specification() -> QualityCollectionSpec:
    """Freeze a chat fixture's complete declaration without claiming runtime attestation."""
    profile = ServingProfileV1(
        engine="fixture",
        engine_version="1.0.0",
        model_revision="a" * 40,
        tokenizer_revision="a" * 40,
        model_manifest_sha256="b" * 64,
        tokenizer_manifest_sha256="b" * 64,
        base_url="http://fixture/v1",
        served_model="fixture",
    )
    revision = Revision(
        revision_id="fixture",
        model_revision=profile.model_revision,
        tokenizer_revision=profile.tokenizer_revision,
        source_revision="c" * 40,
        image_digest="sha256:" + "d" * 64,
        config_digest=profile.digest(),
        engine=profile.engine,
        engine_config=profile.engine_parameters_json,
    )
    config = RunConfig(
        model="fixture",
        revision=revision.source_revision,
        model_revision=revision.model_revision,
        tokenizer_revision=revision.tokenizer_revision,
        engine=revision.engine,
        engine_config=revision.engine_config,
        image_digest=revision.image_digest,
        config_digest=revision.config_digest,
        request_api="chat",
        chat_template_sha256="e" * 64,
    )
    return QualityCollectionSpec(
        collection_id="fixture",
        profile=profile,
        revision=revision,
        suite=default_suite(),
        configuration=config,
    )


async def test_quality_stage_replays_verified_outputs(
    journal: ProducerStages, tmp_path: Path
) -> None:
    """A completed quality collection is durable evidence; it is not itself a quality approval."""
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        """Count actual response requests so replay cannot secretly recollect an easier sample."""
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
                'data: {"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
            ),
        )

    specification = quality_specification()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await quality_stage(
            journal, "job:quality", client, specification, tmp_path / "work"
        )
        assert (
            await quality_stage(journal, "job:quality", client, specification, tmp_path / "work")
            == first
        )
        assert calls == 3 and first.successful == 3
        with pytest.raises(RegistryConflict):
            await quality_stage(
                journal,
                "job:quality",
                client,
                specification.model_copy(update={"max_tokens": 32}),
                tmp_path / "work",
            )


async def test_cancelled_quality_stage_reconciles_before_retry(
    journal: ProducerStages, tmp_path: Path
) -> None:
    """An active HTTP stream is closed before its stage can become retryable."""
    entered = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        """Hold after a partial output so cancellation exercises actual collector ownership."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """The partial content must survive in the attempt directory."""
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            entered.set()
            await asyncio.Event().wait()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))
    ) as client:
        task = asyncio.create_task(
            quality_stage(
                journal, "job:quality", client, quality_specification(), tmp_path / "work"
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    failed = journal.state("job:quality")
    assert failed.status == "failed" and failed.reconciliation is not None
    records = list((tmp_path / "work").rglob("requests.jsonl"))
    assert len(records) == 1 and "partial" in records[0].read_text()


async def test_model_receipt_rejects_incomplete_or_wrong_identity(
    journal: ProducerStages, tmp_path: Path
) -> None:
    """A journal completion cannot substitute a model receipt bound to different frozen input."""
    specification, data = source_specification()
    planned = journal.declare("job:unstarted", journal.artifacts.put(b"input"))
    with pytest.raises(ValueError, match="not completed"):
        verified_model_receipt(journal, planned)
    await failed_after_drain(journal, planned.stage_id, "a" * 32, "IgnoredStaleOwner")
    assert journal.state(planned.stage_id) == planned
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))
    ) as client:
        model = await fetch_stage(journal, "job:fetch", client, specification, tmp_path / "models")
    original = journal.state("job:fetch")
    bad = ModelSnapshotReceipt(
        directory=model.directory,
        specification_sha256="f" * 64,
        manifest=model.manifest,
    )
    altered = original.model_copy(
        update={"output": journal.artifacts.put(bad.model_dump_json().encode())}
    )
    with pytest.raises(ValueError, match="receipt"):
        verified_model_receipt(journal, altered)
    await failed_after_drain(
        journal, original.stage_id, original.attempt_id or "", "IgnoredCompleted"
    )
    assert journal.state(original.stage_id) == original


@pytest.mark.parametrize("kind", ["fetch", "quality"])
@pytest.mark.parametrize("close_fails", [False, True])
async def test_cancel_during_http_close_retains_stage_ownership(
    journal: ProducerStages, tmp_path: Path, kind: str, close_fails: bool
) -> None:
    """Repeated cancellation cannot make a stage retryable while local HTTP cleanup still runs."""
    closing, release = asyncio.Event(), asyncio.Event()
    specification, data = source_specification()

    class Stream(httpx.AsyncByteStream):
        """Delay actual response cleanup after the complete body has arrived."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """Complete bytes force HTTPX's automatic close path inside body iteration."""
            if kind == "fetch":
                yield data
            else:
                yield (
                    b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
                    b'data: {"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
                )

        async def aclose(self) -> None:
            """The owning stage must wait for this real transport cleanup barrier."""
            closing.set()
            await release.wait()
            if close_fails:
                raise OSError("fixture transport close failed")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))
    ) as client:
        operation = (
            fetch_stage(journal, "job:close", client, specification, tmp_path / "models")
            if kind == "fetch"
            else quality_stage(
                journal, "job:close", client, quality_specification(), tmp_path / "work"
            )
        )
        task = asyncio.create_task(operation)
        await asyncio.wait_for(closing.wait(), 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and journal.state("job:close").status == "running"
        release.set()
        with pytest.raises(HTTPClosureError if close_fails else asyncio.CancelledError):
            await task
    assert journal.state("job:close").status == ("running" if close_fails else "failed")
    if kind == "quality" and close_fails:
        records = list((tmp_path / "work").rglob("requests.jsonl"))
        assert len(records) == 1
        assert "HTTPClosureError" in records[0].read_text()
        assert "answer" in records[0].read_text()
