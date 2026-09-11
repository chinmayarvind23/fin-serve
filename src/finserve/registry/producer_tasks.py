"""Trusted resumable model, image and quality stages built on immutable attempt receipts."""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import Field

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.model_assets import ModelFetchSpec, ModelManifest
from finserve.contracts.producer import QualityCollectionSpec
from finserve.http_ownership import HTTPClosureError
from finserve.registry.artifacts import ArtifactRef
from finserve.registry.model_assets import (
    external_root,
    fetch_verified_model,
    owned_disk,
    verify_snapshot,
)
from finserve.registry.producer_stages import ProducerStages, StageState
from finserve.registry.quality_collection import (
    QualityCollectionResult,
    bounded_file,
    collect_quality,
    load_quality_collection,
)
from finserve.registry.runtime_build import RuntimeBuildSpec, RuntimeImage, build_runtime


class ModelSnapshotReceipt(ImmutableModel):
    """A small manifest describes verified weights retained in their owned local volume."""

    kind: Literal["model-snapshot-v1"] = "model-snapshot-v1"
    directory: Path
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest: ArtifactRef


class QualityReceipt(ImmutableModel):
    """This receipt accounts for completed collection bytes; the evaluator decides quality."""

    kind: Literal["quality-collection-receipt-v1"] = "quality-collection-receipt-v1"
    specification: ArtifactRef
    requests: ArtifactRef
    result: ArtifactRef


def declare_input(journal: ProducerStages, stage_id: str, value: object) -> StageState:
    """Freeze stage parameters and local namespace before the first attempt can issue work."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return journal.declare(stage_id, journal.artifacts.put(data))


async def failed_after_drain(
    journal: ProducerStages, stage_id: str, attempt_id: str, code: str
) -> None:
    """Only the known owner can record that its controlled async/file work has fully unwound."""

    def persist() -> None:
        """Record ownership and the drained-work observation in immutable evidence."""
        state = journal.state(stage_id)
        if state.status != "running" or state.attempt_id != attempt_id:
            return
        proof = journal.artifacts.put(
            json.dumps(
                {
                    "attempt_id": attempt_id,
                    "observation": "owned task and disk workers drained",
                    "error_code": code,
                }
            ).encode()
        )
        journal.fail(stage_id, attempt_id, code, reconciliation=proof)

    await owned_disk(persist)


async def start_owned(journal: ProducerStages, stage_id: str) -> StageState:
    """Know ownership before offloading SQL so early cancellation can be reconciled safely."""
    token = uuid4().hex
    try:
        return await owned_disk(lambda: journal.start(stage_id, attempt_id=token))
    except asyncio.CancelledError:
        await failed_after_drain(journal, stage_id, token, "CancelledBeforeAction")
        raise


async def fetch_stage(
    journal: ProducerStages,
    stage_id: str,
    client: httpx.AsyncClient,
    specification: ModelFetchSpec,
    model_root: Path,
) -> ModelSnapshotReceipt:
    """Fetch or reverify exact bytes; retries cannot trust a manifest without its local files."""
    model_root = await owned_disk(lambda: external_root(model_root))
    state = await owned_disk(
        lambda: declare_input(
            journal,
            stage_id,
            {
                "kind": "model-snapshot-v1",
                "model_root": str(model_root),
                "specification": json.loads(specification.canonical()),
            },
        )
    )
    if state.status == "completed":
        return await owned_disk(lambda: verified_model_receipt(journal, state))
    running = await start_owned(journal, stage_id)
    assert running.attempt_id is not None
    try:
        directory, manifest = await fetch_verified_model(client, specification, model_root)

        def publish() -> ModelSnapshotReceipt:
            """Publish the verified manifest before recording the stage's terminal receipt."""
            receipt = ModelSnapshotReceipt(
                directory=directory,
                specification_sha256=specification.digest(),
                manifest=journal.artifacts.put(manifest.canonical().encode()),
            )
            reference = journal.artifacts.put(receipt.model_dump_json().encode())
            assert running.attempt_id is not None
            journal.finish(stage_id, running.attempt_id, reference)
            return receipt

        return await owned_disk(publish)
    except HTTPClosureError:
        raise
    except BaseException as exc:
        await failed_after_drain(journal, stage_id, running.attempt_id, type(exc).__name__)
        raise


def verified_model_receipt(journal: ProducerStages, state: StageState) -> ModelSnapshotReceipt:
    """Completed model stages must still verify their actual volume before image build or reuse."""
    if state.status != "completed" or state.output is None:
        raise ValueError("model stage is not completed")
    receipt = ModelSnapshotReceipt.model_validate_json(journal.artifacts.get(state.output))
    manifest = ModelManifest.model_validate_json(journal.artifacts.get(receipt.manifest))
    frozen = json.loads(journal.artifacts.get(state.input))
    specification = ModelFetchSpec.model_validate(frozen["specification"])
    if (
        manifest.specification.digest() != specification.digest()
        or receipt.specification_sha256 != specification.digest()
        or receipt.directory != Path(frozen["model_root"]) / specification.digest()
        or verify_snapshot(receipt.directory, specification).digest() != manifest.digest()
    ):
        raise ValueError("model stage receipt or local snapshot changed")
    return receipt


def attempt_directory(workspace: Path, state: StageState) -> Path:
    """Attempt paths derive from validated identities; failed attempts are never overwritten."""
    root = external_root(workspace)
    assert state.attempt_id is not None
    return root / hashlib.sha256(state.stage_id.encode()).hexdigest() / state.attempt_id


async def build_stage(
    journal: ProducerStages,
    stage_id: str,
    model_stage_id: str,
    specification: RuntimeBuildSpec,
    repository: Path,
    workspace: Path,
) -> RuntimeImage:
    """Build from verified model and committed source; ambiguous Docker failures remain running."""
    specification = RuntimeBuildSpec.model_validate_json(specification.model_dump_json())
    model = await owned_disk(lambda: verified_model_receipt(journal, journal.state(model_stage_id)))
    manifest = ModelManifest.model_validate_json(
        await owned_disk(lambda: journal.artifacts.get(model.manifest))
    )
    repository = await owned_disk(repository.resolve)
    workspace = await owned_disk(lambda: external_root(workspace))
    state = await owned_disk(
        lambda: declare_input(
            journal,
            stage_id,
            {
                "kind": "runtime-image-v1",
                "repository": str(repository),
                "workspace": str(workspace),
                "model_stage_id": model_stage_id,
                "model_manifest": model.manifest.model_dump(),
                "specification": specification.model_dump(),
            },
        )
    )
    if state.status == "completed":
        reference = state.output
        assert reference is not None
        image = RuntimeImage.model_validate_json(
            await owned_disk(lambda: journal.artifacts.get(reference))
        )
        if image.specification != specification:
            raise ValueError("build receipt differs from frozen specification")
        return image
    running = await start_owned(journal, stage_id)
    directory = await owned_disk(lambda: attempt_directory(workspace, running))
    # A drained CLI does not prove BuildKit stopped. A failed build stays running, requiring
    # reconciliation of this attempt's retained output and daemon state before any retry.
    image = await owned_disk(lambda: build_runtime(repository, specification, manifest, directory))

    def publish() -> RuntimeImage:
        """Persist actual build identities only after the builder verifies its Docker inspection."""
        reference = journal.artifacts.put(image.model_dump_json().encode())
        assert running.attempt_id is not None
        journal.finish(stage_id, running.attempt_id, reference)
        return image

    return await owned_disk(publish)


def quality_receipt(journal: ProducerStages, directory: Path) -> ArtifactRef:
    """Recompute complete quality evidence before publishing its three immutable artifact refs."""
    specification, _ = load_quality_collection(directory)
    receipt = QualityReceipt(
        specification=journal.artifacts.put(
            bounded_file(directory / "specification.json", 2 * 1024**2)
        ),
        requests=journal.artifacts.put(
            bounded_file(directory / "requests.jsonl", specification.maximum_raw_bytes)
        ),
        result=journal.artifacts.put(
            bounded_file(directory / "result.json", specification.maximum_raw_bytes)
        ),
    )
    return journal.artifacts.put(receipt.model_dump_json().encode())


def load_quality_receipt(
    journal: ProducerStages, reference: ArtifactRef
) -> tuple[QualityCollectionSpec, QualityCollectionResult]:
    """Materialize only verified CAS bytes, then reconstruct the completed quality outputs again."""
    receipt = QualityReceipt.model_validate_json(journal.artifacts.get(reference))
    with tempfile.TemporaryDirectory(prefix="finserve-quality-receipt-") as temporary:
        path = Path(temporary)
        for name, artifact in (
            ("specification.json", receipt.specification),
            ("requests.jsonl", receipt.requests),
            ("result.json", receipt.result),
        ):
            (path / name).write_bytes(journal.artifacts.get(artifact))
        return load_quality_collection(path)


async def quality_stage(
    journal: ProducerStages,
    stage_id: str,
    client: httpx.AsyncClient,
    specification: QualityCollectionSpec,
    workspace: Path,
) -> QualityCollectionResult:
    """Completed stages replay verified outputs; interrupted attempts require reconciliation."""
    specification = QualityCollectionSpec.model_validate_json(specification.model_dump_json())
    workspace = await owned_disk(lambda: external_root(workspace))
    state = await owned_disk(
        lambda: declare_input(
            journal,
            stage_id,
            {
                "kind": "quality-collection-v1",
                "workspace": str(workspace),
                "specification": specification.model_dump(),
            },
        )
    )
    if state.status == "completed":
        reference = state.output
        assert reference is not None
        recorded, result = await owned_disk(lambda: load_quality_receipt(journal, reference))
        if recorded.digest() != specification.digest():
            raise ValueError("quality receipt differs from immutable stage input")
        return result
    running = await start_owned(journal, stage_id)
    assert running.attempt_id is not None
    try:
        directory = await owned_disk(lambda: attempt_directory(workspace, running))
        result = await collect_quality(client, specification, directory)

        def publish() -> None:
            """Every published result has passed raw-artifact reconstruction, not a passed flag."""
            reference = quality_receipt(journal, directory)
            assert running.attempt_id is not None
            journal.finish(stage_id, running.attempt_id, reference)

        await owned_disk(publish)
        return result
    except HTTPClosureError:
        raise
    except BaseException as exc:
        await failed_after_drain(journal, stage_id, running.attempt_id, type(exc).__name__)
        raise
