"""Bind raw quality responses to observations of one completed managed runtime start."""

import json
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.managed_runtime import RuntimeLaunchSpec, RuntimeReceipt
from finserve.contracts.producer import QualityCollectionSpec
from finserve.http_ownership import HTTPClosureError
from finserve.registry.artifacts import ArtifactRef
from finserve.registry.managed_runtime import DockerRuntime, owned_directory
from finserve.registry.model_assets import owned_disk
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import (
    attempt_directory,
    declare_input,
    failed_after_drain,
    load_quality_receipt,
    quality_receipt,
    start_owned,
)
from finserve.registry.quality_collection import collect_quality
from finserve.registry.runtime_stages import load_launch


class ManagedQualityReceipt(ImmutableModel):
    """Runtime identity is distinct from correctness; only the release grader grants approval."""

    kind: Literal["managed-quality-v1"] = "managed-quality-v1"
    runtime: RuntimeLaunchSpec
    launch: ArtifactRef
    quality: ArtifactRef
    before: RuntimeReceipt
    after: RuntimeReceipt
    collection_started_at: float = Field(gt=0)
    collection_finished_at: float = Field(gt=0)


def load_managed_quality(journal: ProducerStages, reference: ArtifactRef) -> ManagedQualityReceipt:
    """Reconstruct raw quality evidence and exact runtime linkage without offering requests."""
    receipt = ManagedQualityReceipt.model_validate_json(journal.artifacts.get(reference))
    launch = RuntimeReceipt.model_validate_json(journal.artifacts.get(receipt.launch))
    spec, _ = load_quality_receipt(journal, receipt.quality)
    if receipt.runtime.profile != spec.profile or receipt.runtime.revision != spec.revision:
        raise ValueError("managed quality runtime differs from collection")
    if receipt.runtime.digest() != launch.specification_sha256:
        raise ValueError("managed quality runtime differs from launch")
    for observation in (receipt.before, receipt.after):
        for name in (
            "specification_sha256",
            "attempt_id",
            "container_id",
            "container_started_at",
            "output_directory",
        ):
            if getattr(observation, name) != getattr(launch, name):
                raise ValueError("managed quality observations refer to different runtime starts")
    if not (
        receipt.before.observed_at
        <= receipt.collection_started_at
        <= receipt.collection_finished_at
        <= receipt.after.observed_at
    ):
        raise ValueError("quality collection falls outside runtime observations")
    return receipt


async def managed_quality_stage(
    journal: ProducerStages,
    stage_id: str,
    launch_stage_id: str,
    specification: QualityCollectionSpec,
    workspace: Path,
    client: httpx.AsyncClient,
    runtime: DockerRuntime,
) -> ManagedQualityReceipt:
    """One journal attempt owns observation, collection and immutable publication together."""
    spec = QualityCollectionSpec.model_validate_json(specification.model_dump_json())
    launch_state = await owned_disk(lambda: journal.state(launch_stage_id))
    frozen = json.loads(await owned_disk(lambda: journal.artifacts.get(launch_state.input)))
    runtime_spec = RuntimeLaunchSpec.model_validate(frozen["specification"])
    launch = await owned_disk(lambda: load_launch(journal, launch_state, runtime_spec))
    if runtime_spec.profile != spec.profile or runtime_spec.revision != spec.revision:
        raise ValueError("quality runtime differs from completed launch")
    assert launch_state.output is not None
    launch_ref = launch_state.output
    workspace = await owned_disk(lambda: owned_directory(workspace, create=True))
    state = await owned_disk(
        lambda: declare_input(
            journal,
            stage_id,
            {
                "kind": "managed-quality-v1",
                "specification": json.loads(spec.canonical()),
                "workspace": str(workspace),
                "launch_stage_id": launch_stage_id,
                "launch": launch_ref.model_dump(),
            },
        )
    )
    if state.status == "completed":
        assert state.output is not None
        output = state.output
        receipt = await owned_disk(lambda: load_managed_quality(journal, output))
        recorded, _ = await owned_disk(lambda: load_quality_receipt(journal, receipt.quality))
        if recorded.digest() != spec.digest() or receipt.launch != launch_ref:
            raise ValueError("managed quality receipt differs from stage input")
        return receipt
    running = await start_owned(journal, stage_id)
    assert running.attempt_id is not None
    try:
        directory = await owned_disk(lambda: attempt_directory(workspace, running))
        before = await runtime.observe(runtime_spec, launch, client)
        started = time.time()
        await collect_quality(client, spec, directory)
        finished = time.time()
        after = await runtime.observe(runtime_spec, launch, client)

        def publish() -> ManagedQualityReceipt:
            """Publish reconstructed quality evidence enclosed by exact runtime observations."""
            receipt = ManagedQualityReceipt(
                runtime=runtime_spec,
                launch=launch_ref,
                quality=quality_receipt(journal, directory),
                before=before,
                after=after,
                collection_started_at=started,
                collection_finished_at=finished,
            )
            reference = journal.artifacts.put(receipt.model_dump_json().encode())
            verified = load_managed_quality(journal, reference)
            recorded, _ = load_quality_receipt(journal, verified.quality)
            if recorded.digest() != spec.digest():
                raise ValueError("managed quality collection differs from frozen specification")
            assert running.attempt_id is not None
            journal.finish(stage_id, running.attempt_id, reference)
            return verified

        return await owned_disk(publish)
    except HTTPClosureError:
        raise
    except BaseException as exc:
        await failed_after_drain(journal, stage_id, running.attempt_id, type(exc).__name__)
        raise
