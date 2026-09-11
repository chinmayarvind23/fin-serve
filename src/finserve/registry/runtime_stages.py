"""Bind managed Docker actions to verified upstream receipts and durable attempt identities."""

import json
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.managed_runtime import RuntimeLaunchSpec, RuntimeReceipt
from finserve.registry.artifacts import ArtifactRef
from finserve.registry.managed_runtime import DockerRuntime, owned_directory
from finserve.registry.metadata import RegistryConflict
from finserve.registry.model_assets import owned_disk
from finserve.registry.producer_stages import ProducerStages, StageState
from finserve.registry.producer_tasks import declare_input, start_owned, verified_model_receipt
from finserve.registry.runtime_build import RuntimeImage


class RuntimeStopReceipt(ImmutableModel):
    """An exact owned container was observed stopped or already absent by the trusted adapter."""

    kind: Literal["managed-runtime-stop-v1"] = "managed-runtime-stop-v1"
    launch: ArtifactRef
    container_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: float = Field(gt=0)


def completed_stop(
    journal: ProducerStages,
    state: StageState,
    attempt_id: str | None,
    reference: ArtifactRef,
    container_id: str,
) -> RuntimeStopReceipt:
    """Concurrent cleanup receipts can share only the exact current attempt and launch identity."""
    if state.status != "completed" or state.attempt_id != attempt_id or state.output is None:
        raise RegistryConflict("runtime cleanup attempt changed")
    result = RuntimeStopReceipt.model_validate_json(journal.artifacts.get(state.output))
    if result.launch != reference or result.container_id != container_id:
        raise ValueError("runtime cleanup receipt identity changed")
    return result


def verify_upstream(
    journal: ProducerStages, model_stage_id: str, build_stage_id: str, spec: RuntimeLaunchSpec
) -> tuple[ArtifactRef, ArtifactRef]:
    """Rehash local model bytes and bind the image to the exact completed model/build stages."""
    model_state = journal.state(model_stage_id)
    model = verified_model_receipt(journal, model_state)
    build = journal.state(build_stage_id)
    if build.status != "completed" or build.output is None or model_state.output is None:
        raise ValueError("runtime requires completed upstream stages")
    image = RuntimeImage.model_validate_json(journal.artifacts.get(build.output))
    frozen = json.loads(journal.artifacts.get(build.input))
    if (
        model.directory != spec.model_directory
        or model.specification_sha256 != spec.model.digest()
        or image != spec.image
        or frozen.get("kind") != "runtime-image-v1"
        or frozen.get("model_stage_id") != model_stage_id
        or frozen.get("model_manifest") != model.manifest.model_dump()
        or frozen.get("specification") != image.specification.model_dump(mode="json")
    ):
        raise ValueError("runtime upstream identities differ from frozen launch")
    return model_state.output, build.output


def load_launch(
    journal: ProducerStages, state: StageState, spec: RuntimeLaunchSpec
) -> RuntimeReceipt:
    """Only a completed launch with matching frozen parameters supplies a reusable receipt."""
    if state.status != "completed" or state.output is None:
        raise ValueError("runtime launch stage is not completed")
    frozen = json.loads(journal.artifacts.get(state.input))
    receipt = RuntimeReceipt.model_validate_json(journal.artifacts.get(state.output))
    if (
        frozen.get("kind") != "managed-runtime-launch-v1"
        or frozen.get("specification") != json.loads(spec.canonical())
        or receipt.specification_sha256 != spec.digest()
        or receipt.attempt_id != state.attempt_id
        or receipt.output_directory != Path(frozen["workspace"]) / receipt.attempt_id
    ):
        raise ValueError("runtime launch receipt differs from its stage")
    return receipt


async def launch_runtime_stage(
    journal: ProducerStages,
    stage_id: str,
    model_stage_id: str,
    build_stage_id: str,
    specification: RuntimeLaunchSpec,
    workspace: Path,
    client: httpx.AsyncClient,
    runtime: DockerRuntime,
) -> RuntimeReceipt:
    """Reconcile a running attempt by name; completed replay only probes the same existing start."""
    spec = RuntimeLaunchSpec.model_validate_json(specification.model_dump_json())
    workspace = await owned_disk(lambda: owned_directory(workspace, create=True))
    model, image = await owned_disk(
        lambda: verify_upstream(journal, model_stage_id, build_stage_id, spec)
    )
    state = await owned_disk(
        lambda: declare_input(
            journal,
            stage_id,
            {
                "kind": "managed-runtime-launch-v1",
                "specification": json.loads(spec.canonical()),
                "workspace": str(workspace),
                "model_stage_id": model_stage_id,
                "build_stage_id": build_stage_id,
                "model": model.model_dump(),
                "image": image.model_dump(),
            },
        )
    )
    if state.status == "completed":
        receipt = await owned_disk(lambda: load_launch(journal, state, spec))
        await runtime.observe(spec, receipt, client)
        return receipt
    running = state if state.status == "running" else await start_owned(journal, stage_id)
    assert running.attempt_id is not None
    # Cancellation and daemon errors preserve the same attempt. A drained Docker CLI alone
    # does not establish whether its named external action happened.
    receipt = await runtime.launch(spec, running.attempt_id, workspace, client)

    def publish() -> RuntimeReceipt:
        """Concurrent reconciliations may observe different times but must agree on the start."""
        reference = journal.artifacts.put(receipt.model_dump_json().encode())
        assert running.attempt_id is not None
        try:
            journal.finish(stage_id, running.attempt_id, reference)
        except RegistryConflict:
            actual = load_launch(journal, journal.state(stage_id), spec)
            if (actual.container_id, actual.container_started_at) != (
                receipt.container_id,
                receipt.container_started_at,
            ) or actual.attempt_id != running.attempt_id:
                raise RegistryConflict("runtime completion identity changed") from None
            return actual
        return load_launch(journal, journal.state(stage_id), spec)

    return await owned_disk(publish)


async def stop_runtime_stage(
    journal: ProducerStages,
    stage_id: str,
    launch_stage_id: str,
    specification: RuntimeLaunchSpec,
    runtime: DockerRuntime,
) -> RuntimeStopReceipt:
    """Retry only exact receipt-bound cleanup; absence is safe and never starts replacement work."""
    spec = RuntimeLaunchSpec.model_validate_json(specification.model_dump_json())
    launch_state = await owned_disk(lambda: journal.state(launch_stage_id))
    launch = await owned_disk(lambda: load_launch(journal, launch_state, spec))
    assert launch_state.output is not None
    reference = launch_state.output
    state = await owned_disk(
        lambda: declare_input(
            journal,
            stage_id,
            {
                "kind": "managed-runtime-stop-v1",
                "launch_stage_id": launch_stage_id,
                "launch": reference.model_dump(),
                "specification_sha256": spec.digest(),
            },
        )
    )
    running = (
        state if state.status in {"running", "completed"} else await start_owned(journal, stage_id)
    )
    if state.status == "completed":
        await owned_disk(
            lambda: completed_stop(
                journal, state, running.attempt_id, reference, launch.container_id
            )
        )
    await runtime.stop(spec, launch)

    def publish() -> RuntimeStopReceipt:
        """Keep the first verified cleanup observation as the immutable terminal receipt."""
        actual = journal.state(stage_id)
        if actual.status == "completed" and actual.attempt_id == running.attempt_id:
            return completed_stop(
                journal, actual, running.attempt_id, reference, launch.container_id
            )
        result = RuntimeStopReceipt(
            launch=reference, container_id=launch.container_id, observed_at=time.time()
        )
        assert running.attempt_id is not None
        try:
            journal.finish(
                stage_id,
                running.attempt_id,
                journal.artifacts.put(result.model_dump_json().encode()),
            )
        except RegistryConflict:
            return completed_stop(
                journal, journal.state(stage_id), running.attempt_id, reference, launch.container_id
            )
        return result

    return await owned_disk(publish)
