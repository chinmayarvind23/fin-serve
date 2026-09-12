"""Explicit reconciliation for failed startup; readiness receipts are never synthesized."""

# This trusted companion shares DockerRuntime's internal exact-selector primitives;
# they deliberately remain unavailable as a broad public cleanup API.
# pyright: reportPrivateUsage=false

import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.managed_runtime import RuntimeLaunchSpec
from finserve.registry.artifacts import ArtifactRef
from finserve.registry.managed_runtime import (
    DockerRuntime,
    freeze_file,
    owned_directory,
    runtime_name,
)
from finserve.registry.metadata import RegistryConflict
from finserve.registry.model_assets import owned_disk
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import declare_input, start_owned
from finserve.registry.runtime_build import bounded_document
from finserve.registry.runtime_fence import RUNTIME_OPERATION_PROTOCOL, attempt_fence


class RuntimeAbortReceipt(ImmutableModel):
    """Terminal failed-start reconciliation, distinct from a healthy runtime or normal stop."""

    kind: Literal["managed-runtime-abort-v1"] = "managed-runtime-abort-v1"
    launch_input: ArtifactRef
    launch_stage_id: str
    attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    container_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    container_started_at: str | None = None
    outcome: Literal["absent", "stopped-and-removed"]
    inspections: tuple[ArtifactRef, ...]
    log_error: str | None = None
    observed_at: float = Field(gt=0)


async def verify_absent(
    runtime: DockerRuntime, attempt_id: str, directory: Path, container_id: str | None
) -> None:
    """A failed daemon query or any replacement forbids a terminal absence claim."""
    if container_id is not None:
        if await runtime._lookup("id=" + container_id, directory) is not None:
            raise RegistryConflict("aborted container is still present")
    if await runtime._find(runtime_name(attempt_id), directory) is not None:
        raise RegistryConflict("aborted attempt name is still present")


async def reconcile_container(
    journal: ProducerStages,
    runtime: DockerRuntime,
    spec: RuntimeLaunchSpec,
    attempt_id: str,
    directory: Path,
    launch_input: ArtifactRef,
    launch_stage_id: str,
) -> RuntimeAbortReceipt:
    """Remove only a validated allocation and first start; legacy ambiguity stays fenced."""
    allocation = await owned_disk(
        lambda: (
            bounded_document(directory / "allocation.json")
            if (directory / "allocation.json").exists()
            else None
        )
    )
    start = await owned_disk(
        lambda: (
            bounded_document(directory / "first-start.json")
            if (directory / "first-start.json").exists()
            else None
        )
    )
    bound_id = allocation["container_id"] if allocation else None
    found = await runtime._find(runtime_name(attempt_id), directory)
    if found is None and bound_id is None:
        unresolved_create = await owned_disk(
            lambda: (
                (directory / "create-request.json").exists()
                and not (directory / "create-complete.json").exists()
            )
        )
        if unresolved_create:
            # Killing a CLI does not cancel the daemon's accepted request. A successful
            # lookup cannot rule out an allocation that the daemon has yet to finish.
            raise RegistryConflict("unacknowledged create requires operator reconciliation")
    observations: list[ArtifactRef] = []
    log_error = None
    if found is not None:
        actual = await runtime._inspect(found, spec, attempt_id, directory)
        observations.append(
            await owned_disk(
                lambda: journal.artifacts.put(json.dumps(actual, sort_keys=True).encode())
            )
        )
        if bound_id is not None and found != bound_id:
            raise RegistryConflict("runtime allocation changed before abort")
        if start is None:
            if await owned_disk(lambda: (directory / "start-request.json").exists()):
                raise RegistryConflict("unbound requested start requires operator reconciliation")
            if actual["State"]["Status"] != "created" or actual["State"]["Running"] is not False:
                raise RegistryConflict(
                    "runtime first start is unbound; operator reconciliation required"
                )
        elif start != {"container_id": found, "started_at": actual["State"]["StartedAt"]}:
            raise RegistryConflict("runtime was restarted before abort")
        bound_id = found
        started_at = actual["State"]["StartedAt"]
        # Freeze the validated created allocation too, so a crash cannot authorize a replacement.
        await owned_disk(
            lambda: freeze_file(directory / "allocation.json", json.dumps({"container_id": found}))
        )
        if actual["State"]["Running"] is True:
            await runtime._command(
                [
                    "docker",
                    "container",
                    "stop",
                    "--time",
                    str(spec.shutdown_timeout_seconds),
                    found,
                ],
                directory,
                spec.shutdown_timeout_seconds + 15,
            )
        try:
            await runtime._command(["docker", "container", "logs", found], directory, 30)
        except Exception as error:
            # Log capture is auxiliary: keep the error in the terminal receipt while
            # allowing an independently verified stop/remove to release owned resources.
            log_error = type(error).__name__
        after = await runtime._inspect(found, spec, attempt_id, directory)
        observations.append(
            await owned_disk(
                lambda: journal.artifacts.put(json.dumps(after, sort_keys=True).encode())
            )
        )
        if after["State"]["Running"] is not False or after["State"]["StartedAt"] != started_at:
            raise RegistryConflict("runtime stop identity is not verified")
        await runtime._command(["docker", "container", "rm", found], directory, 30)
    await verify_absent(runtime, attempt_id, directory, bound_id)
    observations.append(
        await owned_disk(
            lambda: journal.artifacts.put(
                json.dumps(
                    {
                        "kind": "managed-runtime-terminal-absence-v1",
                        "container_id": bound_id,
                        "attempt_name": runtime_name(attempt_id),
                        "id_absent": True,
                        "name_absent": True,
                        "observed_at": time.time(),
                    },
                    sort_keys=True,
                ).encode()
            )
        )
    )
    return RuntimeAbortReceipt(
        launch_input=launch_input,
        launch_stage_id=launch_stage_id,
        attempt_id=attempt_id,
        specification_sha256=spec.digest(),
        container_id=bound_id,
        container_started_at=start["started_at"] if start else None,
        outcome="absent" if found is None else "stopped-and-removed",
        inspections=tuple(observations),
        log_error=log_error,
        observed_at=time.time(),
    )


async def abort_runtime_stage(
    journal: ProducerStages,
    abort_stage_id: str,
    launch_stage_id: str,
    expected_attempt_id: str,
    runtime: DockerRuntime,
) -> RuntimeAbortReceipt:
    """Fence the exact incomplete launch, persist intent, reconcile, then publish failure."""
    runtime_name(expected_attempt_id)
    launch = await owned_disk(lambda: journal.state(launch_stage_id))
    if launch.attempt_id != expected_attempt_id or launch.status not in {"running", "failed"}:
        raise RegistryConflict("abort requires the exact incomplete launch attempt")
    frozen = await owned_disk(lambda: json.loads(journal.artifacts.get(launch.input)))
    if frozen.get("kind") != "managed-runtime-launch-v1":
        raise ValueError("abort requires a managed launch input")
    if frozen.get("operation_protocol") != RUNTIME_OPERATION_PROTOCOL:
        raise RegistryConflict("legacy runtime protocol requires operator reconciliation")
    spec = RuntimeLaunchSpec.model_validate(frozen["specification"])
    directory = await owned_disk(
        lambda: owned_directory(
            owned_directory(Path(frozen["workspace"]), create=True) / expected_attempt_id,
            create=True,
        )
    )
    async with attempt_fence(directory):
        current = await owned_disk(lambda: journal.state(launch_stage_id))
        if current.attempt_id != expected_attempt_id or current.status not in {"running", "failed"}:
            raise RegistryConflict("launch completed or changed before abort fence")
        intent = {
            "kind": "managed-runtime-abort-v1",
            "launch_stage_id": launch_stage_id,
            "launch_input": launch.input.model_dump(),
            "attempt_id": expected_attempt_id,
            "abort_stage_id": abort_stage_id,
            "specification_sha256": spec.digest(),
        }
        state = await owned_disk(lambda: declare_input(journal, abort_stage_id, intent))
        running = (
            state
            if state.status in {"running", "completed"}
            else await start_owned(journal, abort_stage_id)
        )
        # Durable intent is published before the first daemon query. It survives either
        # executor dying and permanently blocks replay of this launch attempt.
        await owned_disk(
            lambda: freeze_file(directory / "abort-intent.json", json.dumps(intent, sort_keys=True))
        )
        if current.status == "failed":
            if current.error_code != "RuntimeAborted" or current.reconciliation is None:
                raise RegistryConflict("launch failed with different reconciliation")
            reconciliation = current.reconciliation
            result = RuntimeAbortReceipt.model_validate_json(
                await owned_disk(lambda: journal.artifacts.get(reconciliation))
            )
            if (
                result.launch_input,
                result.attempt_id,
                result.launch_stage_id,
                result.specification_sha256,
            ) != (
                launch.input,
                expected_attempt_id,
                launch_stage_id,
                spec.digest(),
            ):
                raise RegistryConflict("abort reconciliation identity changed")
            await verify_absent(runtime, expected_attempt_id, directory, result.container_id)
        else:
            result = await reconcile_container(
                journal,
                runtime,
                spec,
                expected_attempt_id,
                directory,
                launch.input,
                launch_stage_id,
            )
        reference = await owned_disk(
            lambda: journal.artifacts.put(result.model_dump_json().encode())
        )
        if current.status == "running":
            await owned_disk(
                lambda: journal.fail(
                    launch_stage_id, expected_attempt_id, "RuntimeAborted", reconciliation=reference
                )
            )
        abort_attempt = running.attempt_id
        assert abort_attempt is not None
        await owned_disk(lambda: journal.finish(abort_stage_id, abort_attempt, reference))
        return result
