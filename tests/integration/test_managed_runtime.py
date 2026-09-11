"""Test managed runtimes with real model files, HTTP fixtures and a bounded fake daemon."""

import asyncio
import hashlib
import json
import os
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from finserve.contracts.deployment import Revision
from finserve.contracts.managed_runtime import RuntimeLaunchSpec, RuntimeReceipt
from finserve.contracts.model_assets import ModelFetchSpec, SourceFile
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.http_ownership import HTTPClosureError
from finserve.registry.artifacts import ArtifactRef, LocalArtifactStore
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.managed_runtime import (
    DockerRuntime,
    create_arguments,
    freeze_file,
    inspect_runtime,
    owned_directory,
    runtime_name,
)
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.model_assets import verify_snapshot
from finserve.registry.producer_stages import ProducerStages, StageState
from finserve.registry.producer_tasks import ModelSnapshotReceipt, declare_input
from finserve.registry.runtime_build import RuntimeBuildSpec, RuntimeImage, expected_labels
from finserve.registry.runtime_stages import (
    RuntimeStopReceipt,
    launch_runtime_stage,
    load_launch,
    stop_runtime_stage,
)


def specification(tmp_path: Path) -> RuntimeLaunchSpec:
    """A tiny verified original fixture isolates launch ownership from GPU package execution."""
    directory = tmp_path / "model"
    directory.mkdir()
    content = b'{"model_type":"fixture"}'
    (directory / "config.json").write_bytes(content)
    model = ModelFetchSpec(
        repository="owned/fixture",
        revision="a" * 40,
        maximum_bytes=1024,
        files=(
            SourceFile(
                path="config.json",
                size_bytes=len(content),
                checksum_kind="sha256",
                checksum=hashlib.sha256(content).hexdigest(),
            ),
        ),
    )
    manifest = verify_snapshot(directory, model)
    image = RuntimeImage(
        specification=RuntimeBuildSpec(
            source_revision="b" * 40, model_manifest_sha256=manifest.digest()
        ),
        source_archive_sha256="c" * 64,
        image_config_digest="sha256:" + "d" * 64,
        image_manifest_digest="sha256:" + "e" * 64,
        image_local_id="sha256:" + "e" * 64,
    )
    profile = ServingProfileV1(
        engine="vllm",
        engine_version="0.29.0",
        engine_parameters_json=VLLMParameters().model_dump_json(),
        model_revision=model.revision,
        tokenizer_revision=model.revision,
        model_manifest_sha256=manifest.digest(),
        tokenizer_manifest_sha256=manifest.digest(),
        base_url="http://127.0.0.1:8060/v1",
        served_model="fixture",
    )
    revision = Revision(
        revision_id="fixture",
        model_revision=model.revision,
        tokenizer_revision=model.revision,
        source_revision=image.specification.source_revision,
        image_digest=image.image_manifest_digest,
        config_digest=profile.digest(),
        engine="vllm",
        engine_config=profile.engine_parameters_json,
    )
    return RuntimeLaunchSpec(
        image=image,
        profile=profile,
        revision=revision,
        model=model,
        model_directory=directory.resolve(),
        readiness_timeout_seconds=5,
    )


class Daemon:
    """Simulate Docker state and ambiguity while retaining the actual command log interface."""

    def __init__(self, spec: RuntimeLaunchSpec) -> None:
        """Start with no container and immutable synthetic image inspection metadata."""
        self.spec = spec
        self.container: dict[str, Any] | None = None
        self.calls: list[list[str]] = []
        self.fail_create = False
        self.image_tag = "initial"
        self.hold_create: tuple[threading.Event, threading.Event] | None = None

    def __call__(self, arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Execute one fixture daemon operation and write its retained bounded response."""
        self.calls.append(arguments)
        operation = arguments[2]
        result: object = ""
        if arguments[1] == "image":
            result = [
                {
                    "Id": self.spec.image.image_local_id,
                    "Os": "linux",
                    "Architecture": "amd64",
                    "RepoTags": [self.image_tag],
                    "Config": {"Labels": expected_labels(self.spec.image.specification)},
                }
            ]
        elif operation == "ls":
            result = "" if self.container is None else self.container["Id"]
        elif operation == "create":
            attempt = arguments[arguments.index("--name") + 1].removeprefix("finserve-runtime-")
            assert self.container is None
            self.container = self.inspection(attempt, directory)
            result = self.container["Id"]
            if self.hold_create is not None:
                self.hold_create[0].set()
                assert self.hold_create[1].wait(5)
            if self.fail_create:
                self.fail_create = False
                output.write_text("fixture connection lost after create")
                raise RuntimeError("fixture create response lost")
        else:
            assert self.container is not None
            assert arguments[-1] == self.container["Id"]
            if operation == "inspect":
                result = [self.container]
            elif operation == "start":
                self.container["State"] = {
                    "Status": "running",
                    "Running": True,
                    "StartedAt": "actual-start-01",
                }
            elif operation == "stop":
                self.container["State"]["Running"] = False
                self.container["State"]["Status"] = "exited"
            elif operation == "rm":
                assert not self.container["State"]["Running"]
                self.container = None
            elif operation != "logs":
                raise AssertionError("unexpected Docker operation")
        output.write_text(result if isinstance(result, str) else json.dumps(result))

    def inspection(self, attempt: str, directory: Path) -> dict[str, Any]:
        """Provide independently explicit Docker fields for ownership checks and tamper tests."""
        return {
            "Id": "f" * 64,
            "Name": "/" + runtime_name(attempt),
            "Image": self.spec.image.image_local_id,
            "Config": {
                "Image": self.spec.image.image_local_id,
                "Entrypoint": ["python3", "-m", "finserve.registry.engine_entrypoint"],
                "Cmd": [
                    "--profile-sha256",
                    self.spec.profile.digest(),
                    "--expected-model",
                    "fixture",
                    "--expected-base-url",
                    self.spec.profile.base_url,
                ],
                "User": "10001:10001",
                "Labels": {
                    "finserve.runtime-attempt": attempt,
                    "finserve.runtime-specification": self.spec.digest(),
                },
            },
            "HostConfig": {
                "ReadonlyRootfs": True,
                "Privileged": False,
                "Memory": self.spec.memory_mib * 1024**2,
                "MemorySwap": self.spec.memory_mib * 1024**2,
                "PidsLimit": self.spec.pids_limit,
                "NetworkMode": "bridge",
                "ShmSize": 1024**3,
                "Tmpfs": {"/tmp": "rw,exec,nosuid,size=2147483648"},
                "DeviceRequests": [
                    {
                        "Driver": "",
                        "Count": -1,
                        "DeviceIDs": None,
                        "Capabilities": [["gpu"]],
                        "Options": {},
                    }
                ],
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                "PidMode": "",
                "IpcMode": "private",
                "UTSMode": "",
                "UsernsMode": "",
                "PublishAllPorts": False,
                "AutoRemove": False,
                "PortBindings": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8060"}]},
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": str(self.spec.model_directory),
                    "Destination": "/models",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": str(directory / "profile.json"),
                    "Destination": "/run/finserve/profile.json",
                    "RW": False,
                },
            ],
            "State": {"Status": "created", "Running": False, "StartedAt": "not-started"},
        }


def handler(request: httpx.Request) -> httpx.Response:
    """Actual HTTP response parsing must discover the alias and observe generated content/usage."""
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "fixture"}]})
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=(
            'data: {"choices":[{"index":0,"text":"hello","finish_reason":"length"}]}\n\n'
            'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
        ),
    )


def upstream(journal: ProducerStages, spec: RuntimeLaunchSpec, tmp_path: Path) -> RuntimeLaunchSpec:
    """Freeze actual tiny model bytes and synthetic image evidence without claiming a real build."""
    root = tmp_path / "models"
    root.mkdir()
    directory = root / spec.model.digest()
    spec.model_directory.rename(directory)
    spec = RuntimeLaunchSpec.model_validate({**spec.model_dump(), "model_directory": directory})
    manifest = journal.artifacts.put(verify_snapshot(directory, spec.model).canonical().encode())
    declare_input(
        journal,
        "job:model",
        {"model_root": str(root), "specification": json.loads(spec.model.canonical())},
    )
    state = journal.start("job:model")
    assert state.attempt_id is not None
    model = ModelSnapshotReceipt(
        directory=directory, specification_sha256=spec.model.digest(), manifest=manifest
    )
    journal.finish(
        "job:model", state.attempt_id, journal.artifacts.put(model.model_dump_json().encode())
    )
    declare_input(
        journal,
        "job:build",
        {
            "kind": "runtime-image-v1",
            "model_stage_id": "job:model",
            "model_manifest": manifest.model_dump(),
            "specification": spec.image.specification.model_dump(mode="json"),
        },
    )
    state = journal.start("job:build")
    assert state.attempt_id is not None
    journal.finish(
        "job:build", state.attempt_id, journal.artifacts.put(spec.image.model_dump_json().encode())
    )
    return spec


async def test_runtime_stages_reconcile_replay_and_stop(tmp_path: Path) -> None:
    """A lost daemon response keeps its journal attempt and completed replay cannot recreate it."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
        spec = upstream(journal, specification(tmp_path), tmp_path)
        daemon = Daemon(spec)
        daemon.fail_create = True
        runtime = DockerRuntime(daemon)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RuntimeError, match="response lost"):
                await launch_runtime_stage(
                    journal,
                    "job:launch",
                    "job:model",
                    "job:build",
                    spec,
                    tmp_path / "work",
                    client,
                    runtime,
                )
            running = journal.state("job:launch")
            assert running.status == "running"
            receipt = await launch_runtime_stage(
                journal,
                "job:launch",
                "job:model",
                "job:build",
                spec,
                tmp_path / "work",
                client,
                runtime,
            )
            assert receipt.attempt_id == running.attempt_id
            assert (
                await launch_runtime_stage(
                    journal,
                    "job:launch",
                    "job:model",
                    "job:build",
                    spec,
                    tmp_path / "work",
                    client,
                    runtime,
                )
                == receipt
            )
            stopped = await stop_runtime_stage(journal, "job:stop", "job:launch", spec, runtime)
            assert (
                await stop_runtime_stage(journal, "job:stop", "job:launch", spec, runtime)
                == stopped
            )
            with pytest.raises(AssertionError):
                await launch_runtime_stage(
                    journal,
                    "job:launch",
                    "job:model",
                    "job:build",
                    spec,
                    tmp_path / "work",
                    client,
                    runtime,
                )
        assert sum(call[2] == "create" for call in daemon.calls) == 1
        assert sum(call[2] == "start" for call in daemon.calls) == 1
        assert journal.state("job:launch").attempt_number == 1
        assert journal.state("job:stop").status == "completed"
    finally:
        registry.close()


@pytest.mark.parametrize(
    "fault", ["model", "image", "build_input", "uncompleted", "changed_launch"]
)
async def test_runtime_stage_rejects_unbound_upstream(tmp_path: Path, fault: str) -> None:
    """Stage linkage and actual model contents are checked before any Docker action."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
        spec = upstream(journal, specification(tmp_path), tmp_path)
        daemon = Daemon(spec)
        build_id = "job:build"
        if fault == "model":
            (spec.model_directory / "config.json").write_bytes(b"tampered")
        elif fault == "image":
            spec = spec.model_copy(
                update={"image": spec.image.model_copy(update={"source_archive_sha256": "9" * 64})}
            )
        elif fault in {"build_input", "uncompleted"}:
            build_id = "job:other"
            declare_input(journal, build_id, {})
            if fault == "build_input":
                state = journal.start(build_id)
                assert state.attempt_id is not None
                journal.finish(
                    build_id,
                    state.attempt_id,
                    journal.artifacts.put(spec.image.model_dump_json().encode()),
                )
        else:
            declare_input(journal, "job:launch", {"changed": True})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises((ValueError, RegistryConflict)):
                await launch_runtime_stage(
                    journal,
                    "job:launch",
                    "job:model",
                    build_id,
                    spec,
                    tmp_path / "work",
                    client,
                    DockerRuntime(daemon),
                )
        assert daemon.calls == []
    finally:
        registry.close()


@pytest.mark.parametrize("fault", ["receipt", "before", "http", "during"])
async def test_observe_never_restarts_unhealthy_runtime(tmp_path: Path, fault: str) -> None:
    """Completed replay verifies the same start before and after HTTP and cannot repair it."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        receipt = await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    assert daemon.container is not None
    if fault == "receipt":
        receipt = receipt.model_copy(update={"specification_sha256": "0" * 64})
    elif fault == "before":
        daemon.container["State"]["StartedAt"] = "different-start"

    def changed(request: httpx.Request) -> httpx.Response:
        """Fault only the observation phase after the initial successful launch."""
        if fault == "during":
            assert daemon.container is not None
            daemon.container["State"]["StartedAt"] = "different-start"
        return httpx.Response(503) if fault == "http" else handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(changed)) as client:
        with pytest.raises((ValueError, RuntimeError)):
            await runtime.observe(spec, receipt, client)
    assert sum(call[2] == "start" for call in daemon.calls) == 1


@pytest.mark.parametrize("different_start", [False, True])
async def test_competing_runtime_observations_preserve_exact_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, different_start: bool
) -> None:
    """Concurrent completion may change its clock observation, but cannot substitute a start."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
        spec = upstream(journal, specification(tmp_path), tmp_path)
        original = journal.finish

        def race(stage_id: str, attempt_id: str, output: ArtifactRef) -> StageState:
            """Publish the other reconciler's immutable receipt immediately before this CAS."""
            receipt = RuntimeReceipt.model_validate_json(journal.artifacts.get(output))
            other = receipt.model_copy(
                update={
                    "observed_at": receipt.observed_at + 1,
                    "container_started_at": "different-start"
                    if different_start
                    else receipt.container_started_at,
                }
            )
            original(stage_id, attempt_id, journal.artifacts.put(other.model_dump_json().encode()))
            raise RegistryConflict("competing completion")

        monkeypatch.setattr(journal, "finish", race)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            action = launch_runtime_stage(
                journal,
                "job:launch",
                "job:model",
                "job:build",
                spec,
                tmp_path / "work",
                client,
                DockerRuntime(Daemon(spec)),
            )
            if different_start:
                with pytest.raises(RegistryConflict, match="identity changed"):
                    await action
            else:
                receipt = await action
                assert receipt == load_launch(journal, journal.state("job:launch"), spec)
    finally:
        registry.close()


async def test_runtime_stage_receipts_require_matching_frozen_identity(tmp_path: Path) -> None:
    """Even manually published journal output cannot stand in for a verified matching launch."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
        spec = upstream(journal, specification(tmp_path), tmp_path)
        planned = declare_input(journal, "job:planned", {})
        with pytest.raises(ValueError, match="not completed"):
            load_launch(journal, planned, spec)
        runtime = DockerRuntime(Daemon(spec))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            receipt = await launch_runtime_stage(
                journal,
                "job:launch",
                "job:model",
                "job:build",
                spec,
                tmp_path / "work",
                client,
                runtime,
            )
        state = journal.state("job:launch")
        with pytest.raises(ValueError, match="differs from its stage"):
            load_launch(journal, state, spec.model_copy(update={"memory_mib": 9000}))
        assert state.output is not None
        declare_input(
            journal,
            "job:stop",
            {
                "kind": "managed-runtime-stop-v1",
                "launch_stage_id": "job:launch",
                "launch": state.output.model_dump(),
                "specification_sha256": spec.digest(),
            },
        )
        running = journal.start("job:stop")
        assert running.attempt_id is not None
        wrong = RuntimeStopReceipt(
            launch=state.output, container_id="0" * 64, observed_at=receipt.observed_at
        )
        journal.finish(
            "job:stop", running.attempt_id, journal.artifacts.put(wrong.model_dump_json().encode())
        )
        with pytest.raises(ValueError, match="cleanup receipt identity"):
            await stop_runtime_stage(journal, "job:stop", "job:launch", spec, runtime)
    finally:
        registry.close()


@pytest.mark.parametrize("stale", [False, True])
async def test_competing_stop_completion_is_exactly_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    """A simultaneous verified cleanup shares completion; a superseded attempt cannot do so."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
        spec = upstream(journal, specification(tmp_path), tmp_path)
        runtime = DockerRuntime(Daemon(spec))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await launch_runtime_stage(
                journal,
                "job:launch",
                "job:model",
                "job:build",
                spec,
                tmp_path / "work",
                client,
                runtime,
            )
        original = journal.finish

        def race(stage_id: str, attempt_id: str, output: ArtifactRef) -> StageState:
            """Interleave another completion or a superseding retry between read and CAS."""
            if stale:
                journal.fail(
                    stage_id,
                    attempt_id,
                    "Reconciled",
                    reconciliation=journal.artifacts.put(b"owned cleanup completed"),
                )
                journal.start(stage_id)
            else:
                other = RuntimeStopReceipt.model_validate_json(journal.artifacts.get(output))
                other = other.model_copy(update={"observed_at": other.observed_at + 1})
                original(
                    stage_id, attempt_id, journal.artifacts.put(other.model_dump_json().encode())
                )
            raise RegistryConflict("competing stop")

        monkeypatch.setattr(journal, "finish", race)
        if stale:
            with pytest.raises(RegistryConflict, match="cleanup attempt changed"):
                await stop_runtime_stage(journal, "job:stop", "job:launch", spec, runtime)
        else:
            first = await stop_runtime_stage(journal, "job:stop", "job:launch", spec, runtime)
            assert (
                await stop_runtime_stage(journal, "job:stop", "job:launch", spec, runtime) == first
            )
    finally:
        registry.close()


async def test_managed_runtime_launch_reconcile_and_stop(tmp_path: Path) -> None:
    """A lost create receipt resumes the same exact container, then verifies HTTP before removal."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    daemon.fail_create = True
    runtime = DockerRuntime(daemon)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="response lost"):
            await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
        receipt = await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
        daemon.image_tag = "retagged-same-immutable-image"
        repeated = await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    assert receipt.container_id == repeated.container_id == "f" * 64
    assert sum(call[2] == "create" for call in daemon.calls) == 1
    assert sum(call[2] == "start" for call in daemon.calls) == 1
    assert len(list(receipt.output_directory.glob("*-smoke.json"))) == 2
    await runtime.stop(spec, receipt)
    assert daemon.container is None
    await runtime.stop(spec, receipt)
    assert sum(call[2] == "rm" for call in daemon.calls) == 1


async def test_cancelled_create_is_reconciled_without_second_container(tmp_path: Path) -> None:
    """Cancellation drains the CLI but retains daemon ambiguity until exact named lookup."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    entered, release = threading.Event(), threading.Event()
    daemon.hold_create = (entered, release)
    runtime = DockerRuntime(daemon)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(runtime.launch(spec, "a" * 32, tmp_path / "work", client))
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        receipt = await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    assert receipt.container_id == "f" * 64
    assert sum(call[2] == "create" for call in daemon.calls) == 1


@pytest.mark.parametrize("target", ["mount", "image", "profile", "restart", "environment"])
async def test_foreign_runtime_cannot_be_stopped(tmp_path: Path, target: str) -> None:
    """Changed identity fails before a stop command, even when the caller retains an old receipt."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        receipt = await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    assert daemon.container is not None
    if target == "mount":
        daemon.container["Mounts"][0]["RW"] = True
    elif target == "image":
        daemon.container["Image"] = "sha256:" + "0" * 64
    elif target == "profile":
        (receipt.output_directory / "profile.json").write_text("changed")
    elif target == "environment":
        daemon.container["Config"]["Env"] = ["LD_PRELOAD=/foreign.so"]
    else:
        daemon.container["State"]["StartedAt"] = "different-start"
    with pytest.raises(ValueError):
        await runtime.stop(spec, receipt)
    assert not any(call[2] == "stop" for call in daemon.calls)


def test_launch_canonical_and_command_boundary(tmp_path: Path) -> None:
    """Round-trip identities are stable and shell/mount escape inputs cannot reach Docker."""
    spec = specification(tmp_path)
    assert (
        spec.canonical()
        == RuntimeLaunchSpec.model_validate_json(spec.model_dump_json()).canonical()
    )
    arguments = create_arguments(spec, "a" * 32, tmp_path / "work")
    assert "--pull" in arguments and "never" in arguments and "--read-only" in arguments
    with pytest.raises(ValueError):
        runtime_name("../foreign")
    with pytest.raises(ValueError):
        create_arguments(spec, "a" * 32, tmp_path / "bad,mount")


@pytest.mark.parametrize("change", ["endpoint", "engine", "image"])
def test_launch_rejects_unbound_runtime_inputs(tmp_path: Path, change: str) -> None:
    """Matching superficial aliases cannot authorize a remote endpoint or a different runtime."""
    value = specification(tmp_path).model_dump()
    profile = value["profile"]
    if change == "endpoint":
        profile["base_url"] = "http://remote:8060/v1"
    elif change == "engine":
        profile["engine_version"] = "0.28.0"
    else:
        value["revision"]["image_digest"] = "sha256:" + "0" * 64
    updated = ServingProfileV1.model_validate(profile)
    value["revision"]["config_digest"] = updated.digest()
    with pytest.raises(ValueError):
        RuntimeLaunchSpec.model_validate(value)


@pytest.mark.parametrize(
    "late_phase", ["image", "lookup", "created_inspect", "smoke", "started", "final_inspect"]
)
async def test_cold_budget_never_reports_late_health(tmp_path: Path, late_phase: str) -> None:
    """Verification, startup and final smoke must all finish within the frozen readiness budget."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    clock = [0.0]
    inspections = 0

    def command(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Advance observed elapsed time after a chosen external operation completes."""
        nonlocal inspections
        assert 0 < timeout <= 5
        daemon(arguments, directory, output, timeout)
        if arguments[1:3] == ["container", "inspect"]:
            inspections += 1
        if (
            (late_phase == "image" and arguments[1] == "image")
            or (late_phase == "lookup" and arguments[2] == "ls")
            or (
                late_phase == "created_inspect"
                and arguments[2] == "inspect"
                and arguments[1] == "container"
            )
            or (late_phase == "started" and arguments[2] == "start")
            or (late_phase == "final_inspect" and inspections == 3)
        ):
            clock[0] = 6.0

    def response(request: httpx.Request) -> httpx.Response:
        """A late successful HTTP response must still fail the shared launch budget."""
        if late_phase == "smoke" and request.url.path.endswith("/completions"):
            clock[0] = 6.0
        return handler(request)

    runtime = DockerRuntime(command, lambda: clock[0])
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(TimeoutError):
            await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    assert not any(call[2] in {"stop", "rm"} for call in daemon.calls)
    if late_phase in {"image", "lookup"}:
        assert not any(call[2] == "create" for call in daemon.calls)
    if late_phase == "created_inspect":
        assert not any(call[2] == "start" for call in daemon.calls)


@pytest.mark.parametrize("fault", ["wrong_alias", "http_error", "bad_json", "empty_content"])
async def test_model_discovery_and_real_content_are_required(tmp_path: Path, fault: str) -> None:
    """Container liveness alone cannot satisfy actual inference readiness."""
    spec = specification(tmp_path)

    def response(request: httpx.Request) -> httpx.Response:
        """Return one specific HTTP-level defect without changing the claimed Docker state."""
        if fault == "wrong_alias":
            return httpx.Response(200, json={"data": [{"id": "other"}]})
        if fault == "http_error":
            return httpx.Response(503)
        if fault == "bad_json":
            return httpx.Response(200, text="not JSON")
        if request.url.path.endswith("/models"):
            return handler(request)
        return httpx.Response(200, content="data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        assert not await DockerRuntime(Daemon(spec)).probe_endpoint(spec, client, tmp_path)


async def test_runtime_restart_during_smoke_is_not_healthy(tmp_path: Path) -> None:
    """A successful response cannot attest a container that restarted during observation."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)

    def response(request: httpx.Request) -> httpx.Response:
        """Simulate a daemon restart while an HTTP completion is in progress."""
        if request.url.path.endswith("/completions"):
            assert daemon.container is not None
            daemon.container["State"]["StartedAt"] = "restart-during-smoke"
        return handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(RuntimeError, match="changed"):
            await DockerRuntime(daemon).launch(spec, "a" * 32, tmp_path / "work", client)


async def test_readiness_retries_transient_discovery_only(tmp_path: Path) -> None:
    """A starting server can become ready without issuing another Docker create or start."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    calls = 0

    def response(request: httpx.Request) -> httpx.Response:
        """The first discovery request is unavailable; subsequent protocol requests succeed."""
        nonlocal calls
        calls += 1
        return httpx.Response(503) if calls == 1 else handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        receipt = await DockerRuntime(daemon).launch(spec, "a" * 32, tmp_path / "work", client)
    assert receipt.container_id == "f" * 64 and calls == 3


@pytest.mark.parametrize("before_start", [False, True])
async def test_exited_runtime_cannot_be_restarted_implicitly(
    tmp_path: Path, before_start: bool
) -> None:
    """An exited engine needs a distinct reconciled attempt rather than a hidden restart."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    directory = tmp_path / "work" / ("a" * 32)
    if before_start:
        daemon.container = daemon.inspection("a" * 32, directory)
        daemon.container["State"]["Status"] = "exited"

    def command(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """A newly started fixture can exit before its first health observation."""
        daemon(arguments, directory, output, timeout)
        if arguments[2] == "start":
            assert daemon.container is not None
            daemon.container["State"].update({"Running": False, "Status": "exited"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="exited"):
            await DockerRuntime(command).launch(spec, "a" * 32, tmp_path / "work", client)


@pytest.mark.parametrize("document", [{}, []])
def test_inspection_requires_one_observed_container(tmp_path: Path, document: object) -> None:
    """Missing or malformed inspection output cannot become positive ownership evidence."""
    with pytest.raises(ValueError):
        inspect_runtime(document, specification(tmp_path), "a" * 32, tmp_path)


async def test_noncanonical_model_path_fails_before_docker(tmp_path: Path) -> None:
    """A lexical path alias fails before commands can create ambiguous mounts."""
    original = specification(tmp_path)
    spec = original.model_copy(
        update={"model_directory": original.model_directory / ".." / "model"}
    )
    daemon = Daemon(spec)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="canonical"):
            await DockerRuntime(daemon).launch(spec, "a" * 32, tmp_path / "work", client)
    assert not daemon.calls


@pytest.mark.parametrize("mismatch", ["specification", "directory"])
async def test_stop_rejects_wrong_receipt_before_commands(tmp_path: Path, mismatch: str) -> None:
    """A caller cannot redirect cleanup by altering its launch digest or attempt directory."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        receipt = await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    if mismatch == "specification":
        receipt = receipt.model_copy(update={"specification_sha256": "0" * 64})
    else:
        receipt = receipt.model_copy(update={"output_directory": tmp_path})
    before = len(daemon.calls)
    with pytest.raises(ValueError):
        await runtime.stop(spec, receipt)
    assert len(daemon.calls) == before


@pytest.mark.parametrize("already_exited", [False, True])
async def test_removal_requires_observed_stopped_state(
    tmp_path: Path, already_exited: bool
) -> None:
    """A failed stop remains unresolved; an already-exited owned start can be removed safely."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)

    def command(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Simulate a daemon that accepts stop but continues reporting the process running."""
        daemon(arguments, directory, output, timeout)
        if arguments[2] == "stop":
            assert daemon.container is not None
            daemon.container["State"]["Running"] = True

    runtime = DockerRuntime(command)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        receipt = await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    assert daemon.container is not None
    if already_exited:
        daemon.container["State"].update({"Running": False, "Status": "exited"})
        await runtime.stop(spec, receipt)
        assert daemon.container is None
    else:
        with pytest.raises(RuntimeError, match="not verified"):
            await runtime.stop(spec, receipt)
        assert not any(call[2] == "rm" for call in daemon.calls)


async def test_failed_probe_cleanup_retains_row_and_ambiguity(tmp_path: Path) -> None:
    """Transport cleanup failure cannot silently turn into another readiness request."""
    spec = specification(tmp_path)

    class Stream(httpx.AsyncByteStream):
        """Return content but fail the local close operation at EOF."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """Preserve actual content before the independently failing cleanup."""
            yield b'data: {"choices":[{"index":0,"text":"hello","finish_reason":"length"}]}\n\n'
            yield b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'

        async def aclose(self) -> None:
            """The resource-release observation remains unknown."""
            raise OSError("fixture cleanup failed")

    def response(request: httpx.Request) -> httpx.Response:
        """Discovery succeeds while the actual completion stream cannot be cleanly closed."""
        return (
            handler(request)
            if request.url.path.endswith("/models")
            else httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"})
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(HTTPClosureError):
            await DockerRuntime(Daemon(spec)).probe_endpoint(spec, client, tmp_path)
    rows = await asyncio.to_thread(lambda: list(tmp_path.glob("*-smoke.json")))
    assert len(rows) == 1
    value = json.loads(rows[0].read_text())
    assert (
        value["error"] == "HTTPClosureError" and value["output"] == "hello" and not value["success"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("CapAdd", ["SYS_ADMIN"]),
        ("Devices", [{"PathOnHost": "/dev/foreign"}]),
        ("Binds", ["/foreign:/extra"]),
        ("SecurityOpt", ["seccomp=unconfined"]),
        ("RestartPolicy", {"Name": "always", "MaximumRetryCount": 0}),
    ],
)
async def test_extra_host_privileges_prevent_adoption(
    tmp_path: Path, field: str, value: object
) -> None:
    """A matching label does not authorize an existing container with extra execution privileges."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
        assert daemon.container is not None
        daemon.container["HostConfig"][field] = value
        before = len(daemon.calls)
        with pytest.raises(ValueError, match="frozen launch"):
            await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    assert not any(call[2] in {"start", "stop", "rm"} for call in daemon.calls[before:])


@pytest.mark.parametrize("location", ["relative", "repository"])
async def test_stop_cannot_write_outside_external_attempt(tmp_path: Path, location: str) -> None:
    """An altered receipt is rejected before creating a profile or CLI log in another namespace."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        receipt = await runtime.launch(spec, "a" * 32, tmp_path / "work", client)
    destination = (
        Path("a" * 32)
        if location == "relative"
        else (await asyncio.to_thread(Path(__file__).resolve)).parents[2] / ("a" * 32)
    )
    changed = receipt.model_copy(update={"output_directory": destination})
    before = len(daemon.calls)
    with pytest.raises(ValueError, match="external"):
        await runtime.stop(spec, changed)
    assert len(daemon.calls) == before and not destination.exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation requires local privilege")
async def test_attempt_directory_symlink_cannot_redirect_writes(tmp_path: Path) -> None:
    """A preexisting workspace symlink cannot become an owned attempt directory."""
    spec = specification(tmp_path)
    workspace, foreign = tmp_path / "work", tmp_path / "foreign"
    workspace.mkdir()
    foreign.mkdir()
    (workspace / ("a" * 32)).symlink_to(foreign, target_is_directory=True)
    daemon = Daemon(spec)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="external"):
            await DockerRuntime(daemon).launch(spec, "a" * 32, workspace, client)
    assert not daemon.calls and not list(foreign.iterdir())


def test_missing_directory_and_symlinked_frozen_file(tmp_path: Path) -> None:
    """A receipt cannot recreate a missing attempt namespace or replace a symlinked input."""
    with pytest.raises(ValueError, match="missing"):
        owned_directory(tmp_path / "missing")
    if os.name != "nt":
        (tmp_path / "target").write_text("same")
        (tmp_path / "input").symlink_to(tmp_path / "target")
        with pytest.raises(ValueError, match="changed"):
            freeze_file(tmp_path / "input", "same")


@pytest.mark.parametrize(
    "fault", ["oversize_lookup", "ambiguous_lookup", "wrong_id", "lost_container"]
)
async def test_ambiguous_daemon_observation_never_reports_ready(tmp_path: Path, fault: str) -> None:
    """Malformed lookups and mismatched inspection IDs cannot turn into new container ownership."""
    spec = specification(tmp_path)
    daemon = Daemon(spec)

    def command(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Corrupt one daemon observation after executing its explicit fixture operation."""
        daemon(arguments, directory, output, timeout)
        if arguments[2] == "ls" and fault in {"oversize_lookup", "ambiguous_lookup"}:
            output.write_text("x" * 4097 if fault == "oversize_lookup" else "two IDs")
        elif arguments[1:3] == ["container", "inspect"] and fault == "wrong_id":
            document = json.loads(output.read_text())
            document[0]["Id"] = "0" * 64
            output.write_text(json.dumps(document))
        elif arguments[2] == "create" and fault == "lost_container":
            daemon.container = None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises((ValueError, RuntimeError)):
            await DockerRuntime(command).launch(spec, "a" * 32, tmp_path / "work", client)
    assert not any(call[2] == "start" for call in daemon.calls)


async def test_actual_model_manifest_must_match_image_binding(tmp_path: Path) -> None:
    """Consistent declarations cannot substitute a manifest that differs from actual model bytes."""
    value = specification(tmp_path).model_dump()
    value["profile"]["model_manifest_sha256"] = "0" * 64
    value["profile"]["tokenizer_manifest_sha256"] = "0" * 64
    value["image"]["specification"]["model_manifest_sha256"] = "0" * 64
    value["revision"]["config_digest"] = ServingProfileV1.model_validate(value["profile"]).digest()
    spec = RuntimeLaunchSpec.model_validate(value)
    daemon = Daemon(spec)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="snapshot"):
            await DockerRuntime(daemon).launch(spec, "a" * 32, tmp_path / "work", client)
    assert not daemon.calls
