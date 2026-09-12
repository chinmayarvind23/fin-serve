"""Exercise failed-start reconciliation without Docker, including separate executor processes."""

import asyncio
import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_managed_runtime import Daemon, handler, specification, upstream

from finserve.contracts.managed_runtime import RuntimeLaunchSpec
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import declare_input
from finserve.registry.runtime_abort import abort_runtime_stage
from finserve.registry.runtime_fence import RUNTIME_OPERATION_PROTOCOL, attempt_fence
from finserve.registry.runtime_stages import launch_runtime_stage

pytestmark = pytest.mark.skipif(os.name != "posix", reason="runtime fence fails closed off POSIX")


def setup_launch(
    tmp_path: Path, *, legacy: bool = False
) -> tuple[ProducerStages, RuntimeLaunchSpec, str]:
    """Declare exactly the immutable upstream linkage consumed by the normal stage API."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
    spec = upstream(journal, specification(tmp_path), tmp_path)
    model, image = journal.state("job:model").output, journal.state("job:build").output
    assert model is not None and image is not None
    declare_input(
        journal,
        "job:launch",
        {
            "kind": "managed-runtime-launch-v1",
            **({} if legacy else {"operation_protocol": RUNTIME_OPERATION_PROTOCOL}),
            "specification": json.loads(spec.canonical()),
            "workspace": str(tmp_path / "runtime"),
            "model_stage_id": "job:model",
            "build_stage_id": "job:build",
            "model": model.model_dump(),
            "image": image.model_dump(),
        },
    )
    state = journal.start("job:launch")
    assert state.attempt_id is not None
    (tmp_path / "runtime").mkdir()
    return journal, spec, state.attempt_id


async def fail_readiness(*args: Any, **kwargs: Any) -> bool:
    """Fail after allocation/first-start persistence without fabricating health evidence."""
    raise RuntimeError("fixture readiness failed")


@pytest.mark.parametrize("phase", ["absent", "created", "started"])
async def test_abort_reconciles_failed_start_and_replay(tmp_path: Path, phase: str) -> None:
    """Each incomplete phase yields failure evidence and no readiness receipt."""
    journal, spec, attempt = setup_launch(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    runtime.probe_endpoint = fail_readiness
    daemon.fail_create = phase == "created"
    try:
        if phase != "absent":
            async with httpx.AsyncClient() as client:
                with pytest.raises(RuntimeError):
                    await runtime.launch(spec, attempt, tmp_path / "runtime", client)
        result = await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert daemon.container is None
        assert result.outcome == ("absent" if phase == "absent" else "stopped-and-removed")
        assert result.inspections and journal.state("job:launch").output is None
        assert journal.state("job:launch").status == "failed"
        assert journal.state("job:launch").reconciliation is not None
        assert (
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
            == result
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(RuntimeError, match="abort intent"):
                await runtime.launch(spec, attempt, tmp_path / "runtime", client)
        assert [state.status for state in journal.history("job:launch")] == [
            "planned",
            "running",
            "failed",
        ]
    finally:
        journal.registry.close()


@pytest.mark.parametrize("fault", ["restart", "image", "spec", "legacy", "daemon"])
async def test_abort_rejects_uncertain_ownership(tmp_path: Path, fault: str) -> None:
    """Matching labels never authorize a changed start/image/spec or a failed daemon query."""
    journal, spec, attempt = setup_launch(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    runtime.probe_endpoint = fail_readiness
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises(RuntimeError):
                await runtime.launch(spec, attempt, tmp_path / "runtime", client)
        assert daemon.container is not None
        if fault == "restart":
            daemon.container["State"]["StartedAt"] = "restarted"
        elif fault == "image":
            daemon.container["Image"] = "sha256:" + "0" * 64
        elif fault == "spec":
            daemon.container["HostConfig"]["Memory"] += 1
        elif fault == "legacy":
            (tmp_path / "runtime" / attempt / "first-start.json").unlink()
        else:

            def unavailable(
                arguments: list[str], directory: Path, output: Path, timeout: float
            ) -> None:
                """No daemon observation is possible, regardless of the queried identity."""
                raise RuntimeError("daemon unavailable")

            runtime.command = unavailable
        before = len(daemon.calls)
        with pytest.raises((RegistryConflict, ValueError, RuntimeError)):
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert not any(call[2] in {"stop", "rm"} for call in daemon.calls[before:])
        assert journal.state("job:launch").status == "running"
        assert (tmp_path / "runtime" / attempt / "abort-intent.json").is_file()
    finally:
        journal.registry.close()


async def test_completed_launch_and_wrong_attempt_are_never_aborted(tmp_path: Path) -> None:
    """Explicit abort cannot stop traffic-owned work or a stale attempt."""
    journal, spec, attempt = setup_launch(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    try:
        with pytest.raises(RegistryConflict):
            await abort_runtime_stage(journal, "job:abort", "job:launch", "0" * 32, runtime)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await launch_runtime_stage(
                journal,
                "job:launch",
                "job:model",
                "job:build",
                spec,
                tmp_path / "runtime",
                client,
                runtime,
            )
        before = len(daemon.calls)
        with pytest.raises(RegistryConflict):
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert len(daemon.calls) == before
    finally:
        journal.registry.close()


@pytest.mark.parametrize("crash_phase", ["before_fail", "after_fail"])
async def test_removal_publication_crash_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_phase: str
) -> None:
    """Removal and journal publication need not be atomic when durable intent forbids launch."""
    journal, spec, attempt = setup_launch(tmp_path)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    runtime.probe_endpoint = fail_readiness
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError):
            await runtime.launch(spec, attempt, tmp_path / "runtime", client)
    original = journal.fail

    def crash(*args: Any, **kwargs: Any) -> Any:
        """Inject executor failure on either side of launch-failure publication."""
        if crash_phase == "after_fail":
            original(*args, **kwargs)
        raise RuntimeError("publication crash")

    monkeypatch.setattr(journal, "fail", crash)
    try:
        with pytest.raises(RuntimeError, match="publication crash"):
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        monkeypatch.setattr(journal, "fail", original)
        result = await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert daemon.container is None
        assert result.outcome == (
            "absent" if crash_phase == "before_fail" else "stopped-and-removed"
        )
        assert journal.state("job:abort").status == "completed"
    finally:
        journal.registry.close()


async def test_cancelled_abort_drains_stop_before_unlock(tmp_path: Path) -> None:
    """Repeated cancellation cannot release the fence while its owned stop still executes."""
    journal, spec, attempt = setup_launch(tmp_path)
    daemon = Daemon(spec)
    entered, release = threading.Event(), threading.Event()

    def command(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Hold the actual stop worker to expose premature unlock and auxiliary log failure."""
        if arguments[2] == "stop":
            entered.set()
            assert release.wait(10)
        if arguments[2] == "logs":
            raise OSError("log writer unavailable")
        daemon(arguments, directory, output, timeout)

    runtime = DockerRuntime(command)
    runtime.probe_endpoint = fail_readiness
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises(RuntimeError):
                await runtime.launch(spec, attempt, tmp_path / "runtime", client)
        task = asyncio.create_task(
            abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        )
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        task.cancel()
        peer = asyncio.create_task(
            abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        )
        await asyncio.sleep(0.1)
        assert not task.done() and not peer.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await peer
        assert result.log_error == "OSError" and daemon.container is None
    finally:
        release.set()
        journal.registry.close()


class FileDaemon(Daemon):
    """Share fixture daemon state across real executor processes through a small JSON file."""

    def __init__(self, spec: RuntimeLaunchSpec, root: Path) -> None:
        """The runtime fence should serialize every access to these deliberately unlocked files."""
        super().__init__(spec)
        self.root = root

    def __call__(self, arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Pause before create, so an unfenced abort could incorrectly publish absence."""
        state = self.root / "daemon.json"
        self.container = json.loads(state.read_text()) if state.exists() else None
        if arguments[2] == "create":
            (self.root / "create-entered").touch()
            deadline = time.monotonic() + 15
            while not (self.root / "release-create").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("test create was not released")
                time.sleep(0.02)
            self.fail_create = True
        with (self.root / "calls").open("a") as stream:
            stream.write(arguments[2] + "\n")
        try:
            super().__call__(arguments, directory, output, timeout)
        finally:
            state.write_text(json.dumps(self.container))


def process_operation(root: str, specification_json: str, attempt: str, mode: str) -> None:
    """Construct independent registry/runtime objects in a separate OS process."""

    async def run() -> None:
        """Retain process outcomes as files so assertion failures remain diagnosable."""
        path = Path(root)
        registry = Registry("sqlite:///" + str(path / "registry.sqlite"))
        journal = ProducerStages(registry, LocalArtifactStore(path / "artifacts"))
        spec = RuntimeLaunchSpec.model_validate_json(specification_json)
        runtime = DockerRuntime(FileDaemon(spec, path))
        try:
            if mode == "launch":
                async with httpx.AsyncClient() as client:
                    await launch_runtime_stage(
                        journal,
                        "job:launch",
                        "job:model",
                        "job:build",
                        spec,
                        path / "runtime",
                        client,
                        runtime,
                    )
            else:
                await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
            (path / (mode + "-result")).write_text("ok")
        except Exception as error:
            (path / (mode + "-result")).write_text(type(error).__name__ + ":" + str(error))
        finally:
            registry.close()

    asyncio.run(run())


async def test_interprocess_create_cannot_run_after_abort_absence(tmp_path: Path) -> None:
    """A second interpreter must wait through create ambiguity before removing that allocation."""
    journal, spec, attempt = setup_launch(tmp_path)
    context = multiprocessing.get_context("spawn")
    arguments = (str(tmp_path), spec.model_dump_json(), attempt)
    launch = context.Process(target=process_operation, args=(*arguments, "launch"))
    abort = context.Process(target=process_operation, args=(*arguments, "abort"))
    try:
        launch.start()
        async with asyncio.timeout(15):
            while not (tmp_path / "create-entered").exists():  # noqa: ASYNC110
                await asyncio.sleep(0.02)
        abort.start()
        await asyncio.sleep(0.5)
        assert not (tmp_path / "abort-result").exists()
        assert journal.state("job:launch").status == "running"
        (tmp_path / "release-create").touch()
        await asyncio.to_thread(launch.join, 15)
        await asyncio.to_thread(abort.join, 15)
        assert launch.exitcode == abort.exitcode == 0
        assert (tmp_path / "abort-result").read_text() == "ok"
        assert "create response lost" in (tmp_path / "launch-result").read_text()
        assert json.loads((tmp_path / "daemon.json").read_text()) is None
        assert journal.state("job:launch").status == "failed"
        calls = (tmp_path / "calls").read_text().splitlines()
        assert calls.count("create") == calls.count("rm") == 1
    finally:
        (tmp_path / "release-create").touch()
        for process in (launch, abort):
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)
        journal.registry.close()


def hold_fence(root: str) -> None:
    """Keep an acquired kernel lock until the test kills this separate process."""

    async def run() -> None:
        """Persist an abort-intent stand-in to demonstrate that death releases only the lock."""
        path = Path(root)
        async with attempt_fence(path):
            (path / "locked").touch()
            await asyncio.sleep(60)

    asyncio.run(run())


async def test_kernel_fence_recovers_after_process_death(tmp_path: Path) -> None:
    """Kernel ownership, unlike a persistent lockfile flag or lease, expires on process death."""
    process = multiprocessing.get_context("spawn").Process(target=hold_fence, args=(str(tmp_path),))
    try:
        process.start()
        async with asyncio.timeout(15):
            while not (tmp_path / "locked").exists():  # noqa: ASYNC110
                await asyncio.sleep(0.02)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1), attempt_fence(tmp_path):
                pytest.fail("live peer lock was bypassed")
        process.terminate()
        await asyncio.to_thread(process.join, 5)
        async with asyncio.timeout(2), attempt_fence(tmp_path):
            assert (tmp_path / "operation.lock").is_file()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


async def test_unacknowledged_create_absence_is_not_terminal(tmp_path: Path) -> None:
    """An accepted Docker request may finish after CLI timeout, even after an empty listing."""
    journal, spec, attempt = setup_launch(tmp_path)
    daemon = Daemon(spec)

    def timed_out(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Leave the daemon-side create unresolved without inventing a container observation."""
        if arguments[2] == "create":
            raise TimeoutError("accepted create has not completed")
        daemon(arguments, directory, output, timeout)

    runtime = DockerRuntime(timed_out)
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises(TimeoutError):
                await runtime.launch(spec, attempt, tmp_path / "runtime", client)
            with pytest.raises(RuntimeError, match="requested runtime absent"):
                await runtime.launch(spec, attempt, tmp_path / "runtime", client)
        with pytest.raises(RegistryConflict, match="unacknowledged create"):
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert journal.state("job:launch").status == "running"
        # Once the daemon's delayed create becomes observable, exact configuration
        # validation can reconcile that created allocation without claiming readiness.
        daemon.container = daemon.inspection(attempt, tmp_path / "runtime" / attempt)
        result = await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert result.outcome == "stopped-and-removed"
    finally:
        journal.registry.close()


async def test_unbound_requested_start_cannot_be_removed(tmp_path: Path) -> None:
    """A pending daemon start cannot be mistaken for a permanently never-started container."""
    journal, spec, attempt = setup_launch(tmp_path)
    daemon = Daemon(spec)

    def timed_out(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Leave a pending start with the last daemon observation still created."""
        if arguments[2] == "start":
            raise TimeoutError("accepted start has not completed")
        daemon(arguments, directory, output, timeout)

    runtime = DockerRuntime(timed_out)
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises(TimeoutError):
                await runtime.launch(spec, attempt, tmp_path / "runtime", client)
        with pytest.raises(RegistryConflict, match="unbound requested start"):
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert not any(call[2] in {"stop", "rm"} for call in daemon.calls)
    finally:
        journal.registry.close()


@pytest.mark.parametrize("phase", ["during_stop", "after_abort"])
async def test_changed_start_never_removed_by_abort(tmp_path: Path, phase: str) -> None:
    """A restart during stop or a replacement after completed abort must remain untouched."""
    journal, spec, attempt = setup_launch(tmp_path)
    daemon = Daemon(spec)

    def command(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Inject a restart between the stop command and its required final inspection."""
        daemon(arguments, directory, output, timeout)
        if phase == "during_stop" and arguments[2] == "stop":
            assert daemon.container is not None
            daemon.container["State"]["StartedAt"] = "new-start"

    runtime = DockerRuntime(command)
    runtime.probe_endpoint = fail_readiness
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises(RuntimeError):
                await runtime.launch(spec, attempt, tmp_path / "runtime", client)
        if phase == "after_abort":
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
            daemon.container = daemon.inspection(attempt, tmp_path / "runtime" / attempt)
        before = len(daemon.calls)
        with pytest.raises(RegistryConflict):
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert not any(call[2] == "rm" for call in daemon.calls[before:])
    finally:
        journal.registry.close()


def crash_after_abort_intent(root: str, specification_json: str, attempt: str) -> None:
    """Kill an executor after durable intent but before its first daemon lookup returns."""

    async def run() -> None:
        """The actual stage API, not a hand-written marker, supplies the persisted fence."""
        path = Path(root)
        registry = Registry("sqlite:///" + str(path / "registry.sqlite"))
        journal = ProducerStages(registry, LocalArtifactStore(path / "artifacts"))

        def crash(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
            """Model abrupt death with no Python finally cleanup."""
            os._exit(73)

        await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, DockerRuntime(crash))

    asyncio.run(run())


async def test_process_death_preserves_abort_intent(tmp_path: Path) -> None:
    """A new executor cannot launch after the old abort process dies and its lock releases."""
    journal, spec, attempt = setup_launch(tmp_path)
    process = multiprocessing.get_context("spawn").Process(
        target=crash_after_abort_intent, args=(str(tmp_path), spec.model_dump_json(), attempt)
    )
    try:
        process.start()
        await asyncio.to_thread(process.join, 15)
        assert process.exitcode == 73
        daemon = Daemon(spec)
        runtime = DockerRuntime(daemon)
        async with httpx.AsyncClient() as client:
            with pytest.raises(RuntimeError, match="abort intent"):
                await launch_runtime_stage(
                    journal,
                    "job:launch",
                    "job:model",
                    "job:build",
                    spec,
                    tmp_path / "runtime",
                    client,
                    runtime,
                )
        assert not daemon.calls and journal.state("job:launch").status == "running"
        assert (
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        ).outcome == "absent"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        journal.registry.close()


@pytest.mark.parametrize("phase", ["absent", "created", "started"])
async def test_legacy_launch_cannot_be_aborted_even_with_new_adapter_markers(
    tmp_path: Path, phase: str
) -> None:
    """An older unfenced executor may still mutate a legacy attempt after any observation."""
    journal, spec, attempt = setup_launch(tmp_path, legacy=True)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    runtime.probe_endpoint = fail_readiness
    daemon.fail_create = phase == "created"
    try:
        original = journal.state("job:launch")
        if phase != "absent":
            async with httpx.AsyncClient() as client:
                with pytest.raises(RuntimeError):
                    await runtime.launch(spec, attempt, tmp_path / "runtime", client)
        before = len(daemon.calls)
        with pytest.raises(RegistryConflict, match="legacy runtime protocol"):
            await abort_runtime_stage(journal, "job:abort", "job:launch", attempt, runtime)
        assert len(daemon.calls) == before
        assert journal.state("job:launch") == original
        assert journal.history("job:abort") == []
        assert not (tmp_path / "runtime" / attempt / "abort-intent.json").exists()
    finally:
        journal.registry.close()


async def test_completed_legacy_launch_keeps_original_input_and_replay(tmp_path: Path) -> None:
    """Protocol migration must not rewrite completed historical evidence or create replacements."""
    journal, spec, _ = setup_launch(tmp_path, legacy=True)
    daemon = Daemon(spec)
    runtime = DockerRuntime(daemon)
    original = journal.state("job:launch").input
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            receipt = await launch_runtime_stage(
                journal,
                "job:launch",
                "job:model",
                "job:build",
                spec,
                tmp_path / "runtime",
                client,
                runtime,
            )
            before = len(daemon.calls)
            assert (
                await launch_runtime_stage(
                    journal,
                    "job:launch",
                    "job:model",
                    "job:build",
                    spec,
                    tmp_path / "runtime",
                    client,
                    runtime,
                )
                == receipt
            )
        assert journal.state("job:launch").input == original
        assert not any(call[2] in {"create", "start"} for call in daemon.calls[before:])
    finally:
        journal.registry.close()


async def test_new_protocol_input_rejects_legacy_executor_declaration(tmp_path: Path) -> None:
    """An old stage executor omits the new field and must fail before entering runtime work."""
    journal, spec, _ = setup_launch(tmp_path)
    runtime = DockerRuntime(Daemon(spec))
    runtime.probe_endpoint = fail_readiness
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises(RuntimeError, match="readiness failed"):
                await launch_runtime_stage(
                    journal,
                    "job:new-launch",
                    "job:model",
                    "job:build",
                    spec,
                    tmp_path / "new-runtime",
                    client,
                    runtime,
                )
        state = journal.state("job:new-launch")
        frozen = json.loads(journal.artifacts.get(state.input))
        assert frozen.pop("operation_protocol") == RUNTIME_OPERATION_PROTOCOL
        with pytest.raises(RegistryConflict, match="input identity changed"):
            declare_input(journal, "job:new-launch", frozen)
        assert journal.state("job:new-launch") == state
    finally:
        journal.registry.close()
