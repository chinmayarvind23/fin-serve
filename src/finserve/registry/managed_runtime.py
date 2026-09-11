"""Own local Docker engine attempts without adopting unrelated containers or replaying ambiguity."""

import asyncio
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import httpx

from finserve.contracts.managed_runtime import RuntimeLaunchSpec, RuntimeReceipt
from finserve.http_ownership import HTTPClosureError, own_response
from finserve.registry.model_assets import (
    bounded_response,
    owned_disk,
    sync_directory,
    verify_snapshot,
)
from finserve.registry.runtime_build import (
    CommandRunner,
    bounded_document,
    run_command,
    verify_image_inspection,
)
from finserve.registry.runtime_probe import completion_probe


def runtime_name(attempt_id: str) -> str:
    """The immutable attempt token is also a Docker name fence across ambiguous CLI results."""
    if re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
        raise ValueError("exact producer attempt token required")
    return "finserve-runtime-" + attempt_id


def owned_directory(path: Path, *, create: bool = False) -> Path:
    """Reject relative, repository-local and symlinked paths before any runtime evidence write."""
    repository = Path(__file__).resolve().parents[3]
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path == repository
        or repository in path.parents
        or any(character in str(path) for character in (",", '"', "\n", "\r"))
    ):
        raise ValueError("canonical absolute external runtime directory required")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("owned runtime directory is missing")
    return path


def freeze_file(path: Path, content: str) -> None:
    """Owned launch files are immutable, including when a task resumes after executor failure."""
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
            sync_directory(path.parent)
        except FileExistsError:
            expected = content.encode()
            if path.is_symlink():
                raise ValueError("managed runtime input changed") from None
            with path.open("rb") as existing:
                if existing.read(len(expected) + 1) != expected:
                    raise ValueError("managed runtime input changed") from None
    finally:
        temporary.unlink(missing_ok=True)


def create_arguments(spec: RuntimeLaunchSpec, attempt_id: str, directory: Path) -> list[str]:
    """Construct fixed argv with no shell, caller command, mutable image tag or writable model."""
    if any(character in str(directory) for character in (",", '"', "\n", "\r")):
        raise ValueError("Docker mount path contains unsupported delimiters")
    port = httpx.URL(spec.profile.base_url).port
    return [
        "docker",
        "container",
        "create",
        "--name",
        runtime_name(attempt_id),
        "--label",
        "finserve.runtime-attempt=" + attempt_id,
        "--label",
        "finserve.runtime-specification=" + spec.digest(),
        "--pull",
        "never",
        "--gpus",
        "all",
        "--read-only",
        "--memory",
        str(spec.memory_mib) + "m",
        "--memory-swap",
        str(spec.memory_mib) + "m",
        "--pids-limit",
        str(spec.pids_limit),
        "--shm-size",
        "1g",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=2147483648",
        "--publish",
        f"127.0.0.1:{port}:8000",
        "--mount",
        f"type=bind,src={spec.model_directory},dst=/models,readonly",
        "--mount",
        f"type=bind,src={directory / 'profile.json'},dst=/run/finserve/profile.json,readonly",
        spec.image.image_local_id,
        "--profile-sha256",
        spec.profile.digest(),
        "--expected-model",
        spec.profile.served_model,
        "--expected-base-url",
        spec.profile.base_url,
    ]


def inspect_runtime(
    document: Any, spec: RuntimeLaunchSpec, attempt_id: str, directory: Path
) -> dict[str, Any]:
    """Require exact image, identity, entrypoint, mounts, resource caps and loopback publication."""
    if not isinstance(document, list):
        raise ValueError("one exact managed container required")
    items = cast(list[dict[str, Any]], document)
    if len(items) != 1:
        raise ValueError("one exact managed container required")
    item = items[0]
    config, host = item["Config"], item["HostConfig"]
    image_document = bounded_document(directory / "image-identity.json")
    verify_image_inspection(image_document, spec.image)
    image_environment = image_document[0]["Config"].get("Env", [])
    labels = config["Labels"]
    expected_mounts = {
        "/models": str(spec.model_directory),
        "/run/finserve/profile.json": str(directory / "profile.json"),
    }
    actual_mounts = {
        mount["Destination"]: mount["Source"]
        for mount in item["Mounts"]
        if mount["Type"] == "bind" and mount["RW"] is False
    }
    if (
        re.fullmatch(r"[0-9a-f]{64}", item["Id"]) is None
        or item["Name"] != "/" + runtime_name(attempt_id)
        or item["Image"] != spec.image.image_local_id
        or config["Image"] != spec.image.image_local_id
        or config["Entrypoint"] != ["python3", "-m", "finserve.registry.engine_entrypoint"]
        or config.get("Cmd")
        != [
            "--profile-sha256",
            spec.profile.digest(),
            "--expected-model",
            spec.profile.served_model,
            "--expected-base-url",
            spec.profile.base_url,
        ]
        or config["User"] != "10001:10001"
        or sorted(config.get("Env", [])) != sorted(image_environment)
        or labels.get("finserve.runtime-attempt") != attempt_id
        or labels.get("finserve.runtime-specification") != spec.digest()
        or len(item["Mounts"]) != 2
        or actual_mounts != expected_mounts
        or host["ReadonlyRootfs"] is not True
        or host["Privileged"] is not False
        or host["Memory"] != spec.memory_mib * 1024**2
        or host["MemorySwap"] != spec.memory_mib * 1024**2
        or host["PidsLimit"] != spec.pids_limit
        or any(
            host.get(field) not in (None, [])
            for field in (
                "CapAdd",
                "CapDrop",
                "Devices",
                "Binds",
                "SecurityOpt",
                "GroupAdd",
                "VolumesFrom",
                "ExtraHosts",
                "DeviceCgroupRules",
            )
        )
        or host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0}
        or host.get("PidMode") != ""
        or host.get("IpcMode") != "private"
        or host.get("UTSMode") != ""
        or host.get("UsernsMode") != ""
        or host.get("PublishAllPorts") is not False
        or host.get("AutoRemove") is not False
        or host["NetworkMode"] != "bridge"
        or host["Tmpfs"] != {"/tmp": "rw,exec,nosuid,size=2147483648"}
        or host["ShmSize"] != 1024**3
        or host["DeviceRequests"]
        != [
            {
                "Driver": "",
                "Count": -1,
                "DeviceIDs": None,
                "Capabilities": [["gpu"]],
                "Options": {},
            }
        ]
        or host["PortBindings"]
        != {
            "8000/tcp": [
                {"HostIp": "127.0.0.1", "HostPort": str(httpx.URL(spec.profile.base_url).port)}
            ]
        }
    ):
        raise ValueError("managed container differs from frozen launch")
    return item


class DockerRuntime:
    """A deterministic name permits reconciliation without starting an untracked replacement."""

    def __init__(
        self, command: CommandRunner = run_command, monotonic: Callable[[], float] = time.monotonic
    ) -> None:
        """Inject a command executor for CPU tests; production retains actual Docker CLI output."""
        self.command = command
        self.monotonic = monotonic

    def remaining(self, deadline: float | None, ceiling: float) -> float:
        """Reject new external operations after the shared cold-start budget expires."""
        available = ceiling if deadline is None else min(ceiling, deadline - self.monotonic())
        if available <= 0:
            raise TimeoutError("managed runtime operation deadline expired")
        return available

    async def _command(
        self,
        arguments: list[str],
        directory: Path,
        budget_seconds: float,
        *,
        deadline: float | None = None,
    ) -> Path:
        """Keep unique command logs and drain CLI ownership if cancellation interrupts a task."""
        output = directory / (uuid4().hex + ".log")
        await owned_disk(
            lambda: self.command(
                arguments, directory, output, self.remaining(deadline, budget_seconds)
            )
        )
        return output

    async def _find(self, name: str, directory: Path, deadline: float | None = None) -> str | None:
        """An empty successful daemon listing is distinct from an unavailable Docker daemon."""
        return await self._lookup("name=^/" + name + "$", directory, deadline)

    async def _lookup(
        self, selector: str, directory: Path, deadline: float | None = None
    ) -> str | None:
        """Only internal exact-name or full-ID filters can select a managed container."""
        output = await self._command(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                selector,
                "--format",
                "{{.ID}}",
            ],
            directory,
            30,
            deadline=deadline,
        )

        def read_identifier() -> str:
            """A lookup cannot allocate an unbounded daemon response before validating one ID."""
            with output.open("rb") as stream:
                content = stream.read(4097)
            if len(content) > 4096:
                raise ValueError("managed runtime lookup exceeds byte budget")
            return content.decode().strip()

        content = await owned_disk(read_identifier)
        if not content:
            return None
        if re.fullmatch(r"[0-9a-f]{64}", content) is None:
            raise ValueError("ambiguous managed container lookup")
        return content

    async def _inspect(
        self,
        container_id: str,
        spec: RuntimeLaunchSpec,
        attempt_id: str,
        directory: Path,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Inspect by exact ID before any health claim, start, stop or removal."""
        output = await self._command(
            ["docker", "container", "inspect", container_id], directory, 30, deadline=deadline
        )
        actual = await owned_disk(
            lambda: inspect_runtime(bounded_document(output), spec, attempt_id, directory)
        )
        if actual["Id"] != container_id:
            raise ValueError("managed runtime ID changed")
        return actual

    async def launch(
        self,
        specification: RuntimeLaunchSpec,
        attempt_id: str,
        workspace: Path,
        client: httpx.AsyncClient,
    ) -> RuntimeReceipt:
        """Reverify inputs, reconcile creation, and observe actual inference before readiness."""
        spec = RuntimeLaunchSpec.model_validate_json(specification.model_dump_json())
        name = runtime_name(attempt_id)
        start = self.monotonic()
        deadline = start + spec.readiness_timeout_seconds
        directory = await owned_disk(lambda: owned_directory(workspace, create=True) / attempt_id)

        def prepare() -> None:
            """Verify model bytes and freeze profile mounts in the exclusively owned volume."""
            owned_directory(directory, create=True)
            owned_directory(spec.model_directory)
            freeze_file(directory / "specification.json", spec.canonical())
            freeze_file(directory / "profile.json", spec.profile.canonical())
            manifest = verify_snapshot(spec.model_directory, spec.model)
            if manifest.digest() != spec.profile.model_manifest_sha256:
                raise ValueError("actual model snapshot differs from launch")

        await owned_disk(prepare)
        image_log = await self._command(
            ["docker", "image", "inspect", spec.image.image_local_id],
            directory,
            30,
            deadline=deadline,
        )

        def freeze_image() -> None:
            """Bind the complete inherited environment to the inspected immutable image."""
            document = bounded_document(image_log)
            verify_image_inspection(document, spec.image)
            item = document[0]
            identity = [
                {
                    "Id": item["Id"],
                    "Os": item["Os"],
                    "Architecture": item["Architecture"],
                    "Descriptor": item.get("Descriptor"),
                    "Config": {
                        "Labels": item["Config"]["Labels"],
                        "Env": item["Config"].get("Env", []),
                    },
                }
            ]
            freeze_file(directory / "image-identity.json", json.dumps(identity, sort_keys=True))

        await owned_disk(freeze_image)
        if self.monotonic() - start >= spec.readiness_timeout_seconds:
            raise TimeoutError("model verification exceeded cold-start budget")
        container_id = await self._find(name, directory, deadline)
        if container_id is None:
            await self._command(
                create_arguments(spec, attempt_id, directory), directory, 60, deadline=deadline
            )
            container_id = await self._find(name, directory, deadline)
            if container_id is None:
                raise RuntimeError("created runtime not observed; reconciliation required")
        actual = await self._inspect(container_id, spec, attempt_id, directory, deadline)
        if actual["State"]["Status"] == "created":
            await self._command(
                ["docker", "container", "start", container_id], directory, 60, deadline=deadline
            )
        elif actual["State"]["Running"] is not True:
            raise RuntimeError(
                "owned runtime exited; inspect retained evidence before a new attempt"
            )
        while self.monotonic() - start < spec.readiness_timeout_seconds:
            actual = await self._inspect(container_id, spec, attempt_id, directory, deadline)
            if actual["State"]["Running"] is not True:
                raise RuntimeError("owned runtime exited during readiness")
            if await self.probe_endpoint(spec, client, directory, deadline=deadline):
                after = await self._inspect(container_id, spec, attempt_id, directory, deadline)
                if (
                    after["State"]["Running"] is not True
                    or after["State"]["StartedAt"] != actual["State"]["StartedAt"]
                ):
                    raise RuntimeError("runtime changed during readiness observation")
                elapsed = self.monotonic() - start
                if elapsed >= spec.readiness_timeout_seconds:
                    raise TimeoutError("runtime smoke exceeded cold-start budget")
                return RuntimeReceipt(
                    specification_sha256=spec.digest(),
                    attempt_id=attempt_id,
                    container_id=container_id,
                    container_started_at=after["State"]["StartedAt"],
                    observed_at=time.time(),
                    elapsed_seconds=elapsed,
                    output_directory=directory,
                )
            await asyncio.sleep(self.remaining(deadline, 1))
        raise TimeoutError("managed runtime readiness budget expired")

    async def probe_endpoint(
        self,
        spec: RuntimeLaunchSpec,
        client: httpx.AsyncClient,
        directory: Path,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Observe HTTP content; launch separately binds this probe to Docker identity and time."""
        try:
            async with asyncio.timeout(self.remaining(deadline, 5)):
                async with client.stream(
                    "GET", spec.profile.base_url + "/models", follow_redirects=False
                ) as response:
                    own_response(response)
                    models = json.loads(await bounded_response(response, 65536))
            if not any(item.get("id") == spec.profile.served_model for item in models["data"]):
                return False
            row = await completion_probe(
                client,
                spec.profile.base_url,
                spec.profile.served_model,
                self.remaining(deadline, 10),
            )
            await owned_disk(
                lambda: (directory / (uuid4().hex + "-smoke.json")).write_text(
                    row.model_dump_json()
                )
            )
            if row.error == "HTTPClosureError":
                raise HTTPClosureError("runtime probe cleanup remains unresolved")
            if row.error == "CancelledError":
                raise asyncio.CancelledError
            return row.success and bool(row.output) and row.generated_tokens == 1
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
            return False

    async def stop(self, specification: RuntimeLaunchSpec, receipt: RuntimeReceipt) -> None:
        """Stop the exact owned start and verify stopped state before removing its container."""
        spec = RuntimeLaunchSpec.model_validate_json(specification.model_dump_json())
        if receipt.specification_sha256 != spec.digest():
            raise ValueError("runtime stop receipt differs from launch")
        directory = receipt.output_directory
        if directory.name != receipt.attempt_id:
            raise ValueError("runtime receipt directory differs from owned attempt")
        await owned_disk(lambda: owned_directory(directory))
        await owned_disk(lambda: freeze_file(directory / "specification.json", spec.canonical()))
        await owned_disk(lambda: freeze_file(directory / "profile.json", spec.profile.canonical()))
        found = await self._lookup("id=" + receipt.container_id, directory)
        if found is None:
            return
        if found != receipt.container_id:
            raise ValueError("cleanup lookup differs from exact owned ID")
        actual = await self._inspect(receipt.container_id, spec, receipt.attempt_id, directory)
        if actual["State"]["StartedAt"] != receipt.container_started_at:
            raise ValueError("runtime was restarted after its recorded health observation")
        if actual["State"]["Running"] is True:
            await self._command(
                [
                    "docker",
                    "container",
                    "stop",
                    "--time",
                    str(spec.shutdown_timeout_seconds),
                    receipt.container_id,
                ],
                directory,
                spec.shutdown_timeout_seconds + 15,
            )
        await self._command(["docker", "container", "logs", receipt.container_id], directory, 30)
        after = await self._inspect(receipt.container_id, spec, receipt.attempt_id, directory)
        if after["State"]["Running"] is not False:
            raise RuntimeError("owned runtime stop is not verified")
        await self._command(["docker", "container", "rm", receipt.container_id], directory, 30)
