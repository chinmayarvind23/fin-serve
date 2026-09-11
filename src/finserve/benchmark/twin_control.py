"""Trusted local-file control of two owned engine processes for bounded scaling experiments."""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from finserve.benchmark.gpu import collect
from finserve.benchmark.owned_process import OwnedProcess
from finserve.benchmark.routing_workload import MODEL, REVISION
from finserve.benchmark.runner import prepare_output, write_json


class Command(BaseModel):
    """Accept fixed operations and backends; commands cannot supply executable text."""

    model_config = ConfigDict(extra="forbid")
    operation: Literal["start", "stop", "fail", "status", "shutdown"]
    backend: Literal["a", "b"] = "a"


class TwinController:
    """Hold pidfds for the experiment lifetime and journal every accepted control operation."""

    def __init__(self, executable: Path, profile: Path, output: Path) -> None:
        """Require the pinned model and fixed trusted profile before opening any GPU process."""
        self.profile: dict[str, Any] = json.loads(profile.read_bytes())
        if self.profile["model"] != MODEL or self.profile["model_revision"] != REVISION:
            raise ValueError("unexpected model profile")
        self.executable = executable
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("absolute installed vLLM executable required")
        self.output = prepare_output(output)
        (self.output / "commands").mkdir()
        (self.output / "profile.json").write_bytes(profile.read_bytes())
        self.processes: dict[str, OwnedProcess] = {}
        self.generations = {"a": 0, "b": 0}
        self.done = False
        self.manifest: dict[str, Any] = {
            "status": "running",
            "pid": os.getpid(),
            "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
            "started_epoch_s": time.time(),
            "scope": "owned local processes on one shared GPU",
        }
        write_json(self.output / "manifest.json", self.manifest)

    def endpoint(self, backend: str) -> str:
        """Ports come from the trusted launch profile, never a request or command URL."""
        first = str(self.profile["base_url"])
        return (
            first
            if backend == "a"
            else str(httpx.URL(first).copy_with(port=int(self.profile["second_engine_port"])))
        )

    async def ready(self, process: OwnedProcess, endpoint: str) -> dict[str, Any]:
        """Retain descendant ownership while checking exact-model readiness with bounded bytes."""
        deadline = time.monotonic() + 300
        async with httpx.AsyncClient(timeout=1, trust_env=False, follow_redirects=False) as client:
            while time.monotonic() < deadline:
                evidence = process.evidence()
                if evidence["returncode"] is not None:
                    raise RuntimeError("engine process exited before readiness")
                try:
                    async with asyncio.timeout_at(min(deadline, time.monotonic() + 1)):
                        async with client.stream("GET", endpoint + "/models") as response:
                            response.raise_for_status()
                            body = bytearray()
                            async for chunk in response.aiter_bytes():
                                if len(body) + len(chunk) > 131072:
                                    raise ValueError("model discovery exceeds bound")
                                body.extend(chunk)
                    models = json.loads(body)
                    if not any(row.get("id") == MODEL for row in models["data"]):
                        raise ValueError("model identity mismatch")
                    return {
                        "models": models,
                        "process": process.evidence(),
                        "ready_epoch_s": time.time(),
                        "startup_seconds": time.monotonic() - process.started_monotonic,
                    }
                except (httpx.HTTPError, TimeoutError):
                    await asyncio.sleep(0.5)
        raise TimeoutError("engine model readiness deadline expired")

    async def start(self, backend: str) -> dict[str, Any]:
        """Start a new process generation only when that backend has no owned active generation."""
        if backend in self.processes:
            raise ValueError("backend already owned; stop it explicitly before restart")
        endpoint = self.endpoint(backend)
        # An occupied port is not ours, even if it advertises the expected model.
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", httpx.URL(endpoint).port or 80))
        argv = list(self.profile["launch_argv"])
        argv[0] = str(self.executable)
        argv[argv.index("--port") + 1] = str(httpx.URL(endpoint).port)
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("VLLM_")
        }
        environment.update(self.profile["environment"])
        self.generations[backend] += 1
        log = self.output / f"{backend}-{self.generations[backend]}-startup.log"
        process = OwnedProcess(argv, environment, log)
        self.processes[backend] = process
        try:
            ready = await self.ready(process, endpoint)
            ready.update(backend=backend, endpoint=endpoint, generation=self.generations[backend])
            write_json(self.output / f"{backend}-{self.generations[backend]}-ready.json", ready)
            return ready
        except BaseException:
            await self.stop(backend)
            raise

    async def stop(self, backend: str, force: bool = False) -> dict[str, object]:
        """Native drain is separate from router quarantine; the caller must stop admission first."""
        process = self.processes.get(backend)
        if process is None:
            return {"backend": backend, "already_stopped": True}
        result = await process.stop(force=force)
        del self.processes[backend]
        return result

    async def execute(self, command: Command) -> dict[str, Any]:
        """Journal scale/failure operations; sample the physical device only on explicit status."""
        if command.operation == "start":
            return await self.start(command.backend)
        if command.operation in {"stop", "fail"}:
            return await self.stop(command.backend, force=command.operation == "fail")
        if command.operation == "shutdown":
            self.done = True
        return {
            "processes": {name: process.evidence() for name, process in self.processes.items()},
            "physical_gpu": (await asyncio.to_thread(collect)).model_dump(),
        }

    async def run(self) -> None:
        """Consume unique local command files and retain failures without implicit restarts."""
        seen: set[str] = set()
        outcome = "stopped"
        try:
            while not self.done:
                for path in sorted((self.output / "commands").glob("*.json")):
                    if path.name in seen:
                        continue
                    seen.add(path.name)
                    event: dict[str, Any] = {"command": path.name, "started_epoch_s": time.time()}
                    try:
                        with path.open("rb") as source:
                            raw = source.read(4097)
                        if len(raw) > 4096:
                            raise ValueError("control command exceeds bound")
                        command = Command.model_validate_json(raw)
                        event.update(
                            request=command.model_dump(),
                            result=await self.execute(command),
                            status="complete",
                        )
                    except Exception as exc:
                        event.update(status="failed", error=type(exc).__name__)
                    finally:
                        event["finished_epoch_s"] = time.time()
                        write_json(self.output / ("result-" + path.name), event)
                for process in self.processes.values():
                    process.observe()
                await asyncio.sleep(0.5)
        except BaseException as exc:
            outcome = "failed" if isinstance(exc, Exception) else "interrupted"
            self.manifest["error"] = type(exc).__name__
            raise
        finally:
            failures: dict[str, str] = {}
            for backend in tuple(self.processes):
                try:
                    await self.stop(backend)
                except Exception as exc:
                    failures[backend] = type(exc).__name__
            write_json(
                self.output / "shutdown.json",
                {
                    "epoch_s": time.time(),
                    "failures": failures,
                    "remaining": {
                        name: process.last_cleanup for name, process in self.processes.items()
                    },
                },
            )
            self.manifest.update(
                status="failed" if failures else outcome, finished_epoch_s=time.time()
            )
            write_json(self.output / "manifest.json", self.manifest)


def main() -> None:
    """Expose a local operator tool with no HTTP control port or remote instructions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(TwinController(args.executable, args.profile, args.output).run())


if __name__ == "__main__":
    main()
