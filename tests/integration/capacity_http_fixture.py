"""Actual loopback SSE fixtures started only by the capacity executor's fake Docker boundary."""

import asyncio
import hashlib
import json
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from test_managed_runtime import Daemon

from finserve.contracts.managed_runtime import RuntimeLaunchSpec


class CapacityHTTPBackend:
    """Expose genuine request/stream lifetimes without introducing any model-quality oracle."""

    def __init__(self, name: str, port: int) -> None:
        """Keep readiness smoke immediate while explicitly named test streams can hold capacity."""
        self.name, self.port = name, port
        self.received: list[str] = []
        self.holds: dict[str, asyncio.Event] = {}
        self.entered: dict[str, asyncio.Event] = {}
        self.closed: dict[str, asyncio.Event] = {}
        self.app = FastAPI()
        self.app.get("/v1/models")(self.models)
        self.app.post("/v1/completions")(self.complete)
        self.server: uvicorn.Server | None = None
        self.task: asyncio.Task[None] | None = None
        self.listener: socket.socket | None = None

    async def models(self) -> dict[str, object]:
        """Let the unchanged runtime readiness probe discover the exact fixture model alias."""
        return {"data": [{"id": "fixture"}]}

    def hold(self, prompt: str) -> None:
        """Mark only requested test traffic as held; readiness and smoke still execute normally."""
        self.holds[prompt] = asyncio.Event()
        self.entered[prompt] = asyncio.Event()
        self.closed[prompt] = asyncio.Event()

    async def complete(self, request: Request) -> StreamingResponse:
        """Journal real dispatch and expose physical backend identity through valid SSE content."""
        payload = await request.json()
        prompt = str(payload["prompt"])
        self.received.append(prompt)

        async def events() -> AsyncIterator[str]:
            """Hold after visible output so downscale must respect an existing physical pin."""
            try:
                yield (
                    "data: "
                    + json.dumps(
                        {"choices": [{"index": 0, "text": self.name, "finish_reason": None}]}
                    )
                    + "\n\n"
                )
                if prompt in self.holds:
                    self.entered[prompt].set()
                    await self.holds[prompt].wait()
                yield (
                    'data: {"choices":[{"index":0,"text":"","finish_reason":"stop"}],'
                    '"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
                )
            finally:
                if prompt in self.closed:
                    self.closed[prompt].set()

        return StreamingResponse(events(), media_type="text/event-stream")

    async def start(self) -> None:
        """Bind the frozen endpoint at simulated Docker start, then await real HTTP readiness."""
        if self.task is not None:
            raise AssertionError("fixture endpoint cannot be started twice")
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", self.port))
        self.listener.setblocking(False)
        self.server = uvicorn.Server(uvicorn.Config(self.app, log_level="error", lifespan="on"))
        self.task = asyncio.create_task(self.server.serve(sockets=[self.listener]))
        async with asyncio.timeout(5):
            while not self.server.started:
                if self.task.done():
                    await self.task
                    raise AssertionError("fixture HTTP startup failed")
                await asyncio.sleep(0.01)

    async def stop(self) -> None:
        """Drain only this fixture; a premature capacity stop fails instead of releasing holds."""
        if self.server is not None and self.task is not None:
            self.server.should_exit = True
            await asyncio.wait_for(asyncio.shield(self.task), 5)
        if self.listener is not None:
            self.listener.close()


class CapacityDaemon(Daemon):
    """Reuse immutable fake Docker inspection while executing actual endpoint start and stop."""

    def __init__(self, spec: RuntimeLaunchSpec, backend: CapacityHTTPBackend) -> None:
        """Capture the live test loop because DockerRuntime invokes command runners in threads."""
        super().__init__(spec)
        self.backend = backend
        self.loop = asyncio.get_running_loop()
        self.borrowed: CapacityDaemon | None = None

    def __call__(self, arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Only the external daemon boundary is replaced; launch/probe/receipt code stays real."""
        if (
            self.borrowed is not None
            and self.borrowed.container is not None
            and arguments[-1] == self.borrowed.container["Id"]
        ):
            self.borrowed(arguments, directory, output, timeout)
            return
        super().__call__(arguments, directory, output, timeout)
        if arguments[:3] == ["docker", "container", "start"]:
            asyncio.run_coroutine_threadsafe(self.backend.start(), self.loop).result(timeout=5)
        elif arguments[:3] == ["docker", "container", "stop"]:
            asyncio.run_coroutine_threadsafe(self.backend.stop(), self.loop).result(timeout=6)

    def inspection(self, attempt: str, directory: Path) -> dict[str, Any]:
        """Keep all original ownership fields and adapt only the explicitly frozen test port."""
        value = super().inspection(attempt, directory)
        value["Id"] = hashlib.sha256(self.backend.name.encode()).hexdigest()
        value["HostConfig"]["PortBindings"] = {
            "8000/tcp": [
                {"HostIp": "127.0.0.1", "HostPort": str(httpx.URL(self.spec.profile.base_url).port)}
            ]
        }
        return value
