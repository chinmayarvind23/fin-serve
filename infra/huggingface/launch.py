"""Supervise the two CPU-only Space children without shell commands or secret arguments."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import httpx


def checked_environment() -> dict[str, str]:
    """Require separate bounded secrets and fix the internal data/HTTP namespaces."""
    environment = dict(os.environ)
    for name in tuple(environment):
        if name == "WEB_CONCURRENCY" or name.startswith("UVICORN_"):
            environment.pop(name)
    keys = [environment.get(name, "") for name in ("FINSERVE_API_KEY", "FINSERVE_WEB_KEY")]
    if (
        any(
            not 16 <= len(key) <= 4096 or any(ord(char) < 33 or ord(char) > 126 for char in key)
            for key in keys
        )
        or keys[0] == keys[1]
    ):
        raise ValueError("two distinct printable Space secrets of 16..4096 characters are required")
    Path("/data/artifacts").mkdir(parents=True, exist_ok=True)
    environment.update(
        FINSERVE_REGISTRY_URL="sqlite:////data/evidence.db",
        FINSERVE_ARTIFACT_ROOT="/data/artifacts",
    )
    return environment


def stop_children(children: list[subprocess.Popen[bytes]], grace_seconds: float = 15) -> None:
    """Drain owned direct children; these commands never launch worker subprocesses."""
    failures: list[Exception] = []
    for child in children:
        try:
            if child.poll() is None:
                child.terminate()
        except ProcessLookupError:
            pass
        except OSError as error:
            failures.append(error)
    deadline = time.monotonic() + grace_seconds
    for child in children:
        try:
            child.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                child.kill()
            except ProcessLookupError:
                pass
            except OSError as error:
                failures.append(error)
        except OSError as error:
            failures.append(error)
    for child in children:
        try:
            child.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(error)
    if failures:
        raise RuntimeError("Space cleanup could not confirm every child exit") from None


async def ready(
    key: str, budget_seconds: float, *, transport: httpx.AsyncBaseTransport | None = None
) -> bool:
    """Bound the whole authenticated probe; ambient proxies and redirects never receive its key."""
    try:
        async with asyncio.timeout(budget_seconds):
            async with httpx.AsyncClient(
                trust_env=False, follow_redirects=False, timeout=budget_seconds, transport=transport
            ) as client:
                async with client.stream(
                    "POST",
                    "http://127.0.0.1:8050/graphql",
                    json={"query": "{ runs(first: 1) { id } }"},
                    headers={"Authorization": "Bearer " + key, "Accept-Encoding": "identity"},
                ) as response:
                    if (
                        response.status_code != 200
                        or response.headers.get("content-encoding", "identity") != "identity"
                    ):
                        return False
                    body = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(body) + len(chunk) > 4096:
                            return False
                        body.extend(chunk)
                    payload = json.loads(body)
                    if not isinstance(payload, dict):
                        return False
                    data = cast(dict[str, object], payload).get("data")
                    return isinstance(data, dict) and isinstance(
                        cast(dict[str, object], data).get("runs"), list
                    )
    except (TimeoutError, httpx.HTTPError, ValueError):
        return False


def main() -> int:
    """Start the edge only after registry readiness; stop both children on signal or exit."""
    stopping = False

    def request_stop(_signal: int, _frame: object) -> None:
        """Repeated signals set one stop flag so they cannot interrupt owned cleanup."""
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    children: list[subprocess.Popen[bytes]] = []
    try:
        environment = checked_environment()
        python_environment = dict(environment)
        python_environment.pop("FINSERVE_WEB_KEY", None)
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "finserve.registry.explorer:from_env",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8050",
                    "--no-access-log",
                    "--workers",
                    "1",
                    "--timeout-graceful-shutdown",
                    "10",
                ],
                env=python_environment,
            )
        )
        deadline = time.monotonic() + 60
        while not stopping:
            remaining = deadline - time.monotonic()
            if children[0].poll() is not None or remaining <= 0:
                raise RuntimeError("registry startup failed")
            if asyncio.run(ready(environment["FINSERVE_API_KEY"], min(1, remaining))):
                break
            time.sleep(0.1)
        if stopping:
            return 0
        children.append(
            subprocess.Popen(["bun", "/app/web/space_server.js"], env=environment, cwd="/app/web")
        )
        while not stopping:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("Space child exited unexpectedly")
            time.sleep(0.1)
        return 0
    except Exception as error:
        print(f"Space stopped: {type(error).__name__}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop_children(children)


if __name__ == "__main__":
    raise SystemExit(main())
