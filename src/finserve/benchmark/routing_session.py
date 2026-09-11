"""Run local Ray controls, four frozen cohorts, and explicit owned-engine scaling/failure drills."""

import argparse
import asyncio
import importlib
import json
import os
import secrets
import time
from collections.abc import Callable
from contextlib import aclosing
from pathlib import Path
from typing import Any
from uuid import uuid4

from finserve.benchmark.routing_cohort import BackendIdentity, RoutingTopology, run_cohort
from finserve.benchmark.routing_workload import MODEL, frozen_workload
from finserve.benchmark.runner import prepare_output, write_json
from finserve.contracts.inference import InferenceRequest
from finserve.engines.ray_backends import build_application
from finserve.engines.ray_http import RayHTTPEngine


async def command(controller: Path, operation: str, backend: str = "a") -> dict[str, Any]:
    """Atomically submit one local operator command and retain its exact acknowledged result."""
    name = f"session-{uuid4().hex}.json"
    temporary = controller / "commands" / (name + ".tmp")
    write_json(temporary, {"operation": operation, "backend": backend})
    temporary.replace(controller / "commands" / name)
    result = controller / ("result-" + name)
    deadline = time.monotonic() + 340
    while time.monotonic() < deadline:
        if result.exists():
            try:
                value: dict[str, Any] = json.loads(result.read_bytes())
            except json.JSONDecodeError:
                pass  # Partial publication is not a controller acknowledgement.
            else:
                if value["status"] != "complete":
                    raise RuntimeError(f"owned controller command failed: {name}")
                return value
        await asyncio.sleep(0.2)
    raise TimeoutError("owned controller command acknowledgment missing")


async def control_request(url: str, path: Path, label: str) -> dict[str, Any]:
    """Record a separate functional request without placing controls in performance denominators."""
    request = InferenceRequest(
        model=MODEL, prompt=frozen_workload().cases[0].prompt, max_tokens=16, request_id=uuid4().hex
    )
    engine = RayHTTPEngine(url, api_key=os.environ["FINSERVE_RAY_API_KEY"])
    result: dict[str, Any] = {
        "label": label,
        "request_id": request.request_id,
        "started_epoch_s": time.time(),
        "output": "",
    }
    started = time.perf_counter()
    try:
        async with aclosing(engine.stream(request)) as output:
            async for token in output:
                if token.text and "ttft_seconds" not in result:
                    result["ttft_seconds"] = time.perf_counter() - started
                result["output"] += token.text
                if token.finish_reason:
                    result["generated_tokens"] = token.generated_tokens
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=type(exc).__name__)
        raise
    finally:
        result["e2e_seconds"] = time.perf_counter() - started
        try:
            await engine.close()
        except BaseException as exc:
            result.update(status="failed", cleanup_error=type(exc).__name__)
            raise
        finally:
            write_json(path, result)
    return result


def topology(controller: Path, status: dict[str, Any]) -> RoutingTopology:
    """Bind ready-file PID generations to configured endpoints and one observed device UUID."""
    ready = [json.loads((controller / f"{name}-1-ready.json").read_bytes()) for name in ("a", "b")]
    devices = status["result"]["physical_gpu"]["devices"]
    if len(devices) != 1:
        raise ValueError("one observed physical GPU required")
    current = status["result"]["processes"]
    if set(current) != {"a", "b"}:
        raise ValueError("two currently owned processes required")
    for row in ready:
        live = current[row["backend"]]
        if (
            live["returncode"] is not None
            or live["unowned_session_members"]
            or any(live[field] != row["process"][field] for field in ("pid", "start_ticks"))
        ):
            raise ValueError("ready artifact is stale or ownership is unresolved")
    identities = [
        BackendIdentity(
            replica_id=row["backend"],
            endpoint=row["endpoint"],
            pid=row["process"]["pid"],
            start_ticks=row["process"]["start_ticks"],
        )
        for row in ready
    ]
    return RoutingTopology(
        physical_gpu_uuid=devices[0]["uuid"],
        backends=(identities[0], identities[1]),
        profile=json.loads((controller / "profile.json").read_bytes()),
    )


async def deploy(
    serve: Any, declaration: RoutingTopology, policy: str, single: bool = False
) -> Any:
    """Keep one routing authority and one CPU proxy per actual external engine process."""
    backends = declaration.backends[:1] if single else declaration.backends
    app = build_application(
        {
            "model": MODEL,
            "backends": {row.replica_id: row.endpoint for row in backends},
            "capacity_per_worker": 4,
            "mode": policy,
            "observe_engine_metrics": True,
            "shared_gpu_uuid": declaration.physical_gpu_uuid,
        }
    )
    return await native_owned(serve.run, app, name="twin-routing", route_prefix="/route")


async def drain(handle: Any) -> dict[str, Any]:
    """Require both proxy leases and observed native queues empty before planned scale-down."""
    async with asyncio.timeout(30):
        while True:
            state = await handle.status.remote()
            if state["active_reservations"] == 0 and all(
                row["ongoing_requests"]
                == row.get("engine_running_requests", -1)
                == row.get("engine_waiting_requests", -1)
                == 0
                for row in state["workers"]
            ):
                return state
            await asyncio.sleep(0.2)


async def failure_drill(handle: Any, controller: Path, url: str, output: Path) -> None:
    """Kill owned backend b after observed content and retain the partial-stream error."""
    await handle.set_enabled.remote("a", False)
    engine = RayHTTPEngine(url, api_key=os.environ["FINSERVE_RAY_API_KEY"])
    request = InferenceRequest(
        model=MODEL,
        prompt="Continue counting with one number per line until 500.\n1.",
        max_tokens=512,
        request_id=uuid4().hex,
    )
    result: dict[str, Any] = {"request_id": request.request_id, "output": "", "status": "running"}
    try:
        async with aclosing(engine.stream(request)) as stream:
            first = await anext(stream)
            result["output"] = first.text
            if not first.text:
                raise ValueError("failure injection requires observed content")
            result["kill"] = await command(controller, "fail", "b")
            try:
                async for token in stream:
                    result["output"] += token.text
                result["status"] = "injection_missed_active_stream"
            except Exception as exc:
                result.update(status="partial_stream_failed", error=type(exc).__name__)
    except BaseException as exc:
        result.update(status="failed", error=type(exc).__name__)
        raise
    finally:
        try:
            try:
                await engine.close()
            finally:
                async with asyncio.timeout(5):
                    await handle.set_enabled.remote("a", True)
                    result["post_failure_status"] = await handle.status.remote()
        except BaseException as exc:
            result["cleanup_error"] = type(exc).__name__
            raise
        finally:
            write_json(output / "failure-after-partial.json", result)
    if result["status"] != "partial_stream_failed":
        raise ValueError("failure drill did not interrupt an active stream")


async def drills(handle: Any, controller: Path, url: str, output: Path) -> None:
    """Separate planned drain and unplanned failure from the four immutable measured populations."""
    await handle.set_enabled.remote("b", False)
    write_json(output / "pre-scale-down-drain.json", await drain(handle))
    write_json(output / "scale-down-b.json", await command(controller, "stop", "b"))
    await control_request(url, output / "survivor-after-scale-down.json", "survivor-a")
    write_json(output / "scale-up-b-generation2.json", await command(controller, "start", "b"))
    await handle.set_enabled.remote("b", True)
    await asyncio.sleep(1.1)
    await failure_drill(handle, controller, url, output)
    await control_request(url, output / "survivor-after-failure.json", "survivor-a")
    write_json(output / "scale-up-b-generation3.json", await command(controller, "start", "b"))
    await asyncio.sleep(1.1)
    await handle.set_enabled.remote("a", False)
    try:
        await control_request(url, output / "replacement-b-control.json", "replacement-b")
    finally:
        await handle.set_enabled.remote("a", True)


async def run(controller: Path, output: Path) -> None:
    """Own this Ray cluster; native engine shutdown belongs to the pidfd controller."""
    directory = prepare_output(output)
    os.environ["FINSERVE_RAY_API_KEY"] = secrets.token_urlsafe(32)
    for name in ("FINSERVE_TRACE_PATH", "FINSERVE_OTLP_ENDPOINT", "FINSERVE_OTLP_AUTHORIZATION"):
        os.environ.pop(name, None)
    os.environ["FINSERVE_TRACE_SAMPLE_RATIO"] = "0"
    declaration = topology(controller, await command(controller, "status"))
    write_json(directory / "topology.json", declaration.model_dump())
    ray, serve = importlib.import_module("ray"), importlib.import_module("ray.serve")
    manifest: dict[str, Any] = {
        "status": "running",
        "started_epoch_s": time.time(),
        "ray_version": ray.__version__,
        "physical_devices": 1,
        "tracing": {"enabled": False, "sample_ratio": 0},
    }
    write_json(directory / "manifest.json", manifest)
    try:
        await native_owned(
            ray.init,
            address="local",
            num_cpus=4,
            num_gpus=0,
            include_dashboard=False,
            log_to_driver=False,
            object_store_memory=100 * 1024 * 1024,
        )
        await native_owned(serve.start, http_options={"host": "127.0.0.1", "port": 8034})
        url = "http://127.0.0.1:8034/route"
        await deploy(serve, declaration, "least_load", single=True)
        await control_request(url, directory / "single-ray-control.json", "single-routed-backend")
        handle = await deploy(serve, declaration, "least_load")
        await asyncio.gather(
            *(
                control_request(url, directory / f"pair-control-{index}.json", "pair")
                for index in range(2)
            )
        )
        write_json(directory / "pair-control-status.json", await handle.status.remote())
        for index, mode in enumerate(("least_load", "adaptive", "adaptive", "least_load"), 1):
            handle = await deploy(serve, declaration, mode)
            await run_cohort(
                url, frozen_workload(), mode, directory / f"cohort-{index}-{mode}", declaration
            )
        await drills(handle, controller, url, directory)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(
            status="failed" if isinstance(exc, Exception) else "interrupted",
            error=type(exc).__name__,
        )
        raise
    finally:
        await finish_session(serve, ray, directory, manifest)


async def native_owned(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Finish native init/deploy before cancellation propagates and starts conflicting shutdown."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def finish_session(serve: Any, ray: Any, directory: Path, manifest: dict[str, Any]) -> None:
    """One shielded cleanup attempts both shutdowns and always finalizes lifecycle evidence."""

    async def cleanup() -> None:
        """A Serve shutdown failure cannot skip the independently owned Ray cluster shutdown."""
        failures: dict[str, str] = {}
        for name, function in (("serve", serve.shutdown), ("ray", ray.shutdown)):
            try:
                await asyncio.to_thread(function)
            except BaseException as exc:
                failures[name] = type(exc).__name__
        if failures:
            manifest.update(status="failed", cleanup_errors=failures)
        manifest["finished_epoch_s"] = time.time()
        write_json(directory / "manifest.json", manifest)
        if failures:
            raise RuntimeError("Ray session cleanup failed")

    task = asyncio.create_task(cleanup())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            manifest.update(status="interrupted", error="CancelledError")
    task.result()
    if cancelled:
        raise asyncio.CancelledError


def main() -> None:
    """Run the fixed local experiment only after two independently owned engines are ready."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.controller, args.output))


if __name__ == "__main__":
    main()
