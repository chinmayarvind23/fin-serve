"""Measure local JSON tracing overhead through separate real HTTP fixture processes.

This balanced sequence measures gateway instrumentation and local file export only.
It does not measure GPU serving, Ray propagation, OTLP network export or model quality.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from finserve.benchmark.owned_process import OwnedProcess
from finserve.benchmark.routing_cohort import repository_state
from finserve.benchmark.runner import RunConfig, prepare_output, run_benchmark, write_json
from finserve.benchmark.workload import WorkItem, Workload

ORDER = ("off", "sampled", "full", "full", "sampled", "off")
RATIOS = {"off": None, "sampled": "0.01", "full": "1"}
WORKLOAD = Workload(
    suite_id="original-http-tracing-overhead-v1",
    version=1,
    items=(WorkItem(case_id="echo-32", family="TRANSPORT", prompt="0123456789abcdef" * 2),),
)


async def ready(process: OwnedProcess, client: httpx.AsyncClient, url: str, identity: str) -> None:
    """Bound startup and retain process identity rather than accepting an exited child's URL."""
    async with asyncio.timeout(30):
        while True:
            if process.evidence()["returncode"] is not None:
                raise RuntimeError("owned gateway exited before readiness")
            try:
                response = await client.get(url + "/healthz", timeout=0.5)
                if response.status_code == 200 and response.json() == {
                    "status": "ok",
                    "model": "reference",
                    "revision": identity,
                }:
                    return
            except (httpx.HTTPError, TimeoutError):
                pass
            await asyncio.sleep(0.1)


async def cohort(mode: str, output: Path, port: int, revision: str) -> dict[str, Any]:
    """Keep process start and exporter drain outside the measured request interval."""
    directory = prepare_output(output)
    key = secrets.token_urlsafe(32)
    identity = revision + "-" + secrets.token_hex(8)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("FINSERVE_", "OTEL_"))
    }
    environment.update(
        FINSERVE_ENGINE="fixture",
        FINSERVE_MODEL="reference",
        FINSERVE_MAX_CONCURRENCY="16",
        FINSERVE_API_KEY=key,
        FINSERVE_REVISION=identity,
    )
    trace = directory / "traces.jsonl"
    if RATIOS[mode] is not None:
        environment.update(FINSERVE_TRACE_PATH=str(trace), FINSERVE_TRACE_SAMPLE_RATIO=RATIOS[mode])
    config = RunConfig(
        requests=1024,
        warmup=32,
        concurrency=8,
        timeout_s=10,
        model="reference",
        engine="fixture-character-v1",
        engine_config="json-tracing-" + mode,
        revision=revision,
        hardware=platform.platform(),
        model_revision="reference-character-v1",
        tokenizer_revision="unicode-codepoint-v1",
        cache_policy="no-cache",
    )
    receipt: dict[str, Any] = {"mode": mode, "status": "running", "started_epoch_s": time.time()}
    write_json(directory / "process.json", receipt)
    process: OwnedProcess | None = None
    try:
        process = OwnedProcess(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "finserve.gateway.app:from_env",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-access-log",
                "--log-level",
                "warning",
            ],
            environment,
            directory / "gateway.log",
        )
        async with httpx.AsyncClient(
            headers={"Authorization": "Bearer " + key},
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=config.concurrency),
        ) as client:
            url = f"http://127.0.0.1:{port}"
            await ready(process, client, url, identity)
            receipt["ready"] = process.evidence()
            receipt["summary"] = await run_benchmark(
                client, url + "/v1/completions", WORKLOAD, config, directory / "run"
            )
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt.update(status="failed", error=type(exc).__name__)
        raise
    finally:
        try:
            if process is not None:
                started = time.perf_counter()
                receipt["shutdown"] = await process.stop()
                receipt["shutdown_seconds"] = time.perf_counter() - started
        except BaseException as exc:
            receipt.update(status="failed", cleanup_error=type(exc).__name__)
            raise
        finally:
            receipt["finished_epoch_s"] = time.time()
            write_json(directory / "process.json", receipt)
    verify_receipt(receipt, directory, mode, config)
    return receipt


def verify_receipt(receipt: dict[str, Any], directory: Path, mode: str, config: RunConfig) -> None:
    """A parse or population failure must replace the successful process-exit status."""
    try:
        trace = directory / "traces.jsonl"
        raw = trace.read_bytes() if trace.exists() else b""
        spans = [json.loads(line) for line in raw.splitlines()]
        if (
            any(row["name"] != "finserve.inference" or row["status"] != "OK" for row in spans)
            or (mode == "full" and len(spans) != config.requests + config.warmup)
            or receipt["summary"]["successful_requests"] != config.requests
            or receipt["summary"]["generated_tokens"] != config.requests * 32
        ):
            raise ValueError("unexpected request or trace population")
        receipt["exported_spans_including_warmup"] = len(spans)
        receipt["trace_sha256"] = hashlib.sha256(raw).hexdigest() if trace.exists() else None
    except BaseException as exc:
        receipt.update(status="failed", validation_error=type(exc).__name__)
        raise
    finally:
        write_json(directory / "process.json", receipt)


def freeze(directory: Path) -> dict[str, Any]:
    """Archive source and freeze order before the event loop can start any measured process."""
    repository = Path(__file__).resolve().parents[1]
    state = repository_state()
    sources = {}
    source_directory = directory / "source"
    for path in [Path(__file__).resolve(), *sorted((repository / "src/finserve").rglob("*.py"))]:
        relative = path.relative_to(repository)
        destination = source_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        destination.write_bytes(raw)
        sources[str(relative)] = hashlib.sha256(raw).hexdigest()
    plan = {
        "scope": __doc__,
        "repository": state,
        "source_sha256": sources,
        "order": ORDER,
        "requests_per_cohort": 1024,
        "warmup_per_cohort": 32,
        "concurrency": 8,
        "workload": WORKLOAD.model_dump(),
        "workload_hash": WORKLOAD.digest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("httpx", "uvicorn", "fastapi", "opentelemetry-api", "opentelemetry-sdk")
        },
        "limitations": [
            "One balanced sequence on a shared workstation; host noise remains.",
            "Startup and exporter shutdown are excluded from measured requests.",
            "Local JSON export only; no GPU or hosted OTLP performance claim.",
        ],
    }
    write_json(directory / "plan.json", plan)
    return state


async def run(directory: Path, port: int, state: dict[str, Any]) -> None:
    """Preserve every attempted cohort and finalized process receipt in declared order."""
    results = []
    status = "failed"
    try:
        for index, mode in enumerate(ORDER, 1):
            results.append(
                await cohort(
                    mode, directory / f"cohort-{index}-{mode}", port, str(state["git_revision"])
                )
            )
        status = "complete"
    finally:
        write_json(directory / "result.json", {"status": status, "cohorts": results})


def main() -> None:
    """Use an operator-selected loopback port and fresh external evidence directory on Linux."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--port", type=int, default=8074, choices=range(1024, 65536), metavar="PORT"
    )
    args = parser.parse_args()
    directory = prepare_output(args.output.resolve())
    state = freeze(directory)
    asyncio.run(run(directory, args.port, state))


if __name__ == "__main__":
    main()
