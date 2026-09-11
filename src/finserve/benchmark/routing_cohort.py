"""One frozen HTTP-to-Ray routing cohort with raw requests and shared-device evidence."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from collections import deque
from collections.abc import Callable
from contextlib import aclosing
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from finserve.benchmark.gpu import TelemetrySample, aggregate
from finserve.benchmark.metrics import RequestRecord, summarize
from finserve.benchmark.routing_workload import (
    MODEL,
    REVISION,
    RoutingCase,
    RoutingWorkload,
    frozen_workload,
)
from finserve.benchmark.runner import prepare_output, write_json
from finserve.contracts.inference import InferenceRequest
from finserve.engines.ray_backends import BackendConfiguration
from finserve.engines.ray_http import RayHTTPEngine


class BackendIdentity(BaseModel):
    """Explicit process generations distinguish two actual engines from aliases of one server."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    replica_id: str
    endpoint: str
    pid: int = Field(gt=0, strict=True)
    start_ticks: int = Field(gt=0, strict=True)


class RoutingTopology(BaseModel):
    """Keep coordinator declarations separate from observed runtime identity and health."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Literal["Qwen/Qwen2.5-0.5B-Instruct"] = MODEL
    revision: Literal["7ae557604adf67be50417f59c2c2f167def9a775"] = REVISION
    engine_version: Literal["0.29.0"] = "0.29.0"
    physical_gpu_uuid: str = Field(min_length=1, max_length=256)
    backends: tuple[BackendIdentity, BackendIdentity]
    profile: dict[str, Any]

    @model_validator(mode="after")
    def distinct_engines(self) -> "RoutingTopology":
        """Reject repeated PIDs/names/endpoints before a two-backend denominator can be reported."""
        if (
            len({item.pid for item in self.backends}) != 2
            or len({item.replica_id for item in self.backends}) != 2
        ):
            raise ValueError("two distinct engine process identities required")
        BackendConfiguration(
            model=self.model, backends={item.replica_id: item.endpoint for item in self.backends}
        )
        if (
            self.profile.get("model") != self.model
            or self.profile.get("model_revision") != self.revision
        ):
            raise ValueError("profile model identity differs from frozen workload")
        return self


def verify_initial_status(status: dict[str, Any], topology: RoutingTopology) -> None:
    """Require both configured models healthy and one actual matching physical GPU before warmup."""
    workers = {row["replica_id"]: row for row in status["workers"]}
    if set(workers) != {item.replica_id for item in topology.backends} or not all(
        row["healthy"] and row["model"] == topology.model for row in workers.values()
    ):
        raise ValueError("two configured healthy engines required")
    physical = status["physical_gpu_observation"]
    if (
        physical.get("error")
        or physical.get("configured_shared_uuid") != topology.physical_gpu_uuid
    ):
        raise ValueError("configured physical GPU observation missing")
    if [row["uuid"] for row in physical["devices"]] != [topology.physical_gpu_uuid]:
        raise ValueError("physical GPU inventory differs from declared one-device experiment")


async def sample(
    engine: RayHTTPEngine,
    case: RoutingCase,
    index: int,
    phase: str,
    run_id: str,
    persist: Callable[[str, RequestRecord], None],
) -> RequestRecord:
    """Time from actual send to content/EOF, retaining partial evidence when transport fails."""
    request_id = f"{run_id}-{phase}-{index}"
    request = InferenceRequest(
        request_id=request_id,
        model=MODEL,
        prompt=case.prompt,
        max_tokens=case.max_tokens,
        temperature=0,
        timeout_seconds=30,
    )
    text, first, count, error, success = "", None, None, None, False
    started = time.perf_counter()
    try:
        async with aclosing(engine.stream(request)) as output:
            async for token in output:
                if token.text:
                    first = first if first is not None else time.perf_counter()
                    text += token.text
                if token.finish_reason is not None:
                    count = token.generated_tokens
        success = count is not None
        if not success:
            raise ValueError("missing terminal accounting")
    except BaseException as exc:
        error = type(exc).__name__
        if not isinstance(exc, Exception):
            raise
    finally:
        record = RequestRecord(
            logical_id=index,
            case_id=case.case_id,
            family=f"{case.input_class}-{case.max_tokens}",
            phase=phase,
            scheduled_s=started,
            send_s=started,
            first_content_s=first,
            complete_s=time.perf_counter(),
            success=success,
            error=error,
            generated_tokens=count,
            output=text,
        )
        persist(request_id, record)
    return record


async def phase_requests(
    engine: RayHTTPEngine,
    workload: RoutingWorkload,
    phase: str,
    run_id: str,
    persist: Callable[[str, RequestRecord], None],
) -> list[RequestRecord]:
    """Eight closed-loop workers retain every scheduled case, including interrupted leftovers."""
    cases = workload.cases[:8] if phase == "warmup" else workload.cases
    pending = deque(enumerate(cases))
    records: list[RequestRecord] = []

    async def worker() -> None:
        """A worker offers its next logical request only after its previous stream has closed."""
        while pending:
            index, case = pending.popleft()
            records.append(await sample(engine, case, index, phase, run_id, persist))

    try:
        async with asyncio.TaskGroup() as group:
            for _ in range(workload.concurrency):
                group.create_task(worker())
    finally:
        while pending:
            index, case = pending.popleft()
            now = time.perf_counter()
            persist(
                f"{run_id}-{phase}-{index}",
                RequestRecord(
                    logical_id=index,
                    case_id=case.case_id,
                    family=f"{case.input_class}-{case.max_tokens}",
                    phase=phase,
                    offered=False,
                    scheduled_s=now,
                    complete_s=now,
                    success=False,
                    error="run_interrupted_before_send",
                ),
            )
    return records


async def bounded_status(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    """Read bounded trusted-router metadata without letting a full response allocate arbitrarily."""
    async with asyncio.timeout(6):
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > 2 * 1024 * 1024:
                    raise ValueError("routing status exceeds bound")
                body.extend(chunk)
    value: Any = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("invalid routing status")
    return cast(dict[str, Any], value)


async def observe(
    client: httpx.AsyncClient,
    url: str,
    stop: asyncio.Event,
    path: Path,
    samples: list[TelemetrySample],
) -> None:
    """Poll one router-owned physical series; repeated cached epochs are never duplicate GPUs."""
    seen: set[float] = set()
    with path.open("x", encoding="utf-8") as target:
        while not stop.is_set():
            attempted = time.monotonic()
            try:
                status = await bounded_status(client, url)
                raw = status.get("physical_gpu_observation")
                if raw is None:
                    raise ValueError("physical observation missing")
                parsed = TelemetrySample.model_validate(
                    {
                        key: value
                        for key, value in raw.items()
                        if key not in {"router_observed_at", "configured_shared_uuid"}
                    }
                )
                if parsed.epoch_s not in seen and (
                    not samples or parsed.epoch_s > samples[-1].epoch_s
                ):
                    seen.add(parsed.epoch_s)
                    samples.append(parsed)
                event: dict[str, Any] = {"epoch_s": time.time(), "status": status}
            except Exception as exc:
                event = {"epoch_s": time.time(), "error": type(exc).__name__}
                samples.append(
                    TelemetrySample(
                        epoch_s=event["epoch_s"],
                        collection_seconds=time.monotonic() - attempted,
                        devices=[],
                        error=type(exc).__name__,
                    )
                )
            target.write(json.dumps(event, allow_nan=False) + "\n")
            target.flush()
            try:
                await asyncio.wait_for(stop.wait(), 1)
            except TimeoutError:
                pass


def archive_sources(output: Path) -> dict[str, str]:
    """Archive actual caller/router/policy bytes; a Git label alone cannot identify dirty code."""
    repository = Path(__file__).resolve().parents[3]
    hashes: dict[str, str] = {}
    paths = [
        "benchmark/routing_cohort.py",
        "benchmark/routing_workload.py",
        "benchmark/routing_session.py",
        "benchmark/gpu.py",
        "benchmark/metrics.py",
        "benchmark/runner.py",
        "engines/ray_http.py",
        "engines/ray_backends.py",
        "engines/ray_serve.py",
        "engines/backend_observations.py",
        "engines/openai_adapter.py",
        "engines/chat_protocol.py",
        "contracts/inference.py",
        "contracts/routing.py",
        "scheduler/policy.py",
        "scheduler/router.py",
        "telemetry/propagation.py",
        "telemetry/tracing.py",
    ]
    for relative in paths:
        raw = (repository / "src/finserve" / relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        (output / ("source__" + relative.replace("/", "__"))).write_bytes(raw)
    return hashes


async def run_cohort(
    url: str,
    workload: RoutingWorkload,
    policy: Literal["least_load", "adaptive"],
    output: Path,
    topology: RoutingTopology,
) -> dict[str, Any]:
    """Retain failures and interruptions; claims require complete matched policy IDs.

    This longer function keeps the experiment's evidence lifetime and phase boundaries visible.
    Token protocol validation and concurrent offering remain separately tested components.
    """
    if workload.digest() != frozen_workload().digest():
        raise ValueError("routing workload differs from the frozen held-out population")
    destination = httpx.URL(url)
    if (
        destination.scheme not in {"http", "https"}
        or not destination.host
        or destination.userinfo
        or destination.query
        or destination.fragment
    ):
        raise ValueError("invalid routing benchmark endpoint")
    key = os.getenv("FINSERVE_RAY_API_KEY")
    if not key:
        raise ValueError("benchmark requires configured internal routing authentication")
    directory = prepare_output(output)
    manifest: dict[str, Any] = {
        "run_id": uuid4().hex,
        "status": "running",
        "url": url,
        "policy": policy,
        "workload_sha256": workload.digest(),
        "started_epoch_s": time.time(),
        "topology": topology.model_dump(),
        "tracing": {
            "enabled": bool(
                os.getenv("FINSERVE_TRACE_PATH") or os.getenv("FINSERVE_OTLP_ENDPOINT")
            ),
            "sample_ratio": os.getenv("FINSERVE_TRACE_SAMPLE_RATIO", "0.01"),
            "scope": "launcher declaration; actor settings require deployment evidence",
        },
        "scope": "local shared-GPU routing; declared topology requires separate attestation",
    }
    write_json(directory / "manifest.json", manifest)
    try:
        manifest["sources"] = archive_sources(directory)
        manifest["repository"] = repository_state()
        (directory / "workload.json").write_bytes(workload.canonical_bytes())
    except BaseException as exc:
        manifest.update(status="failed", error=type(exc).__name__)
        write_json(directory / "manifest.json", manifest)
        raise
    samples: list[TelemetrySample] = []
    stop = asyncio.Event()
    headers = {"Authorization": f"Bearer {key}"}
    records: list[RequestRecord] = []
    request_ids: list[str] = []
    with (directory / "requests.jsonl").open("x", encoding="utf-8") as raw:

        def persist(request_id: str, record: RequestRecord) -> None:
            """Flush each attempt before offering further work, retaining interruption data."""
            records.append(record)
            request_ids.append(request_id)
            raw.write(json.dumps({"request_id": request_id, "record": record.model_dump()}) + "\n")
            raw.flush()

        async with httpx.AsyncClient(
            headers=headers, trust_env=False, follow_redirects=False
        ) as client:
            engine = RayHTTPEngine(url, api_key=key)
            observer = asyncio.create_task(
                observe(client, url, stop, directory / "status.jsonl", samples)
            )
            try:
                initial = await bounded_status(client, url)
                write_json(directory / "initial-status.json", initial)
                verify_initial_status(initial, topology)
                warmup = await phase_requests(
                    engine, workload, "warmup", manifest["run_id"], persist
                )
                if not all(record.success for record in warmup):
                    raise ValueError("warmup failed; frozen measured population not attempted")
                started_epoch, started = time.time(), time.perf_counter()
                await phase_requests(engine, workload, "measured", manifest["run_id"], persist)
                ended, ended_epoch = time.perf_counter(), time.time()
                stop.set()
                await observer
                write_json(
                    directory / "gpu-samples.json", [sample.model_dump() for sample in samples]
                )
                status = await bounded_status(client, url)
                write_json(directory / "final-status.json", status)
                selections = {row["request_id"]: row for row in status["recent_decisions"]}
                for request_id, record in zip(request_ids, records, strict=True):
                    if record.success and selections[request_id]["decision"]["policy"] != policy:
                        raise ValueError("observed policy differs from declared cohort")
                manifest.update(
                    status="complete",
                    measured_started_s=started,
                    measured_ended_s=ended,
                    measured_started_epoch_s=started_epoch,
                    measured_ended_epoch_s=ended_epoch,
                )
                write_json(
                    directory / "summary.json",
                    {
                        "requests": cohort_summary(records, workload, ended - started),
                        "physical_gpu": gpu_summary(
                            samples, started_epoch, ended_epoch, started, ended
                        ),
                    },
                )
            except BaseException as exc:
                manifest.update(
                    status="interrupted" if not isinstance(exc, Exception) else "failed",
                    error=type(exc).__name__,
                )
                raise
            finally:
                stop.set()
                await finish_owned(observer, engine, directory, manifest)
    return manifest


async def finish_owned(
    observer: asyncio.Task[None],
    engine: RayHTTPEngine,
    directory: Path,
    manifest: dict[str, Any],
) -> None:
    """Drain observation and HTTP pools before finalizing evidence, despite cancellation."""

    async def cleanup() -> None:
        """Preserve both shutdown ownership and final manifest state if either close path fails."""
        try:
            try:
                await observer
            finally:
                await engine.close()
        except BaseException as exc:
            manifest.update(status="failed", cleanup_error=type(exc).__name__)
            raise
        finally:
            manifest["finished_epoch_s"] = time.time()
            manifest["artifacts"] = artifact_hashes(directory)
            write_json(directory / "manifest.json", manifest)

    owned = asyncio.create_task(cleanup())
    cancelled = False
    while not owned.done():
        try:
            await asyncio.shield(owned)
        except asyncio.CancelledError:
            cancelled = True
            manifest.update(status="interrupted", error="CancelledError")
    owned.result()
    if cancelled:
        raise asyncio.CancelledError


def artifact_hashes(directory: Path) -> dict[str, str]:
    """Hash closed/flushed evidence after measurement; exclude the self-referential manifest."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }


def repository_state() -> dict[str, object]:
    """Record checkout dirtiness before warmup; archived bytes remain authoritative for source."""
    repository = Path(__file__).resolve().parents[3]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    return {
        "git_revision": head,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def gpu_summary(
    samples: list[TelemetrySample],
    start_epoch: float,
    end_epoch: float,
    start_monotonic: float,
    end_monotonic: float,
) -> dict[str, object]:
    """Invalidate device utilization if wall-clock mapping drift exceeds the existing 100ms gate."""
    result = aggregate(samples, start_epoch, end_epoch)
    drift = (end_epoch - start_epoch) - (end_monotonic - start_monotonic)
    result["clock_drift_seconds"] = drift
    if abs(drift) > 0.1:
        result["average_gpu_utilization_percent"] = None
        result["clock_warning"] = "wall/monotonic clock drift exceeded 100ms"
    return result


def cohort_summary(
    records: list[RequestRecord],
    workload: RoutingWorkload,
    duration: float,
) -> dict[str, object]:
    """Keep length and declared SLO slices separate from throughput and task correctness claims."""
    cases = {case.case_id: case for case in workload.cases}
    measured = [record for record in records if record.phase == "measured" and record.offered]
    slo_pass = sum(
        record.success
        and record.send_s is not None
        and record.first_content_s is not None
        and record.first_content_s - record.send_s <= cases[record.case_id].ttft_slo_seconds
        and record.complete_s - record.send_s <= cases[record.case_id].e2e_slo_seconds
        for record in measured
    )
    return {
        "all": summarize(records, duration),
        "by_input_output_budget": {
            family: summarize([record for record in records if record.family == family], duration)
            for family in sorted({record.family for record in measured})
        },
        "declared_slo_passes": slo_pass,
        "declared_slo_denominator": len(measured),
        "quality_scope": "mechanical held-out load; output agreement is not task correctness",
    }


def main() -> None:
    """Run only an already deployed routing endpoint; GPU process startup is separately owned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--policy", choices=("least_load", "adaptive"), required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("FINSERVE_RAY_API_KEY"):
        parser.error("FINSERVE_RAY_API_KEY must be configured for benchmark and router")
    asyncio.run(
        run_cohort(
            args.url,
            RoutingWorkload.model_validate_json(args.workload.read_bytes()),
            args.policy,
            args.output,
            RoutingTopology.model_validate_json(args.topology.read_bytes()),
        )
    )


if __name__ == "__main__":
    main()
