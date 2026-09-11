"""Isolate the benchmark event loop, preserving local ownership through cancellation and I/O."""

import asyncio
import json
import threading
import time
from pathlib import Path

from finserve.benchmark.experiment import drain, experiment
from finserve.contracts.performance import PerformanceCollectionSpec
from finserve.evaluation.quality import unique_object
from finserve.http_ownership import HTTPClosureError
from finserve.registry.managed_runtime import owned_directory


def evidence_bytes(directory: Path) -> int:
    """Count only the fixed collector outputs; arbitrary directory traversal is unnecessary."""
    return sum(
        path.stat().st_size
        for path in (
            directory / "environment.json",
            directory / "gpu.jsonl",
            directory / "experiment-status.json",
            directory / "gpu-summary.json",
            directory / "run" / "manifest.json",
            directory / "run" / "requests.jsonl",
            directory / "run" / "summary.json",
        )
        if path.exists()
    )


def verify_local_http_closure(directory: Path) -> None:
    """An ended request task does not reconcile a stream whose underlying close failed."""
    raw = directory / "run" / "requests.jsonl"
    if raw.exists():
        with raw.open(encoding="utf-8") as stream:
            for line in stream:
                if (
                    json.loads(line, object_pairs_hook=unique_object).get("error")
                    == "HTTPClosureError"
                ):
                    raise HTTPClosureError("performance HTTP cleanup remains unresolved")


async def worker_experiment(
    spec: PerformanceCollectionSpec, output: Path, cancelled: threading.Event
) -> None:
    """Poll cancellation/budgets on the isolated loop and retain the benchmark's terminal files."""
    start = time.monotonic()
    task = asyncio.create_task(
        experiment(
            spec.endpoint(),
            output,
            spec.workload,
            spec.configuration,
            collector_revision=spec.collector_revision,
        )
    )
    closure_failure: HTTPClosureError | None = None
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.05)
            if cancelled.is_set():
                raise asyncio.CancelledError
            if time.monotonic() - start > spec.timeout_seconds:
                raise TimeoutError("performance collection deadline exceeded")
            if evidence_bytes(output) > spec.maximum_raw_bytes:
                raise ValueError("performance collection output budget exceeded")
        await task
    except BaseException as failure:
        if isinstance(failure, HTTPClosureError):
            closure_failure = failure
        task.cancel()
        try:
            await drain(task)
        except HTTPClosureError as failure:
            closure_failure = failure
        except BaseException:
            pass
        raise
    finally:
        try:
            verify_local_http_closure(output)
        finally:
            # A partial raw write cannot erase the live task's unresolved transport receipt.
            if closure_failure is not None:
                raise closure_failure


async def collect_performance(specification: PerformanceCollectionSpec, output: Path) -> None:
    """A dedicated loop keeps synchronous benchmark writes off the orchestration loop.

    Cancellation signals the worker, then drains it under repeated cancellation. An in-flight
    native call may outlive the deadline, but its result cannot become a successful receipt.
    The byte limit is observed between bounded response writes; active records remain retained.
    """
    spec = PerformanceCollectionSpec.model_validate_json(specification.model_dump_json())
    cancelled = threading.Event()

    def run() -> None:
        """Own preparation, worker loop and its default executor until all native work finishes."""
        owned_directory(output.parent, create=True)
        if output.exists() or output.is_symlink():
            raise ValueError("performance output already exists")
        asyncio.run(worker_experiment(spec, output, cancelled))

    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        try:
            await drain(task)
        except BaseException:
            pass
        if not task.cancelled():
            failure = task.exception()
            if isinstance(failure, HTTPClosureError):
                raise failure from None
        raise
