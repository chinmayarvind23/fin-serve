"""Collect a real HTTP run together with frozen configuration and raw GPU observations."""

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

from finserve.benchmark.gpu import TelemetrySample, aggregate, collect
from finserve.benchmark.runner import RunConfig, benchmark_client, run_benchmark, write_json
from finserve.benchmark.workload import Workload


async def drain[T](task: asyncio.Task[T]) -> T:
    """Repeated cancellation cannot orphan a collector's bounded native subprocess."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def record_gpu(path: Path, stopped: asyncio.Event) -> list[TelemetrySample]:
    """Collect off-loop at 1 Hz; append each success/failure before waiting for another sample."""
    samples: list[TelemetrySample] = []
    with path.open("x", encoding="utf-8") as stream:
        while not stopped.is_set():
            sample = await asyncio.to_thread(collect)
            samples.append(sample)
            stream.write(sample.model_dump_json() + "\n")
            stream.flush()
            try:
                await asyncio.wait_for(stopped.wait(), timeout=1)
            except TimeoutError:
                continue
    return samples


def prepare_experiment(
    output: Path, workload: Workload, config: RunConfig
) -> tuple[Path, float, float]:
    """Complete filesystem and provenance setup before starting background measurement work."""
    repository = Path(__file__).resolve().parents[3]
    output = output.resolve()
    if output == repository or repository in output.parents:
        raise ValueError("experiment evidence must be outside the source repository")
    output.mkdir(parents=True, exist_ok=False)
    epoch_anchor, monotonic_anchor = time.time(), time.perf_counter()
    write_json(
        output / "environment.json",
        {
            "clock_epoch_anchor_s": epoch_anchor,
            "clock_monotonic_anchor_s": monotonic_anchor,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repository, text=True
            ).splitlines(),
            "configuration": config.model_dump(),
            "workload_hash": workload.digest(),
            "gpu_observation": "local nvidia-smi; fixed GPU inventory; query-end timestamps",
            "cost": "unmeasured: local hardware is not a billed cloud instance",
        },
    )
    return output, epoch_anchor, monotonic_anchor


async def experiment(
    url: str, output: Path, workload: Workload, config: RunConfig
) -> dict[str, object]:
    """Keep startup/measurement failures and declare local clock mapping uncertainty."""
    output, epoch_anchor, monotonic_anchor = await asyncio.to_thread(
        prepare_experiment, output, workload, config
    )
    stopped = asyncio.Event()
    collector = asyncio.create_task(record_gpu(output / "gpu.jsonl", stopped))
    status: dict[str, object] = {"status": "running"}
    samples: list[TelemetrySample] = []

    async def benchmark() -> dict[str, object]:
        """One pool is owned by the measured task, including its cancellation path."""
        async with benchmark_client(config) as client:
            return await run_benchmark(client, url, workload, config, output / "run")

    measured = asyncio.create_task(benchmark())
    try:
        done, _ = await asyncio.wait({collector, measured}, return_when=asyncio.FIRST_COMPLETED)
        if collector in done:
            await collector
            raise RuntimeError("GPU collector ended before experiment stopped")
        summary = await measured
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="interrupted", error=type(exc).__name__)
        measured.cancel()
        try:
            await drain(measured)
        except (Exception, asyncio.CancelledError):
            pass
        raise
    finally:
        stopped.set()
        try:
            samples = await drain(collector)
        except (Exception, asyncio.CancelledError) as exc:
            status["telemetry_error"] = type(exc).__name__
            if status["status"] == "completed":
                status["status"] = "telemetry_failed"
                raise
        finally:
            elapsed = time.perf_counter() - monotonic_anchor
            status["clock_drift_seconds"] = time.time() - epoch_anchor - elapsed
            write_json(output / "experiment-status.json", status)
    manifest = json.loads((output / "run" / "manifest.json").read_text())
    start = epoch_anchor + manifest["measured_started_s"] - monotonic_anchor
    end = epoch_anchor + manifest["measured_finished_s"] - monotonic_anchor
    utilization = aggregate(samples, start, end)
    if abs(float(status["clock_drift_seconds"])) > 0.1:
        utilization["average_gpu_utilization_percent"] = None
        utilization["clock_warning"] = "wall/monotonic clock drift exceeded 100ms"
    write_json(output / "gpu-summary.json", utilization)
    return {"serving": summary, "gpu": utilization}


def main() -> None:
    """Accept frozen JSON inputs rather than a changing command-line performance workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    workload = Workload.model_validate_json(args.workload.read_text())
    config = RunConfig.model_validate_json(args.config.read_text())
    result = asyncio.run(experiment(args.url, args.output, workload, config))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
