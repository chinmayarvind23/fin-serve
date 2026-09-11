"""Native telemetry lifetime must remain owned when an experiment is cancelled repeatedly."""

import asyncio
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from finserve.benchmark.experiment import experiment
from finserve.benchmark.gpu import TelemetrySample
from finserve.benchmark.runner import RunConfig
from finserve.benchmark.workload import default_workload


async def test_repeated_cancellation_drains_native_collection(tmp_path: Path) -> None:
    """A real worker thread models the nvidia-smi query without requiring GPU hardware."""
    entered, release = threading.Event(), threading.Event()

    def blocked_collection() -> TelemetrySample:
        """Hold collection until cancellation has exercised cleanup ownership."""
        entered.set()
        assert release.wait(5)
        return TelemetrySample(epoch_s=1, collection_seconds=1, devices=[], error="fixture")

    async def blocked_benchmark(*args: object, **kwargs: object) -> dict[str, object]:
        """The benchmark is cancelled while collection is still running in a thread."""
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    with patch("finserve.benchmark.experiment.collect", side_effect=blocked_collection):
        with patch("finserve.benchmark.experiment.run_benchmark", side_effect=blocked_benchmark):
            task = asyncio.create_task(
                experiment("http://test", tmp_path / "run", default_workload(), RunConfig())
            )
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert '"error":"fixture"' in (tmp_path / "run" / "gpu.jsonl").read_text()
    assert "interrupted" in (tmp_path / "run" / "experiment-status.json").read_text()
