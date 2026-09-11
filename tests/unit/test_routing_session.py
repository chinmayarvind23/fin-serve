"""CPU concurrency boundaries for experiment-owned Ray lifecycle operations."""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from finserve.benchmark.routing_session import finish_session, native_owned


async def test_native_deploy_drains_before_cancellation_propagates() -> None:
    """A cancelled caller cannot race actor shutdown against still-running native deployment."""
    entered, released = threading.Event(), threading.Event()
    finished = False

    def native() -> None:
        """Represent a blocking Ray constructor which has no asyncio cancellation protocol."""
        nonlocal finished
        entered.set()
        assert released.wait(3)
        finished = True

    task = asyncio.create_task(native_owned(native))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    released.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished


async def test_failed_serve_shutdown_still_stops_ray_and_finalizes_manifest(tmp_path: Path) -> None:
    """One cleanup error cannot suppress independent cluster cleanup or leave running evidence."""
    serve, ray = MagicMock(), MagicMock()
    serve.shutdown.side_effect = RuntimeError("fixture shutdown failure")
    manifest: dict[str, object] = {"status": "complete"}
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await finish_session(serve, ray, tmp_path, manifest)
    ray.shutdown.assert_called_once()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["status"] == "failed" and saved["cleanup_errors"] == {"serve": "RuntimeError"}
