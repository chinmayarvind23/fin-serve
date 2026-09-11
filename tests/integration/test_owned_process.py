"""Real Linux CPU processes verify bounded ownership without engine or GPU dependencies."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from finserve.benchmark.owned_process import OwnedProcess, identity, session_members

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd ownership contract")


async def test_owned_tree_drains_without_signalling_unrelated_process(tmp_path: Path) -> None:
    """Force cleanup only pinned descendants; another live Python process must remain unaffected."""
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    script = (
        "import subprocess,sys,signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "print(p.pid,flush=True); time.sleep(60)"
    )
    unrelated = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)"
    )
    log = tmp_path / "owned.log"
    owned = OwnedProcess([sys.executable, "-c", script], os.environ, log)
    try:
        for _ in range(100):
            if log.read_text().strip():
                break
            await asyncio.sleep(0.01)
        child_pid = int(log.read_text().strip())
        evidence = owned.evidence()
        assert evidence["owned_descendants"] == sorted([owned.leader.pid, child_pid])
        assert evidence["unowned_session_members"] == []
        result = await owned.stop(grace_seconds=0.1)
        assert result["drained"] and not session_members(owned.leader.session)
        assert unrelated.returncode is None
        assert await owned.stop() == result
    finally:
        await owned.stop(grace_seconds=0.1)
        unrelated.terminate()
        await asyncio.wait_for(unrelated.wait(), 5)


async def test_repeated_cancellation_cannot_abandon_owned_shutdown(tmp_path: Path) -> None:
    """Cleanup suppresses caller cancellation so its returned result proves actual process exit."""
    script = (
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "print('ready',flush=True); time.sleep(60)"
    )
    log = tmp_path / "cancellation.log"
    owned = OwnedProcess([sys.executable, "-c", script], os.environ, log)
    for _ in range(100):
        if log.read_text().strip():
            break
        await asyncio.sleep(0.01)
    closing = asyncio.create_task(owned.stop(grace_seconds=0.2))
    await asyncio.sleep(0.01)
    closing.cancel()
    await asyncio.sleep(0.01)
    closing.cancel()
    assert not closing.done()
    assert (await closing)["drained"]
    assert not session_members(owned.leader.session)


async def test_discovery_failure_still_drains_known_process_and_can_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A proc inventory failure cannot strand a process already pinned by the supervisor."""
    owned = OwnedProcess(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        os.environ,
        tmp_path / "discovery-failure.log",
    )
    original = owned.observe

    def failed_discovery() -> list[object]:
        """Model a descendant-bound or proc-read failure before shutdown signals are sent."""
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(owned, "observe", failed_discovery)
    with pytest.raises(RuntimeError, match="unresolved"):
        await owned.stop(grace_seconds=0.1)
    assert owned.last_cleanup is not None and owned.last_cleanup["known_processes_drained"]
    assert owned.process.returncode is not None
    monkeypatch.setattr(owned, "observe", original)
    assert (await owned.stop(grace_seconds=0.1))["drained"]


def test_kernel_identity_records_current_process_generation() -> None:
    """Generation and ancestry come from the kernel rather than a mutable display-name match."""
    observed = identity(os.getpid())
    assert observed.pid == os.getpid() and observed.parent == os.getppid()
    assert observed.start_ticks > 0 and observed.session > 0
