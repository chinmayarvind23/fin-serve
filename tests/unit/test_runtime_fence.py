"""Check kernel-fence platform and asyncio task boundaries independently of Docker fixtures."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from finserve.registry import runtime_fence
from finserve.registry.runtime_fence import attempt_fence


async def test_unsupported_platform_fails_before_lock_or_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows cannot silently degrade the cross-process fence to a process-local mutex."""
    monkeypatch.setattr(runtime_fence, "os", SimpleNamespace(name="nt"))
    with pytest.raises(RuntimeError, match="requires POSIX"):
        async with attempt_fence(tmp_path):
            pytest.fail("unsupported lock allowed protected work")
    assert not await asyncio.to_thread(lambda: list(tmp_path.iterdir()))


@pytest.mark.skipif(os.name != "posix", reason="POSIX kernel fence")
async def test_child_task_cannot_inherit_parent_lock_ownership(tmp_path: Path) -> None:
    """Reentrance is limited to the owner task even though asyncio copies context variables."""
    async def child() -> None:
        """The copied context cannot authorize work while the parent still owns its lock."""
        async with asyncio.timeout(0.1), attempt_fence(tmp_path):
            pytest.fail("child inherited operation ownership")

    async with attempt_fence(tmp_path):
        async with attempt_fence(tmp_path):
            with pytest.raises(TimeoutError):
                await asyncio.create_task(child())
    async with asyncio.timeout(1), attempt_fence(tmp_path):
        pass
