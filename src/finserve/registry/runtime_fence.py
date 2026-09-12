"""Kernel-owned attempt fences; lock files are never unlinked or expired by wall time."""

import asyncio
import errno
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

RUNTIME_OPERATION_PROTOCOL = "posix-flock-abort-v1"


# Task identity prevents a child inheriting its parent's context from bypassing the fence.
_held: ContextVar[tuple[tuple[str, object], ...]] = ContextVar("runtime_fences", default=())


@asynccontextmanager
async def attempt_fence(directory: Path) -> AsyncGenerator[None]:
    """Serialize cooperating processes, releasing on death and only after owned I/O drains."""
    if os.name != "posix":
        raise RuntimeError("managed runtime fencing requires POSIX flock")
    import fcntl

    key = str(directory)
    owner = asyncio.current_task()
    if (key, owner) in _held.get():
        yield
        return
    # These short local operations intentionally have no cancellation boundary between
    # acquiring a descriptor/lock and registering its finally cleanup.
    path = directory / "operation.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                await asyncio.sleep(0.02)
        token = _held.set((*_held.get(), (key, owner)))
        try:
            yield
        finally:
            _held.reset(token)
    finally:
        os.close(descriptor)
