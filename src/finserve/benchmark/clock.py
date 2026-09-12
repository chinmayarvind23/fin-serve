"""Identify a perf-counter domain without confusing replay or forked collectors."""

import os
import sys
from uuid import uuid4

_process_nonce = uuid4().hex


def _fork_domain() -> None:
    """PID reuse among successive fork children must not reuse the parent's nonce."""
    global _process_nonce
    _process_nonce = uuid4().hex


if sys.platform != "win32":
    os.register_at_fork(after_in_child=_fork_domain)


def clock_domain() -> str:
    """Threads share identity; fork children and freshly started processes cannot reuse it."""
    return f"{_process_nonce}:{os.getpid()}"
