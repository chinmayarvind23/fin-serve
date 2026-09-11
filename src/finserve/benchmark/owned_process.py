"""Linux-only process ownership for local engine scaling experiments, not fleet orchestration.

PID file descriptors fence signals against PID reuse. Unknown descendants fail cleanup
closed; neither command-name searches nor process-group-wide signals choose kill targets.
"""

import asyncio
import importlib
import os
import platform
import select
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class ProcessIdentity:
    """Retain kernel generation and ancestry, independent of mutable process display names."""

    pid: int
    parent: int
    session: int
    start_ticks: int
    state: str


def identity(pid: int) -> ProcessIdentity:
    """Parse only /proc kernel fields; command names may contain spaces and parentheses."""
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rindex(")") + 2 :].split()
    return ProcessIdentity(pid, int(fields[1]), int(fields[3]), int(fields[19]), fields[0])


def session_members(session_id: int) -> list[ProcessIdentity]:
    """Inventory this owned session, retaining unknown members as unresolved cleanup evidence."""
    result: list[ProcessIdentity] = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            current = identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if current.session == session_id and current.state != "Z":
            result.append(current)
    return result


class OwnedProcess:
    """One supervisor owns fixed trusted argv and pinned handles for its observed descendants."""

    def __init__(
        self,
        argv: Sequence[str],
        environment: Mapping[str, str],
        log_path: Path,
    ) -> None:
        """Spawn without a shell into a fresh session; a unique log preserves failed starts."""
        if platform.system() != "Linux" or not argv or not Path(argv[0]).is_absolute():
            raise ValueError("owned engine processes require Linux and an absolute executable")
        linux_os = importlib.import_module("os")
        linux_signal = importlib.import_module("signal")
        self._open_pid = cast(Callable[[int, int], int], linux_os.pidfd_open)
        self._send_pid = cast(Callable[[int, int, None, int], None], linux_signal.pidfd_send_signal)
        self._kill_signal = cast(int, linux_signal.SIGKILL)
        self._handles: dict[int, tuple[ProcessIdentity, int]] = {}
        self._stop_task: asyncio.Task[dict[str, object]] | None = None
        self.last_cleanup: dict[str, object] | None = None
        self.started_epoch_s = time.time()
        self.started_monotonic = time.monotonic()
        with log_path.open("xb") as log:
            self.process = subprocess.Popen(
                list(argv),
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            self.leader = identity(self.process.pid)
            self._pin(self.leader)
        except BaseException:
            # Popen retains the unreaped child here; its own signal method checks returncode.
            self.process.kill()
            self.process.wait(timeout=5)
            raise

    def _pin(self, observed: ProcessIdentity) -> None:
        """Check generation around pidfd_open to prevent attaching a reused numeric PID."""
        descriptor = self._open_pid(observed.pid, 0)
        try:
            if identity(observed.pid).start_ticks != observed.start_ticks:
                raise RuntimeError("process generation changed during ownership capture")
            self._handles[observed.pid] = (observed, descriptor)
        except BaseException:
            os.close(descriptor)
            raise

    def _alive(self, pid: int) -> bool:
        """A readable pidfd proves the captured process exited, even after numeric PID reuse."""
        return not select.select([self._handles[pid][1]], [], [], 0)[0]

    def observe(self) -> list[ProcessIdentity]:
        """Pin bounded child metadata only while its parent is still owned and alive."""
        members = session_members(self.leader.session)
        for _ in range(4):
            for child in members:
                if child.pid in self._handles or child.parent not in self._handles:
                    continue
                if not self._alive(child.parent):
                    continue
                if len(self._handles) >= 32:
                    raise RuntimeError("owned process descendant bound exceeded")
                try:
                    self._pin(child)
                except (ProcessLookupError, FileNotFoundError):
                    continue
        return members

    def evidence(self) -> dict[str, object]:
        """Record PID generations and executable arguments without environment secrets."""
        members = self.observe()
        return {
            "pid": self.leader.pid,
            "start_ticks": self.leader.start_ticks,
            "session": self.leader.session,
            "started_epoch_s": self.started_epoch_s,
            "argv": self.process.args,
            "owned_descendants": sorted(self._handles),
            "unowned_session_members": [
                item.pid for item in members if item.pid not in self._handles
            ],
            "returncode": self.process.poll(),
        }

    def _signal(self, pid: int, number: int) -> None:
        """Signal only a pinned kernel process identity; an exited target is already drained."""
        try:
            self._send_pid(self._handles[pid][1], number, None, 0)
        except ProcessLookupError:
            pass

    async def _stop(self, force: bool, grace_seconds: float) -> dict[str, object]:
        """Drain pinned handles even when discovery fails; unknown descendants remain unresolved."""
        discovery_errors: set[str] = set()

        def discover() -> None:
            """Discovery cannot prevent cleanup of already pinned handles."""
            try:
                self.observe()
            except Exception as exc:
                discovery_errors.add(type(exc).__name__)
            if force:
                for pid in tuple(self._handles):
                    self._signal(pid, self._kill_signal)

        discover()
        if force:
            for pid in tuple(self._handles):
                self._signal(pid, self._kill_signal)
        else:
            self._signal(self.leader.pid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while any(self._alive(pid) for pid in self._handles) and time.monotonic() < deadline:
            discover()
            await asyncio.sleep(0.05)
        for pid in tuple(self._handles):
            if self._alive(pid):
                self._signal(pid, self._kill_signal)
        deadline = time.monotonic() + 5
        # Native process exit has no application Event; poll pinned descriptors for at most 5s.
        while any(self._alive(pid) for pid in self._handles) and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.05)
        try:
            members = session_members(self.leader.session)
        except Exception as exc:
            discovery_errors.add(type(exc).__name__)
            members = []
        known_drained = not any(self._alive(pid) for pid in self._handles)
        self.last_cleanup = {
            "known_processes_drained": known_drained,
            "unresolved_session_members": [member.pid for member in members],
            "discovery_errors": sorted(discovery_errors),
        }
        if known_drained:
            self.process.wait(timeout=1)
        if members or not known_drained or discovery_errors:
            raise RuntimeError("owned process cleanup unresolved; session members remain")
        result: dict[str, object] = {
            "pid": self.leader.pid,
            "returncode": self.process.returncode,
            "drained": True,
            "forced": force,
        }
        for _, descriptor in self._handles.values():
            os.close(descriptor)
        return result

    async def stop(self, *, force: bool = False, grace_seconds: float = 30) -> dict[str, object]:
        """Suppress cancellation during drain; return its outcome or explicit unresolved failure.

        A caller's already-propagating cancellation resumes after its finally block. A failed
        discovery can be retried later; successful cleanup remains idempotent.
        """
        if not 0 <= grace_seconds <= 30:
            raise ValueError("cleanup grace must be between zero and 30 seconds")
        if self._stop_task is None or (
            self._stop_task.done() and self._stop_task.exception() is not None
        ):
            self._stop_task = asyncio.create_task(self._stop(force, grace_seconds))
        while not self._stop_task.done():
            try:
                await asyncio.shield(self._stop_task)
            except asyncio.CancelledError:
                continue
        return self._stop_task.result()
