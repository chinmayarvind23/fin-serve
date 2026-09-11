"""SQLite-backed single-coordinator visual jobs with durable intent and fenced terminal writes."""

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import TypeAdapter

from finserve.contracts.visual import (
    Identifier,
    VisualArtifact,
    VisualAttempt,
    VisualJob,
    VisualJobRequest,
)
from finserve.multimodal.visual_rpc import RPCExecution, VisualRPCClient


class JobConflict(ValueError):
    """The idempotency key already names a different immutable request."""


class JobCapacityError(RuntimeError):
    """Reject before durable acceptance when pending work or retained record capacity is full."""


class VisualJobStore:
    """A local durable database; production multi-host intent/artifacts belong in PostgreSQL/S3."""

    def __init__(self, path: Path, *, max_pending: int = 32, max_records: int = 4096) -> None:
        """Use WAL/FULL sync locally and preserve running attempts across restart."""
        if not 1 <= max_pending <= max_records <= 65536:
            raise ValueError("invalid job capacity")
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_pending, self.max_records = max_pending, max_records
        with self._transaction() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS visual_jobs (
                job_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                request_digest TEXT NOT NULL, request_json TEXT NOT NULL, state TEXT NOT NULL,
                generation INTEGER NOT NULL DEFAULT 0, attempt_generation INTEGER,
                worker_instance TEXT, artifact BLOB, artifact_sha256 TEXT, failure_type TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                UNIQUE(tenant, idempotency_key))""")

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        """Serialize admission and fences across database connections."""
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _job(row: sqlite3.Row) -> VisualJob:
        """Project private storage into a bounded public status without image or credential data."""
        return VisualJob.model_validate(
            {name: row[name] for name in VisualJob.model_fields if name != "model_revision"}
            | {"model_revision": json.loads(row["request_json"])["model_revision"]}
        )

    @staticmethod
    def _find(connection: sqlite3.Connection, tenant: str, job_id: str) -> sqlite3.Row:
        """Always scope lookup by authenticated tenant, including artifact and cancel operations."""
        row = connection.execute(
            "SELECT * FROM visual_jobs WHERE tenant=? AND job_id=?", (tenant, job_id)
        ).fetchone()
        if row is None:
            raise KeyError("visual job not found")
        return row

    def submit(self, tenant: str, key: str, request: VisualJobRequest) -> VisualJob:
        """Commit intent before returning and allow exact replay when saturated."""
        identifier = TypeAdapter[str](Identifier)
        identifier.validate_python(tenant)
        identifier.validate_python(key)
        encoded = json.dumps(request.model_dump(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self._transaction() as connection:
            previous = connection.execute(
                "SELECT * FROM visual_jobs WHERE tenant=? AND idempotency_key=?", (tenant, key)
            ).fetchone()
            if previous is not None:
                if previous["request_digest"] != digest:
                    raise JobConflict("idempotency key names a different visual request")
                return self._job(previous)
            total, pending = connection.execute("""SELECT COUNT(*), COALESCE(SUM(
                state IN ('queued','running','cancel_requested')),0) FROM visual_jobs""").fetchone()
            if total >= self.max_records or pending >= self.max_pending:
                raise JobCapacityError("visual job capacity exhausted")
            job_id, now = str(uuid4()), time.time()
            connection.execute(
                """INSERT INTO visual_jobs
                (job_id,tenant,idempotency_key,request_digest,request_json,state,created_at,updated_at)
                VALUES(?,?,?,?,?,'queued',?,?)""",
                (job_id, tenant, key, digest, encoded, now, now),
            )
            return self._job(self._find(connection, tenant, job_id))

    def get(self, tenant: str, job_id: str) -> VisualJob:
        """Polling reads status only and has no execution or cancellation side effect."""
        with self._transaction() as connection:
            return self._job(self._find(connection, tenant, job_id))

    def claim_next(self) -> tuple[str, VisualAttempt] | None:
        """Claim only queued work; ambiguous running attempts survive restart untouched."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM visual_jobs WHERE state='queued' ORDER BY created_at,job_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            generation = row["generation"] + 1
            connection.execute(
                """UPDATE visual_jobs SET state='running',generation=?,
                attempt_generation=?,updated_at=? WHERE job_id=?""",
                (generation, generation, time.time(), row["job_id"]),
            )
            return row["tenant"], VisualAttempt(
                job_id=row["job_id"],
                generation=generation,
                request=VisualJobRequest.model_validate_json(row["request_json"]),
            )

    def record_worker(self, execution: RPCExecution) -> None:
        """Persist observed process identity even after the cancellation fence changes."""
        with self._transaction() as connection:
            connection.execute(
                """UPDATE visual_jobs SET worker_instance=? WHERE job_id=?
                AND attempt_generation=? AND state IN ('running','cancel_requested')""",
                (execution.worker_instance, execution.attempt.job_id, execution.attempt.generation),
            )

    def request_cancel(self, tenant: str, job_id: str) -> VisualJob:
        """Fence late output; only never-claimed queued work is immediately cancelled."""
        with self._transaction() as connection:
            row = self._find(connection, tenant, job_id)
            if row["state"] in ("queued", "running"):
                state = "cancelled" if row["state"] == "queued" else "cancel_requested"
                connection.execute(
                    """UPDATE visual_jobs SET state=?,generation=generation+1,
                    updated_at=? WHERE job_id=?""",
                    (state, time.time(), job_id),
                )
            return self._job(self._find(connection, tenant, job_id))

    def complete(self, attempt: VisualAttempt, artifact: VisualArtifact) -> bool:
        """Commit bytes and success in one transaction only if the current fence still owns work."""
        if hashlib.sha256(artifact.png).hexdigest() != artifact.sha256:
            raise ValueError("artifact integrity mismatch")
        with self._transaction() as connection:
            result = connection.execute(
                """UPDATE visual_jobs SET state='succeeded',artifact=?,
                artifact_sha256=?,updated_at=? WHERE job_id=? AND generation=? AND state='running'
                """,
                (artifact.png, artifact.sha256, time.time(), attempt.job_id, attempt.generation),
            )
            return result.rowcount == 1

    def finish_cancel(
        self,
        attempt: VisualAttempt,
        *,
        drained: bool,
        terminal: Literal["cancelled", "failed"] = "cancelled",
        failure_type: str | None = None,
    ) -> bool:
        """Unknown worker lifetime cannot authorize cancellation or free pending quota."""
        if not drained:
            return False
        with self._transaction() as connection:
            result = connection.execute(
                """UPDATE visual_jobs SET state=?,failure_type=?,updated_at=?
                WHERE job_id=? AND attempt_generation=? AND generation=?
                AND state='cancel_requested'
                """,
                (
                    terminal,
                    failure_type,
                    time.time(),
                    attempt.job_id,
                    attempt.generation,
                    attempt.generation + 1,
                ),
            )
            return result.rowcount == 1

    def artifact(self, tenant: str, job_id: str) -> bytes:
        """Read a tenant-owned successful artifact and recheck stored bytes before download."""
        with self._transaction() as connection:
            row = self._find(connection, tenant, job_id)
            if row["state"] != "succeeded":
                raise JobConflict("visual artifact is not available")
            content = bytes(row["artifact"])
            if hashlib.sha256(content).hexdigest() != row["artifact_sha256"]:
                raise RuntimeError("stored visual artifact failed integrity")
            return content

    def unresolved(self) -> list[tuple[str, RPCExecution]]:
        """Expose restart ambiguity for explicit reconciliation without silently reclaiming work."""
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM visual_jobs WHERE state IN ('running','cancel_requested')"
            ).fetchall()
            return [
                (
                    row["tenant"],
                    RPCExecution(
                        VisualAttempt(
                            job_id=row["job_id"],
                            generation=row["attempt_generation"],
                            request=VisualJobRequest.model_validate_json(row["request_json"]),
                        ),
                        worker_instance=row["worker_instance"],
                    ),
                )
                for row in rows
            ]


class VisualJobCoordinator:
    """One process owns the dispatch loop; HTTP request/poll lifetimes never own accepted work."""

    def __init__(self, store: VisualJobStore, client: VisualRPCClient) -> None:
        """Keep one active RPC and a bounded durable queue without an in-memory backlog."""
        self.store, self.client = store, client
        self._loop_task: asyncio.Task[None] | None = None
        self._execution_task: asyncio.Task[None] | None = None
        self._execution: RPCExecution | None = None
        self._wake = asyncio.Event()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self.failure_type: str | None = None

    @property
    def ready(self) -> bool:
        """Expose dispatcher readiness separately from durable storage availability."""
        return self._loop_task is not None and not self._loop_task.done() and not self._closing

    async def submit(self, tenant: str, key: str, request: VisualJobRequest) -> VisualJob:
        """Run SQLite outside the event loop and wake dispatch after durable commit."""
        if self._closing or (self._loop_task is not None and self._loop_task.done()):
            raise RuntimeError("visual coordinator is unavailable")
        job = await asyncio.to_thread(self.store.submit, tenant, key, request)
        self._wake.set()
        return job

    def start(self) -> None:
        """Resume queued intent only; old running jobs remain visible as unresolved attempts."""
        if self._loop_task is not None or self._closing:
            raise RuntimeError("visual coordinator cannot start twice")
        self._loop_task = asyncio.create_task(self._run())
        self._loop_task.add_done_callback(self._dispatch_finished)

    def _dispatch_finished(self, task: asyncio.Task[None]) -> None:
        """Observe dispatcher failures without leaking exception details or accepting more work."""
        if task.cancelled():
            self.failure_type = "CancelledError"
        elif (error := task.exception()) is not None:
            self.failure_type = type(error).__name__

    async def _run(self) -> None:
        """Serialize work to match the reference worker's single native-compute capacity."""
        while not self._closing:
            self._wake.clear()
            claimed = await asyncio.to_thread(self.store.claim_next)
            if claimed is None:
                try:
                    async with asyncio.timeout(1):
                        await self._wake.wait()
                except TimeoutError:
                    pass
                continue
            tenant, attempt = claimed
            self._execution = RPCExecution(attempt)
            self._execution_task = asyncio.create_task(self._execute(tenant, self._execution))
            await asyncio.gather(self._execution_task, return_exceptions=True)
            self._execution, self._execution_task = None, None

    async def _observe_admission(self, execution: RPCExecution) -> None:
        """Persist observed server identity, including cancelled work."""
        await execution.admitted.wait()
        await asyncio.to_thread(self.store.record_worker, execution)

    async def _execute(self, tenant: str, execution: RPCExecution) -> None:
        """Fence lost RPCs and resolve them only with a same-instance barrier."""
        observer = asyncio.create_task(self._observe_admission(execution))
        try:
            state = await asyncio.to_thread(self.store.get, tenant, execution.attempt.job_id)
            if state.state != "running" or self._closing:
                await asyncio.to_thread(self.store.request_cancel, tenant, execution.attempt.job_id)
                await asyncio.to_thread(self.store.finish_cancel, execution.attempt, drained=True)
                return
            artifact = await self.client.generate(execution)
            committed = await asyncio.to_thread(self.store.complete, execution.attempt, artifact)
            if not committed:
                await asyncio.to_thread(self.store.finish_cancel, execution.attempt, drained=True)
        except BaseException as error:
            cleanup = asyncio.create_task(self._cleanup(tenant, execution, error))
            await self._await_owned(cleanup)
        finally:
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)

    @staticmethod
    async def _await_owned(task: asyncio.Task[None]) -> None:
        """Drain an owned cleanup through repeated outer cancellation, including all SQL writes."""
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        task.result()

    async def _cleanup(self, tenant: str, execution: RPCExecution, error: BaseException) -> None:
        """Keep persistence, remote barrier and terminal commit inside one shielded owner."""
        await asyncio.to_thread(self.store.record_worker, execution)
        await asyncio.to_thread(self.store.request_cancel, tenant, execution.attempt.job_id)
        try:
            drained = await self.client.cancel(execution)
        except Exception:
            drained = False
        terminal = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
        await asyncio.to_thread(
            self.store.finish_cancel,
            execution.attempt,
            drained=drained,
            terminal=terminal,
            failure_type=None if terminal == "cancelled" else type(error).__name__,
        )

    async def cancel(self, tenant: str, job_id: str) -> VisualJob:
        """Explicit cancellation fences immediately and waits only for coordinator-owned cleanup."""
        job = await asyncio.to_thread(self.store.request_cancel, tenant, job_id)
        if self._execution and self._execution.attempt.job_id == job_id and self._execution_task:
            if (
                self._execution.started
                and job.state == "cancel_requested"
                and not self._execution_task.cancelling()
            ):
                self._execution_task.cancel()
            await asyncio.shield(self._execution_task)
        self._wake.set()
        return await asyncio.to_thread(self.store.get, tenant, job_id)

    async def close(self) -> None:
        """Stop dispatch, fence/drain owned work, then close the channel; queued intent stays."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._shutdown())
        await self._await_owned(self._close_task)

    async def _shutdown(self) -> None:
        """One shutdown task owns native cleanup and channel close regardless of caller lifetime."""
        self._closing = True
        self._wake.set()
        if (
            self._execution_task is not None
            and self._execution
            and self._execution.started
            and not self._execution_task.cancelling()
        ):
            self._execution_task.cancel()
        try:
            if self._loop_task is not None:
                await self._loop_task
        finally:
            await self.client.close()
