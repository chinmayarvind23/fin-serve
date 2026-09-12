"""Durable local admission evidence; crashes never expire into a stream-drain claim."""

import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import Field

from finserve.contracts.deployment import ImmutableModel
from finserve.http_ownership import OwnedCloseStream

if TYPE_CHECKING:
    from finserve.reliability.warm_routes import RouteSnapshot, WarmBackend

ADMISSION_PROTOCOL = "durable-http-close-v1"


class WarmDrainReceipt(ImmutableModel):
    """An irreversible retirement and zero durable pins were observed in one transaction."""

    kind: Literal["warm-runtime-drain-v1"] = "warm-runtime-drain-v1"
    store_identity: str = Field(pattern=r"^[0-9a-f]{32}$")
    revision_id: str
    revision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    admission_protocol: Literal["durable-http-close-v1"] = ADMISSION_PROTOCOL
    remaining_admissions: Literal[0] = 0
    observed_at: float = Field(gt=0)


def initialize_admissions(connection: sqlite3.Connection, *, new_store: bool) -> None:
    """Only a fresh store can opt in; triggers prevent old writers creating unfenced routes."""
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS warm_admission_protocol(protocol TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS warm_admissions(
            admission_id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS warm_drain_receipts(
            revision_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
    """)
    connection.execute("BEGIN IMMEDIATE")
    try:
        # A legacy initializer racing creation may already have published a route.
        # Never retrofit that history: its executor might still hold an unfenced pin.
        if new_store and not connection.execute("SELECT 1 FROM warm_events LIMIT 1").fetchone():
            connection.execute(
                "INSERT OR IGNORE INTO warm_admission_protocol VALUES(?)", (ADMISSION_PROTOCOL,)
            )
        if connection.execute("SELECT 1 FROM warm_admission_protocol").fetchone():
            for table in ("warm_routes", "warm_events", "warm_actions"):
                for action in ("INSERT", "UPDATE"):
                    connection.execute(f"""
                        CREATE TRIGGER IF NOT EXISTS admission_{table}_{action}
                        BEFORE {action} ON {table}
                        WHEN json_extract(NEW.payload,'$.admission_protocol')
                             IS NOT '{ADMISSION_PROTOCOL}'
                        BEGIN SELECT RAISE(ABORT,'legacy admission protocol rejected'); END
                    """)
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def read_admission_protocol(
    connection: sqlite3.Connection,
) -> Literal["durable-http-close-v1"] | None:
    """Legacy databases without protocol metadata cannot establish drain eligibility."""
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='warm_admission_protocol'"
    ).fetchone()
    if not exists:
        return None
    rows = connection.execute("SELECT protocol FROM warm_admission_protocol").fetchall()
    if not rows:
        return None
    if rows != [(ADMISSION_PROTOCOL,)]:
        raise ValueError("unsupported durable admission protocol")
    return ADMISSION_PROTOCOL


@dataclass
class AdmissionLease:
    """A request retains its exact endpoint even after retirement blocks ordinary lookups."""

    admission_id: str
    snapshot: "RouteSnapshot"
    backend: "WarmBackend"
    backend_started: int = 0
    backend_closed: int = 0

    @property
    def verified_closed(self) -> bool:
        """A failed send without a response cannot masquerade as verified connection closure."""
        return self.backend_started == self.backend_closed


# Bind borrowed work to the exact frozen task input and owner asyncio task. Child tasks
# inherit ContextVars but cannot independently authorize or acknowledge this obligation.
collector_admission: ContextVar[tuple[AdmissionLease, str, str, object] | None] = ContextVar(
    "finserve_collector_admission", default=None
)


class AdmissionCloseStream(OwnedCloseStream):
    """Record positive close evidence even when cancellation interrupts the caller awaiting it."""

    def __init__(self, stream: httpx.AsyncByteStream, lease: AdmissionLease) -> None:
        """Retain one proof per returned response rather than accepting generator exhaustion."""
        super().__init__(stream)
        self.lease, self.recorded = lease, False

    async def aclose(self) -> None:
        """Repeated close calls cannot turn an earlier failed close into success."""
        try:
            await super().aclose()
        finally:
            task = self.close_task
            if (
                not self.recorded
                and task is not None
                and task.done()
                and not task.cancelled()
                and task.exception() is None
            ):
                self.lease.backend_closed += 1
                self.recorded = True


class AdmissionTransport(httpx.AsyncBaseTransport):
    """Wrap owned backend transports without changing request bytes or engine parsing."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """An injectable inner transport supports actual closure failure/cancellation tests."""
        self.transport = transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Mark entry before send; absence of a returned response leaves durable ambiguity."""
        from finserve.gateway.warm_route_app import request_admission

        lease = request_admission.get()
        if lease is not None:
            lease.backend_started += 1
        response = await self.transport.handle_async_request(request)
        if lease is not None:
            if not isinstance(response.stream, httpx.AsyncByteStream):
                raise TypeError("asynchronous backend stream required")
            response.stream = AdmissionCloseStream(response.stream, lease)
        return response

    async def aclose(self) -> None:
        """The pool closes at gateway shutdown, after individual request ownership settles."""
        await self.transport.aclose()
