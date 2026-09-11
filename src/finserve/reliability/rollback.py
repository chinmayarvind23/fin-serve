"""SQLite truth and a lease-fenced rollback workflow with explicit external-action ambiguity."""

import asyncio
import hashlib
import math
import sqlite3
import time
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from finserve.contracts.deployment import (
    DeploymentState,
    HealthObservation,
    ImmutableModel,
    RegressionSignal,
    Revision,
    RollbackRecord,
    RollbackStatus,
)
from finserve.reliability.promotion import PromotionDecision

Table = Literal["revisions", "deployments", "rollbacks", "decisions", "activations"]


class ControlConflict(RuntimeError):
    """A stale generation, active rollback or competing worker prevents a state change."""


class ActivationReceipt(ImmutableModel):
    """An idempotent local acknowledgement of externally verified candidate activation."""

    deployment_id: str
    expected_generation: int
    decision_digest: str
    state: DeploymentState


class ApplyRequest(ImmutableModel):
    """Adapters must enforce generation/revision fencing and honor the same idempotency key."""

    deployment_id: str
    expected_revision: str
    expected_generation: int = Field(ge=0)
    target: Revision
    idempotency_key: str


class DeploymentAdapter(Protocol):
    """Implementations own real deployment actions and current-route readiness/smoke evidence."""

    supports_idempotency: bool

    async def apply(self, request: ApplyRequest) -> None:
        """Fence old generations and prevent duplicate effects for supported repeated keys."""
        ...

    async def health(self, deployment_id: str) -> HealthObservation:
        """Probe the active traffic route and report the immutable revision actually tested."""
        ...


def load_record[ModelT: BaseModel](
    connection: sqlite3.Connection, table: Table, key: str, model: type[ModelT]
) -> ModelT:
    """Only fixed internal table names are interpolated; all externally supplied IDs are bound."""
    row: tuple[str] | None = connection.execute(
        f"SELECT payload FROM {table} WHERE id=?", (key,)
    ).fetchone()
    if row is None:
        raise KeyError(f"missing {table} record")
    return model.model_validate_json(row[0])


def save_record(connection: sqlite3.Connection, table: Table, key: str, value: BaseModel) -> None:
    """Callers hold BEGIN IMMEDIATE and have checked the semantic transition before replacement."""
    connection.execute(
        f"INSERT INTO {table}(id,payload) VALUES(?,?) "
        "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
        (key, value.model_dump_json()),
    )


def validate_transition(
    operation: RollbackRecord, status: RollbackStatus, start_attempt: bool
) -> None:
    """Keep the workflow graph pure; acknowledged apply may only proceed through health checks."""
    allowed: dict[RollbackStatus, set[RollbackStatus]] = {
        "detected": {"detected", "applying"},
        "applying": {"applying", "verifying", "needs_reconciliation", "restored"},
        "verifying": {"verifying", "restored"},
        "needs_reconciliation": {"needs_reconciliation", "applying", "restored"},
        "restored": set(),
    }
    if status not in allowed[operation.status]:
        raise ControlConflict("illegal rollback transition")
    if start_attempt and status != "applying":
        raise ControlConflict("an attempt can only start in applying state")
    if status == "applying" and operation.status != "applying" and not start_attempt:
        raise ControlConflict("apply transition requires a persisted attempt")


class DeploymentStore:
    """One local durable truth database; transactions are short and never span network awaits."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        """Use WAL/FULL sync for crash recovery and real UTC epoch time by default."""
        self.path = path.resolve()
        repository = Path(__file__).resolve().parents[3]
        if self.path == repository or repository in self.path.parents:
            raise ValueError("control-plane truth must be outside the source repository")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS revisions(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS deployments(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rollbacks(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS activations(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS probation(deployment_id TEXT NOT NULL,
                    generation INTEGER NOT NULL, evidence_digest TEXT NOT NULL,
                    decision_digest TEXT NOT NULL, observed_at REAL NOT NULL,
                    PRIMARY KEY(deployment_id,generation));
                CREATE TABLE IF NOT EXISTS leases(id TEXT PRIMARY KEY, owner TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL, observed_at REAL NOT NULL, payload TEXT NOT NULL);
            """)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection]:
        """BEGIN IMMEDIATE serializes compare-and-swap decisions across processes."""
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def register_revision(self, revision: Revision) -> None:
        """Identical registration is idempotent; reusing its ID with new bits is forbidden."""
        with self.transaction() as connection:
            try:
                previous = load_record(connection, "revisions", revision.revision_id, Revision)
            except KeyError:
                save_record(connection, "revisions", revision.revision_id, revision)
                return
            if previous != revision:
                raise ControlConflict("immutable revision ID reused")

    def revision(self, revision_id: str) -> Revision:
        """Resolve rollback targets exclusively from the immutable revision registry."""
        with self.transaction() as connection:
            return load_record(connection, "revisions", revision_id, Revision)

    def deployment(self, deployment_id: str) -> DeploymentState:
        """Read the generation and rollback owner together to avoid mixed snapshots."""
        with self.transaction() as connection:
            return load_record(connection, "deployments", deployment_id, DeploymentState)

    def operation(self, operation_id: str) -> RollbackRecord:
        """Expose persisted truth after restarts; memory is not the source of recovery status."""
        with self.transaction() as connection:
            return load_record(connection, "rollbacks", operation_id, RollbackRecord)

    def history(self, operation_id: str) -> list[RollbackRecord]:
        """Return append-only transition snapshots in database order for drill evidence."""
        with self.transaction() as connection:
            rows: list[tuple[str]] = connection.execute(
                "SELECT payload FROM events WHERE operation_id=? ORDER BY sequence", (operation_id,)
            ).fetchall()
            return [RollbackRecord.model_validate_json(row[0]) for row in rows]

    def record_decision(self, decision: PromotionDecision) -> str:
        """Persist both failed and passed gate outcomes under their content identity."""
        key = hashlib.sha256(decision.model_dump_json().encode()).hexdigest()
        with self.transaction() as connection:
            save_record(connection, "decisions", key, decision)
        return key

    def decision(self, digest: str) -> PromotionDecision:
        """Read a prior evaluation outcome without rerunning or replacing failed evidence."""
        with self.transaction() as connection:
            return load_record(connection, "decisions", digest, PromotionDecision)

    def bootstrap(
        self, deployment_id: str, revision_id: str, health: HealthObservation
    ) -> DeploymentState:
        """Seed an existing health-verified deployment; this method does not deploy it."""
        with self.transaction() as connection:
            revision = load_record(connection, "revisions", revision_id, Revision)
            if not health.verifies(revision):
                raise ValueError("bootstrap requires exact revision health")
            try:
                existing = load_record(connection, "deployments", deployment_id, DeploymentState)
            except KeyError:
                existing = DeploymentState(
                    deployment_id=deployment_id,
                    active_revision=revision_id,
                    known_good_revision=revision_id,
                    generation=0,
                )
                save_record(connection, "deployments", deployment_id, existing)
                return existing
            if existing.generation != 0 or existing.active_revision != revision_id:
                raise ControlConflict("deployment already initialized differently")
            return existing

    def activate_candidate(
        self,
        deployment_id: str,
        expected_generation: int,
        decision: PromotionDecision,
        health: HealthObservation,
    ) -> DeploymentState:
        """Record externally verified activation; retain the previous known-good rollback target."""
        decision_key = self.record_decision(decision)
        intent = f"{deployment_id}:{expected_generation}:{decision_key}"
        activation_key = hashlib.sha256(intent.encode()).hexdigest()
        with self.transaction() as connection:
            state = load_record(connection, "deployments", deployment_id, DeploymentState)
            revision = load_record(connection, "revisions", decision.candidate_revision, Revision)
            if (
                not decision.approved
                or decision.candidate_digest != revision.digest()
                or not health.verifies(revision)
            ):
                raise ValueError("activation requires approved evidence and exact candidate health")
            try:
                receipt = load_record(connection, "activations", activation_key, ActivationReceipt)
            except KeyError:
                receipt = None
            if receipt is not None:
                return state
            if state.generation != expected_generation or state.rollback_id is not None:
                raise ControlConflict("stale activation or rollback in progress")
            updated = DeploymentState(
                deployment_id=deployment_id,
                active_revision=revision.revision_id,
                known_good_revision=state.known_good_revision,
                generation=state.generation + 1,
            )
            save_record(connection, "deployments", deployment_id, updated)
            save_record(
                connection,
                "activations",
                activation_key,
                ActivationReceipt(
                    deployment_id=deployment_id,
                    expected_generation=expected_generation,
                    decision_digest=decision_key,
                    state=updated,
                ),
            )
            return updated

    def detect(self, signal: RegressionSignal) -> RollbackRecord:
        """Persist detector time and freeze promotion if the signal names active traffic."""
        with self.transaction() as connection:
            try:
                existing = load_record(connection, "rollbacks", signal.signal_id, RollbackRecord)
            except KeyError:
                existing = None
            if existing is not None:
                if existing.signal != signal:
                    raise ControlConflict("signal idempotency key reused with different intent")
                return existing
            state = load_record(connection, "deployments", signal.deployment_id, DeploymentState)
            if (
                state.active_revision != signal.observed_revision
                or state.generation != signal.observed_generation
                or state.rollback_id is not None
                or state.active_revision == state.known_good_revision
            ):
                raise ControlConflict(
                    "stale signal, active rollback, or no distinct known-good target"
                )
            target = load_record(connection, "revisions", state.known_good_revision, Revision)
            received_at = self.clock()
            operation = RollbackRecord(
                operation_id=signal.signal_id,
                signal=signal,
                target_revision=target.revision_id,
                target_digest=target.digest(),
                detected_at=signal.detected_at,
                received_at=received_at,
            )
            save_record(connection, "rollbacks", operation.operation_id, operation)
            save_record(
                connection,
                "deployments",
                state.deployment_id,
                state.model_copy(update={"rollback_id": operation.operation_id}),
            )
            self._event(connection, operation)
            return operation

    def mark_stable(
        self,
        deployment_id: str,
        expected_generation: int,
        decision: PromotionDecision,
        health: HealthObservation,
        probation_evidence_digest: str,
    ) -> DeploymentState:
        """Advance known-good only after explicitly recorded probation evidence and exact health."""
        if len(probation_evidence_digest) != 64 or any(
            char not in "0123456789abcdef" for char in probation_evidence_digest
        ):
            raise ValueError("probation evidence requires a SHA256 digest")
        decision_key = self.record_decision(decision)
        with self.transaction() as connection:
            state = load_record(connection, "deployments", deployment_id, DeploymentState)
            revision = load_record(connection, "revisions", state.active_revision, Revision)
            if state.generation != expected_generation or state.rollback_id is not None:
                raise ControlConflict("stale probation or active rollback")
            if (
                not decision.approved
                or decision.candidate_digest != revision.digest()
                or decision.candidate_revision != revision.revision_id
                or not health.verifies(revision)
            ):
                raise ValueError("probation requires approved active revision and exact health")
            prior: tuple[str, str] | None = connection.execute(
                "SELECT evidence_digest,decision_digest FROM probation "
                "WHERE deployment_id=? AND generation=?",
                (deployment_id, expected_generation),
            ).fetchone()
            if prior is not None and prior != (probation_evidence_digest, decision_key):
                raise ControlConflict("probation evidence is immutable for a deployment generation")
            connection.execute(
                "INSERT INTO probation VALUES(?,?,?,?,?) "
                "ON CONFLICT(deployment_id,generation) DO NOTHING",
                (
                    deployment_id,
                    expected_generation,
                    probation_evidence_digest,
                    decision_key,
                    self.clock(),
                ),
            )
            updated = state.model_copy(update={"known_good_revision": state.active_revision})
            save_record(connection, "deployments", deployment_id, updated)
            return updated

    def _event(self, connection: sqlite3.Connection, operation: RollbackRecord) -> None:
        """Append transition evidence in the same transaction as its state change."""
        connection.execute(
            "INSERT INTO events(operation_id,observed_at,payload) VALUES(?,?,?)",
            (operation.operation_id, self.clock(), operation.model_dump_json()),
        )

    def claim(self, operation_id: str, owner: str, lease_seconds: float) -> RollbackRecord:
        """A durable lease bounds runners; expired owners cannot commit later transitions."""
        if not math.isfinite(lease_seconds) or lease_seconds <= 0 or not owner:
            raise ValueError("lease owner and finite positive duration required")
        with self.transaction() as connection:
            operation = load_record(connection, "rollbacks", operation_id, RollbackRecord)
            self._fence(connection, operation)
            row: tuple[str | None, float] | None = connection.execute(
                "SELECT owner,expires FROM leases WHERE id=?", (operation_id,)
            ).fetchone()
            now = self.clock()
            if row is not None and row[0] is not None and row[1] > now:
                raise ControlConflict("rollback worker lease is active")
            connection.execute(
                "INSERT INTO leases(id,owner,expires) VALUES(?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET owner=excluded.owner,expires=excluded.expires",
                (operation_id, owner, now + lease_seconds),
            )
            return operation

    def _fence(self, connection: sqlite3.Connection, operation: RollbackRecord) -> DeploymentState:
        """An old operation cannot mutate a newer deployment after a delayed response."""
        state = load_record(
            connection, "deployments", operation.signal.deployment_id, DeploymentState
        )
        if (
            state.rollback_id != operation.operation_id
            or state.generation != operation.signal.observed_generation
            or state.active_revision != operation.signal.observed_revision
        ):
            raise ControlConflict("rollback operation no longer owns active deployment")
        return state

    def update(
        self,
        operation_id: str,
        owner: str,
        status: RollbackStatus,
        error: str | None = None,
        start_attempt: bool = False,
        health: HealthObservation | None = None,
    ) -> RollbackRecord:
        """Recheck lease and generation; restoration additionally needs exact health."""
        with self.transaction() as connection:
            operation = load_record(connection, "rollbacks", operation_id, RollbackRecord)
            validate_transition(operation, status, start_attempt)
            state = self._fence(connection, operation)
            lease: tuple[str | None, float] | None = connection.execute(
                "SELECT owner,expires FROM leases WHERE id=?", (operation_id,)
            ).fetchone()
            now = self.clock()
            if lease is None or lease[0] != owner or lease[1] <= now:
                raise ControlConflict("rollback lease lost or expired")
            changes: dict[str, object] = {"status": status, "last_error": error}
            if start_attempt:
                changes.update(
                    started_at=operation.started_at or now,
                    apply_attempts=operation.apply_attempts + 1,
                )
            if status == "restored":
                target = load_record(connection, "revisions", operation.target_revision, Revision)
                if (
                    health is None
                    or not health.verifies(target)
                    or target.digest() != operation.target_digest
                ):
                    raise ValueError("restoration requires exact known-good health")
                changes["restored_at"] = now
                save_record(
                    connection,
                    "deployments",
                    state.deployment_id,
                    DeploymentState(
                        deployment_id=state.deployment_id,
                        active_revision=target.revision_id,
                        known_good_revision=target.revision_id,
                        generation=state.generation + 1,
                    ),
                )
            updated = RollbackRecord.model_validate({**operation.model_dump(), **changes})
            save_record(connection, "rollbacks", operation_id, updated)
            self._event(connection, updated)
            return updated

    def release(self, operation_id: str, owner: str) -> None:
        """A former owner cannot release a newer worker's lease."""
        with self.transaction() as connection:
            connection.execute(
                "UPDATE leases SET owner=NULL,expires=0 WHERE id=? AND owner=?",
                (operation_id, owner),
            )


class RollbackController:
    """Resume one bounded action/health cycle; external adapters are deliberately injected."""

    def __init__(self, store: DeploymentStore, timeout_seconds: float = 30) -> None:
        """The lease outlives bounded apply plus two health probes; retries remain explicit."""
        if not 0 < timeout_seconds <= 3600:
            raise ValueError("timeout must be finite and positive")
        self.store, self.timeout_seconds = store, timeout_seconds

    async def _health(self, adapter: DeploymentAdapter, deployment_id: str) -> HealthObservation:
        """Readiness checks get their own finite deadline and never imply that apply succeeded."""
        async with asyncio.timeout(self.timeout_seconds):
            return await adapter.health(deployment_id)

    async def resume(self, operation_id: str, adapter: DeploymentAdapter) -> RollbackRecord:
        """Reconcile ambiguity before replay; non-idempotent adapters are never blindly retried."""
        operation = self.store.operation(operation_id)
        if operation.status == "restored":
            return operation
        owner = str(uuid4())
        operation = self.store.claim(operation_id, owner, 3 * self.timeout_seconds + 5)
        try:
            return await self._run(operation, owner, adapter)
        except asyncio.CancelledError:
            current = self.store.operation(operation_id)
            self.store.update(operation_id, owner, current.status, error="CancelledError")
            raise
        except Exception as exc:
            current = self.store.operation(operation_id)
            return self.store.update(operation_id, owner, current.status, error=type(exc).__name__)
        finally:
            self.store.release(operation_id, owner)

    async def _run(
        self, operation: RollbackRecord, owner: str, adapter: DeploymentAdapter
    ) -> RollbackRecord:
        """Persist intent before apply, then require exact target readiness and smoke checks."""
        target = self.store.revision(operation.target_revision)
        if operation.status != "detected":
            health = await self._health(adapter, operation.signal.deployment_id)
            if health.verifies(target):
                return self.store.update(operation.operation_id, owner, "restored", health=health)
            if operation.status == "verifying" or adapter.supports_idempotency is not True:
                status: RollbackStatus = (
                    "verifying" if operation.status == "verifying" else "needs_reconciliation"
                )
                return self.store.update(
                    operation.operation_id, owner, status, error="health_not_verified"
                )
        operation = self.store.update(operation.operation_id, owner, "applying", start_attempt=True)
        request = ApplyRequest(
            deployment_id=operation.signal.deployment_id,
            expected_revision=operation.signal.observed_revision,
            expected_generation=operation.signal.observed_generation,
            target=target,
            idempotency_key=operation.operation_id,
        )
        async with asyncio.timeout(self.timeout_seconds):
            await adapter.apply(request)
        self.store.update(operation.operation_id, owner, "verifying")
        health = await self._health(adapter, operation.signal.deployment_id)
        if health.verifies(target):
            return self.store.update(operation.operation_id, owner, "restored", health=health)
        return self.store.update(
            operation.operation_id, owner, "verifying", error="health_not_verified"
        )
