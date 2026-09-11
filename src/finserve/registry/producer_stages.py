"""Durable immutable stage inputs and fenced attempt receipts for trusted offline producers."""

import time
from collections.abc import Callable
from typing import Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, insert, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from finserve.contracts.deployment import ImmutableModel
from finserve.registry.artifacts import ArtifactRef, ArtifactStore
from finserve.registry.metadata import Registry, RegistryConflict

schema = MetaData()
stages = Table(
    "producer_stages",
    schema,
    Column("stage_id", String(256), primary_key=True),
    Column("version", Integer, nullable=False),
    Column("payload", Text, nullable=False),
)
events = Table(
    "producer_stage_events",
    schema,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("stage_id", String(256), nullable=False),
    Column("payload", Text, nullable=False),
)


class StageState(ImmutableModel):
    """A crashed running attempt stays ambiguous until its trusted executor reconciles it."""

    stage_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}:[A-Za-z0-9_-]{1,64}$")
    input: ArtifactRef
    status: Literal["planned", "running", "failed", "completed"] = "planned"
    version: int = Field(default=0, ge=0, strict=True)
    attempt_number: int = Field(default=0, ge=0, le=32, strict=True)
    attempt_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    started_at: float | None = Field(default=None, gt=0)
    finished_at: float | None = Field(default=None, gt=0)
    output: ArtifactRef | None = None
    reconciliation: ArtifactRef | None = None
    error_code: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

    @model_validator(mode="after")
    def coherent_attempt(self) -> Self:
        """Reject invented completion, backwards timestamps and results without an owned attempt."""
        if self.status == "planned":
            if self.attempt_number or self.attempt_id or self.started_at or self.finished_at:
                raise ValueError("planned stage cannot have attempt observations")
        elif self.attempt_number < 1 or self.attempt_id is None or self.started_at is None:
            raise ValueError("stage outcome requires an owned attempt")
        if self.status in {"completed", "failed"}:
            if (
                self.finished_at is None
                or self.started_at is None
                or self.finished_at < self.started_at
            ):
                raise ValueError("stage finish must follow actual start")
        elif self.finished_at is not None:
            raise ValueError("unfinished stage cannot have a finish timestamp")
        if (self.status == "completed") != (self.output is not None):
            raise ValueError("only a completed stage can publish output")
        if (self.status == "failed") != (self.error_code is not None):
            raise ValueError("failed stages require a bounded error code")
        if (self.status == "failed") != (self.reconciliation is not None):
            raise ValueError("failed stages require retained reconciliation evidence")
        return self


class ProducerStages:
    """The journal is not authorization: server-owned executors validate each stage's semantics."""

    def __init__(
        self, registry: Registry, artifacts: ArtifactStore, clock: Callable[[], float] = time.time
    ) -> None:
        """Use the registry's SQLAlchemy engine and a verified immutable artifact namespace."""
        self.registry, self.artifacts, self.clock = registry, artifacts, clock
        schema.create_all(registry.engine)

    def _read(self, connection: Connection, stage_id: str) -> StageState:
        """Read authoritative payload inside the transaction, never caller-supplied state."""
        payload = connection.execute(
            select(stages.c.payload).where(stages.c.stage_id == stage_id)
        ).scalar_one()
        return StageState.model_validate_json(payload)

    def state(self, stage_id: str) -> StageState:
        """Expose incomplete attempts for reconciliation before any action replay."""
        with self.registry.engine.connect() as connection:
            return self._read(connection, stage_id)

    def declare(self, stage_id: str, reference: ArtifactRef) -> StageState:
        """Verify input bytes and reject changed specifications under an existing stage."""
        self.artifacts.get(reference)
        initial = StageState(stage_id=stage_id, input=reference)
        try:
            with self.registry.engine.begin() as connection:
                connection.execute(
                    insert(stages).values(
                        stage_id=stage_id, version=0, payload=initial.model_dump_json()
                    )
                )
                connection.execute(
                    insert(events).values(stage_id=stage_id, payload=initial.model_dump_json())
                )
        except IntegrityError:
            actual = self.state(stage_id)
            if actual.input != reference:
                raise RegistryConflict("producer stage input identity changed") from None
            return actual
        return initial

    def _save(self, connection: Connection, before: StageState, after: StageState) -> StageState:
        """A compare-and-swap prevents concurrent or stale attempts from replacing current state."""
        result = connection.execute(
            update(stages)
            .where(stages.c.stage_id == before.stage_id, stages.c.version == before.version)
            .values(version=after.version, payload=after.model_dump_json())
        )
        if result.rowcount != 1:
            raise RegistryConflict("producer stage changed concurrently")
        connection.execute(
            insert(events).values(stage_id=after.stage_id, payload=after.model_dump_json())
        )
        return after

    def start(self, stage_id: str, *, attempt_id: str | None = None) -> StageState:
        """Only planned or explicitly failed work starts; elapsed time cannot resolve ambiguity."""
        # Artifact access can reach S3; verify immutable input outside a database transaction.
        self.artifacts.get(self.state(stage_id).input)
        with self.registry.engine.begin() as connection:
            before = self._read(connection, stage_id)
            if before.status not in {"planned", "failed"}:
                raise RegistryConflict(
                    "running or completed producer stage requires reconciliation"
                )
            if attempt_id is not None:
                prior = connection.execute(
                    select(events.c.payload).where(events.c.stage_id == stage_id)
                ).scalars()
                if any(
                    StageState.model_validate_json(value).attempt_id == attempt_id
                    for value in prior
                ):
                    raise RegistryConflict("producer attempt identity cannot be reused")
            after = StageState(
                stage_id=stage_id,
                input=before.input,
                status="running",
                version=before.version + 1,
                attempt_number=before.attempt_number + 1,
                attempt_id=attempt_id if attempt_id is not None else uuid4().hex,
                started_at=self.clock(),
            )
            return self._save(connection, before, after)

    def finish(self, stage_id: str, attempt_id: str, output: ArtifactRef) -> StageState:
        """Publish a semantically verified receipt only for the exact still-current attempt."""
        self.artifacts.get(output)
        with self.registry.engine.begin() as connection:
            before = self._read(connection, stage_id)
            if (
                before.status == "completed"
                and before.attempt_id == attempt_id
                and before.output == output
            ):
                return before
            self._require_attempt(before, attempt_id)
            after = StageState.model_validate(
                {
                    **before.model_dump(),
                    "status": "completed",
                    "version": before.version + 1,
                    "output": output,
                    "finished_at": self.clock(),
                }
            )
            return self._save(connection, before, after)

    def fail(
        self, stage_id: str, attempt_id: str, error_code: str, *, reconciliation: ArtifactRef
    ) -> StageState:
        """Retain proof of drained or reconciled work before allowing a new attempt."""
        self.artifacts.get(reconciliation)
        with self.registry.engine.begin() as connection:
            before = self._read(connection, stage_id)
            self._require_attempt(before, attempt_id)
            after = StageState.model_validate(
                {
                    **before.model_dump(),
                    "status": "failed",
                    "version": before.version + 1,
                    "error_code": error_code,
                    "reconciliation": reconciliation,
                    "finished_at": self.clock(),
                }
            )
            return self._save(connection, before, after)

    def _require_attempt(self, state: StageState, attempt_id: str) -> None:
        """Tokens from a superseded attempt cannot finish or fail a newer producer operation."""
        if state.status != "running" or state.attempt_id != attempt_id:
            raise RegistryConflict("producer attempt is stale or not running")

    def history(self, stage_id: str) -> list[StageState]:
        """Every attempt and receipt remains visible after retries or a rejected optimization."""
        with self.registry.engine.connect() as connection:
            payloads = connection.execute(
                select(events.c.payload)
                .where(events.c.stage_id == stage_id)
                .order_by(events.c.event_id)
            ).scalars()
            return [StageState.model_validate_json(payload) for payload in payloads]
