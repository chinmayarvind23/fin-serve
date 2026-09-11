"""SQLAlchemy metadata truth using portable constraints and transactional compare-and-swap."""

import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Literal

from pydantic import Field
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import ConnectionPoolEntry

from finserve.benchmark.runner import validate_evidence
from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.registry.artifacts import ArtifactRef, ArtifactStore
from finserve.reliability.promotion import PromotionDecision, verify_candidate_identity

schema = MetaData()
models = Table(
    "finserve_models",
    schema,
    Column("id", String(64), primary_key=True),
    Column("digest", String(64), nullable=False),
    Column("payload", Text, nullable=False),
)
revisions = Table(
    "finserve_revisions",
    schema,
    Column("id", String(128), primary_key=True),
    Column("model_id", String(64), ForeignKey(models.c.id), nullable=False),
    Column("digest", String(64), nullable=False),
    Column("payload", Text, nullable=False),
)
definitions = Table(
    "finserve_workloads",
    schema,
    Column("id", String(64), primary_key=True),
    Column("digest", String(64), nullable=False),
    Column("payload", Text, nullable=False),
)
runs = Table(
    "finserve_runs",
    schema,
    Column("id", String(128), primary_key=True),
    Column("model_id", String(64), ForeignKey(models.c.id), nullable=False),
    Column("revision_id", String(128), ForeignKey(revisions.c.id), nullable=True),
    Column("workload_id", String(64), ForeignKey(definitions.c.id), nullable=False),
    Column("digest", String(64), nullable=False),
    Column("payload", Text, nullable=False),
)
decisions = Table(
    "finserve_decisions",
    schema,
    Column("id", String(64), primary_key=True),
    Column("revision_id", String(128), ForeignKey(revisions.c.id), nullable=False),
    Column("candidate_run_id", String(128), ForeignKey(runs.c.id), nullable=False),
    Column("digest", String(64), nullable=False),
    Column("payload", Text, nullable=False),
)
jobs = Table(
    "finserve_lifecycle",
    schema,
    Column("id", String(128), primary_key=True),
    Column("spec_digest", String(64), nullable=False),
    Column("spec", Text, nullable=False),
    Column("payload", Text, nullable=False),
    Column("version", Integer, nullable=False),
    Column("owner", String(128), nullable=True),
    Column[float]("lease_until", Float, nullable=False),
    Column("locked_deployment", String(128), nullable=True, unique=True),
)
events = Table(
    "finserve_lifecycle_events",
    schema,
    Column("id", Integer, primary_key=True),
    Column("job_id", String(128), ForeignKey(jobs.c.id), nullable=False),
    Column[float]("observed_at", Float, nullable=False),
    Column("payload", Text, nullable=False),
)


class RegistryConflict(RuntimeError):
    """Immutable identity collision, concurrent ownership or a stale state transition."""


class RunBundle(ImmutableModel):
    """A run's immutable artifact references are separate from its frozen workload definition."""

    run_id: str = Field(min_length=1, max_length=128)
    workload_hash: str
    model_identity: str
    revision_id: str | None
    manifest: ArtifactRef
    requests: ArtifactRef
    summary: ArtifactRef


LifecycleStatus = Literal[
    "registered",
    "evaluated",
    "rejected",
    "deploying",
    "verifying",
    "promoted",
    "needs_reconciliation",
]


class LifecycleState(ImmutableModel):
    """Persist stage and action ambiguity separately from Airflow task retry state."""

    job_id: str
    status: LifecycleStatus = "registered"
    version: int = Field(default=0, ge=0)
    decision_digest: str | None = None
    last_error: str | None = None
    apply_attempts: int = Field(default=0, ge=0)


def canonical_json(value: object) -> str:
    """Canonical finite JSON identities are independent of dictionary insertion order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_text(value: str) -> str:
    """Use content identity for definitions, decisions and lifecycle input specifications."""
    return hashlib.sha256(value.encode()).hexdigest()


def validate_stage(before: LifecycleState, after: LifecycleState) -> None:
    """Keep illegal skips out of durable state even if an orchestration caller has a bug."""
    allowed: dict[LifecycleStatus, set[LifecycleStatus]] = {
        "registered": {"registered", "evaluated", "rejected"},
        "evaluated": {"evaluated", "deploying", "rejected"},
        "deploying": {"deploying", "verifying", "promoted", "needs_reconciliation"},
        "verifying": {"verifying", "promoted"},
        "needs_reconciliation": {"needs_reconciliation", "deploying", "promoted"},
        "promoted": set(),
        "rejected": set(),
    }
    if after.job_id != before.job_id:
        raise RegistryConflict("lifecycle job identity is immutable")
    if after.status not in allowed[before.status]:
        raise RegistryConflict("illegal lifecycle transition")
    if after.status not in {"registered", "rejected"} and after.decision_digest is None:
        raise RegistryConflict("evaluated and active states require a recorded decision")


def sqlite_constraints(connection: sqlite3.Connection, _: ConnectionPoolEntry) -> None:
    """SQLite must enforce the same foreign-key relationships expected from PostgreSQL."""
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")


class Registry:
    """SQLite is locally exercised; PostgreSQL uses the same tables and optimistic concurrency
    rules.
    """

    def __init__(self, database_url: str) -> None:
        """Create schema without exposing URL credentials in SQL errors or logs."""
        url = make_url(database_url)
        if url.get_backend_name() == "sqlite" and url.database not in {None, "", ":memory:"}:
            database = Path(str(url.database)).resolve()
            repository = Path(__file__).resolve().parents[3]
            if database == repository or repository in database.parents:
                raise ValueError("registry truth must be outside source repository")
            database.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(url, hide_parameters=True)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", sqlite_constraints)
        schema.create_all(self.engine)

    def close(self) -> None:
        """Release pooled connections when an offline workflow or test finishes."""
        self.engine.dispose()

    def _immutable(self, table: Table, identity: str, payload: str, **columns: object) -> None:
        """Unique keys arbitrate inserts; identical retries succeed and mutations fail."""
        digest = digest_text(payload)
        try:
            with self.engine.begin() as connection:
                found = connection.execute(
                    select(table.c.digest).where(table.c.id == identity)
                ).scalar_one_or_none()
                if found is None:
                    connection.execute(
                        insert(table).values(id=identity, digest=digest, payload=payload, **columns)
                    )
                elif found != digest:
                    raise RegistryConflict("immutable registry identity reused")
        except IntegrityError:
            with self.engine.connect() as connection:
                found = connection.execute(
                    select(table.c.digest).where(table.c.id == identity)
                ).scalar_one_or_none()
                if found != digest:
                    raise RegistryConflict(
                        "concurrent identity conflict or missing related record"
                    ) from None

    def _payload(self, table: Table, identity: str) -> str:
        """All lookups use bound identities; missing evidence is never returned as a passing
        default.
        """
        with self.engine.connect() as connection:
            value = connection.execute(
                select(table.c.payload).where(table.c.id == identity)
            ).scalar_one_or_none()
        if value is None:
            raise KeyError("registry record not found")
        return str(value)

    def register_model(self, model_revision: str, tokenizer_revision: str) -> str:
        """Record declared model/tokenizer identity without claiming weights were uploaded or
        verified.
        """
        if not model_revision or not tokenizer_revision:
            raise ValueError("model and tokenizer identity required")
        payload = canonical_json(
            {"model_revision": model_revision, "tokenizer_revision": tokenizer_revision}
        )
        identity = digest_text(payload)
        self._immutable(models, identity, payload)
        return identity

    def register_revision(self, revision: Revision) -> None:
        """Link immutable deployment revisions to model identities through foreign keys."""
        model_id = self.register_model(revision.model_revision, revision.tokenizer_revision)
        self._immutable(
            revisions,
            revision.revision_id,
            canonical_json(revision.model_dump()),
            model_id=model_id,
        )

    def revision(self, identity: str) -> Revision:
        """Resolve the exact target associated with a registered candidate run."""
        return Revision.model_validate_json(self._payload(revisions, identity))

    def register_run(
        self, directory: Path, store: ArtifactStore, revision: Revision | None = None
    ) -> RunBundle:
        """Validate before registration; host observations may omit deployment images."""
        before = {
            name: (directory / name).read_bytes()
            for name in ("manifest.json", "requests.jsonl", "summary.json")
        }
        manifest = validate_evidence(directory)
        if revision is not None:
            verify_candidate_identity(manifest, revision)
            self.register_revision(revision)
        if any((directory / name).read_bytes() != value for name, value in before.items()):
            raise ValueError("run artifacts changed during registration")
        model_id = self.register_model(
            manifest.configuration.model_revision, manifest.configuration.tokenizer_revision
        )
        self._immutable(
            definitions, manifest.workload_hash, canonical_json(manifest.workload.model_dump())
        )
        identity: dict[str, object] = json.loads(before["manifest.json"])
        if not isinstance(identity.get("run_id"), str):
            raise ValueError("run ID must be a nonempty string")
        bundle = RunBundle(
            run_id=str(identity["run_id"]),
            workload_hash=manifest.workload_hash,
            model_identity=model_id,
            revision_id=revision.revision_id if revision else None,
            manifest=store.put(before["manifest.json"]),
            requests=store.put(before["requests.jsonl"]),
            summary=store.put(before["summary.json"]),
        )
        self._immutable(
            runs,
            bundle.run_id,
            canonical_json(bundle.model_dump()),
            model_id=model_id,
            revision_id=bundle.revision_id,
            workload_id=bundle.workload_hash,
        )
        return bundle

    def run(self, run_id: str) -> RunBundle:
        """Retrieve registered references without bypassing verification in the artifact store."""
        return RunBundle.model_validate_json(self._payload(runs, run_id))

    def record_decision(self, decision: PromotionDecision, candidate_run_id: str) -> str:
        """Preserve rejections as well as approvals under content hashes and a revision foreign
        key.
        """
        payload = canonical_json(decision.model_dump())
        identity = digest_text(canonical_json([candidate_run_id, decision.model_dump()]))
        if self.run(candidate_run_id).revision_id != decision.candidate_revision:
            raise ValueError("decision candidate run must bind the exact deployment revision")
        self._immutable(
            decisions,
            identity,
            payload,
            revision_id=decision.candidate_revision,
            candidate_run_id=candidate_run_id,
        )
        self.verify_decision_run(decision, candidate_run_id)
        return identity

    def verify_decision_run(self, decision: PromotionDecision, candidate_run_id: str) -> None:
        """Only the gate recorder establishes which run a mirrored decision actually evaluated."""
        identity = digest_text(canonical_json([candidate_run_id, decision.model_dump()]))
        with self.engine.connect() as connection:
            recorded = connection.execute(
                select(decisions.c.candidate_run_id).where(decisions.c.id == identity)
            ).scalar_one_or_none()
        if recorded != candidate_run_id:
            raise ValueError("decision is not registered for this candidate run")

    def decision(self, identity: str) -> PromotionDecision:
        """A lifecycle consumes a server-generated recorded decision, not a client passed flag."""
        return PromotionDecision.model_validate_json(self._payload(decisions, identity))

    def create_job(self, job_id: str, specification: str) -> LifecycleState:
        """The same job ID cannot be retried with a different image, suite, policy or run bundle."""
        state = LifecycleState(job_id=job_id)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(jobs).values(
                        id=job_id,
                        spec_digest=digest_text(specification),
                        spec=specification,
                        payload=state.model_dump_json(),
                        version=0,
                        lease_until=0,
                    )
                )
        except IntegrityError:
            with self.engine.connect() as connection:
                existing = connection.execute(
                    select(jobs.c.spec_digest).where(jobs.c.id == job_id)
                ).scalar_one()
            if existing != digest_text(specification):
                raise RegistryConflict(
                    "lifecycle job identity reused with changed specification"
                ) from None
        return self.job(job_id)

    def job(self, job_id: str) -> LifecycleState:
        """Read durable lifecycle state after task retries or controller restarts."""
        return LifecycleState.model_validate_json(self._payload(jobs, job_id))

    def specification(self, job_id: str) -> str:
        """Tasks exchange only job IDs; immutable specifications stay in durable registry truth."""
        with self.engine.connect() as connection:
            value = connection.execute(select(jobs.c.spec).where(jobs.c.id == job_id)).scalar_one()
        return str(value)

    def history(self, job_id: str) -> list[LifecycleState]:
        """Expose persisted stage observations in transaction order for audits and retries."""
        with self.engine.connect() as connection:
            values = connection.execute(
                select(events.c.payload).where(events.c.job_id == job_id).order_by(events.c.id)
            ).scalars()
            return [LifecycleState.model_validate_json(str(value)) for value in values]

    def claim(self, job_id: str, deployment_id: str, owner: str, seconds: float) -> LifecycleState:
        """Claim a job lease and deployment lock; ambiguity keeps the lock until reconciled."""
        if not math.isfinite(seconds) or seconds <= 0 or not owner or not deployment_id:
            raise ValueError("finite positive lease and nonempty identities required")
        now = time.time()
        try:
            with self.engine.begin() as connection:
                result = connection.execute(
                    update(jobs)
                    .where(
                        jobs.c.id == job_id, or_(jobs.c.owner.is_(None), jobs.c.lease_until < now)
                    )
                    .values(owner=owner, lease_until=now + seconds, locked_deployment=deployment_id)
                )
                if result.rowcount != 1:
                    raise RegistryConflict("lifecycle worker lease is active or job missing")
        except IntegrityError:
            raise RegistryConflict("another lifecycle owns this deployment") from None
        return self.job(job_id)

    def advance(self, state: LifecycleState, owner: str, **changes: object) -> LifecycleState:
        """Check owner, lease and version together and append the transition in the same
        transaction.
        """
        with self.engine.begin() as connection:
            payload = connection.execute(
                select(jobs.c.payload).where(jobs.c.id == state.job_id)
            ).scalar_one_or_none()
            if payload is None or LifecycleState.model_validate_json(str(payload)) != state:
                raise RegistryConflict("caller state differs from durable lifecycle truth")
            updated = LifecycleState.model_validate(
                {**state.model_dump(), **changes, "version": state.version + 1}
            )
            validate_stage(state, updated)
            result = connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == state.job_id,
                    jobs.c.version == state.version,
                    jobs.c.owner == owner,
                    jobs.c.lease_until > time.time(),
                )
                .values(payload=updated.model_dump_json(), version=updated.version)
            )
            if result.rowcount != 1:
                raise RegistryConflict("stale lifecycle transition or expired worker")
            connection.execute(
                insert(events).values(
                    job_id=state.job_id, observed_at=time.time(), payload=updated.model_dump_json()
                )
            )
        return updated

    def release(self, job_id: str, owner: str) -> None:
        """Release worker ownership; deployment ownership persists after ambiguous external
        actions.
        """
        state = self.job(job_id)
        values: dict[str, object] = {"owner": None, "lease_until": 0}
        if state.status in {"registered", "evaluated", "rejected", "promoted"}:
            values["locked_deployment"] = None
        with self.engine.begin() as connection:
            connection.execute(
                update(jobs).where(jobs.c.id == job_id, jobs.c.owner == owner).values(**values)
            )
