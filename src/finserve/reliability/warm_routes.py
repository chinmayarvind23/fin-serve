"""Durable local traffic cutover between already-running immutable backend bindings."""

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Generator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Literal, Self
from uuid import uuid4

import httpx
from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from finserve.contracts.deployment import (
    DeploymentState,
    HealthObservation,
    ImmutableModel,
    Revision,
)
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.engines.openai_adapter import CompletionState, sse_events
from finserve.http_ownership import HTTPClosureError, own_response
from finserve.registry.model_assets import owned_disk
from finserve.reliability.promotion import PromotionDecision
from finserve.reliability.rollback import (
    ApplyRequest,
    ControlConflict,
    DeploymentStore,
    load_record,
)
from finserve.reliability.store_identity import initialize_identity, open_existing, verify_identity
from finserve.reliability.warm_drain import (
    ADMISSION_PROTOCOL,
    AdmissionLease,
    WarmDrainReceipt,
    initialize_admissions,
    read_admission_protocol,
)


class BackendConfiguration(ImmutableModel):
    """The endpoint and credential reference belong to trusted deployment configuration."""

    base_url: str
    model: str = Field(min_length=1, max_length=256)
    credential_env: str | None = Field(default=None, pattern=r"^[A-Z_][A-Z0-9_]{0,127}$")

    @model_validator(mode="after")
    def fixed_destination(self) -> Self:
        """Reject embedded secrets and ambiguous endpoints before storing immutable identity."""
        address = httpx.URL(self.base_url)
        if (
            address.scheme not in {"http", "https"}
            or not address.host
            or address.username
            or address.password
            or address.query
            or address.fragment
            or address.path != "/v1"
            or str(address) != self.base_url
        ):
            raise ValueError("backend must be a canonical HTTP(S) /v1 URL without credentials")
        return self

    def digest(self) -> str:
        """Bind configuration identity to endpoint, model and credential reference."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class WarmBackend(ImmutableModel):
    """A revision's configuration digest must identify this exact already-running endpoint."""

    revision: Revision
    configuration: BackendConfiguration
    serving_profile: ServingProfileV1 | None = None

    @model_validator(mode="after")
    def configuration_matches(self) -> Self:
        """A caller cannot label an unrelated endpoint with an otherwise valid revision."""
        if self.serving_profile is not None:
            self.serving_profile.verify_revision(self.revision)
            if (
                self.serving_profile.base_url != self.configuration.base_url
                or self.serving_profile.served_model != self.configuration.model
                or self.serving_profile.credential_env != self.configuration.credential_env
            ):
                raise ValueError("warm endpoint differs from canonical serving profile")
        elif self.revision.config_digest != self.configuration.digest():
            raise ValueError("revision configuration digest does not bind the backend")
        return self


class RouteSnapshot(ImmutableModel):
    """A request pins one route generation for its entire stream, including after cutover."""

    deployment_id: str = Field(min_length=1, max_length=128)
    revision_id: str
    revision_digest: str
    generation: int = Field(ge=0, strict=True)
    changed_at: float = Field(gt=0)
    admission_protocol: Literal["durable-http-close-v1"] | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep legacy bytes; new payloads reject old extra-forbid readers."""
        result: dict[str, Any] = handler(self)
        if self.admission_protocol is None:
            result.pop("admission_protocol", None)
        return result


class WarmRouteStore:
    """External traffic truth is separate from orchestration intent but owns the final CAS fence."""

    admission_protocol: Literal["durable-http-close-v1"] | None

    def __init__(self, path: Path, *, expected_identity: str | None = None) -> None:
        """Keep SQLite WAL/FULL state outside source, with short transactions around cutovers."""
        self.path = path.resolve()
        repository = Path(__file__).resolve().parents[3]
        if self.path == repository or repository in self.path.parents:
            raise ValueError("routing truth must be outside the source repository")
        new_store = not self.path.exists()
        if expected_identity is not None:
            self.identity = expected_identity
            with self.transaction() as connection:
                self.admission_protocol = read_admission_protocol(connection)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS warm_backends(id TEXT PRIMARY KEY,
                    base_url TEXT NOT NULL UNIQUE,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS warm_routes(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS warm_actions(id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS warm_events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS warm_retirements(id TEXT PRIMARY KEY,
                    digest TEXT NOT NULL);
            """)
            self.identity = initialize_identity(connection)
            initialize_admissions(connection, new_store=new_store)
            self.admission_protocol = read_admission_protocol(connection)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection]:
        """BEGIN IMMEDIATE commits route selection and its action receipt atomically."""
        connection = open_existing(self.path)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            verify_identity(connection, self.identity)
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def register(self, backend: WarmBackend) -> None:
        """Bound registered endpoints and forbid revision reuse with changed configuration."""
        with self.transaction() as connection:
            self._require_available(connection, backend.revision.revision_id)
            found: tuple[str] | None = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?", (backend.revision.revision_id,)
            ).fetchone()
            if found is not None:
                if WarmBackend.model_validate_json(found[0]) != backend:
                    raise ControlConflict("backend revision is immutable")
                return
            count: tuple[int] = connection.execute(
                "SELECT COUNT(*) FROM warm_backends "
                "WHERE id NOT IN (SELECT id FROM warm_retirements)"
            ).fetchone()
            if count[0] >= 32:
                raise ControlConflict("warm backend registry limit reached")
            try:
                connection.execute(
                    "INSERT INTO warm_backends VALUES(?,?,?)",
                    (
                        backend.revision.revision_id,
                        backend.configuration.base_url,
                        backend.model_dump_json(),
                    ),
                )
            except sqlite3.IntegrityError:
                raise ControlConflict("each warm revision requires a distinct endpoint") from None

    def backend(self, revision_id: str) -> WarmBackend:
        """Resolve only a pre-registered endpoint; request payloads cannot supply destinations."""
        with self.transaction() as connection:
            self._require_available(connection, revision_id)
            row: tuple[str] | None = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?", (revision_id,)
            ).fetchone()
        if row is None:
            raise KeyError("warm backend not registered")
        return WarmBackend.model_validate_json(row[0])

    def is_retired(self, revision_id: str) -> bool:
        """Expose irreversible retirement for local idle connection-pool reclamation."""
        with self.transaction() as connection:
            return connection.execute(
                "SELECT 1 FROM warm_retirements WHERE id=?", (revision_id,)
            ).fetchone() is not None

    def _require_available(self, connection: sqlite3.Connection, revision_id: str) -> None:
        """A retired producer revision cannot become traffic after cleanup has been authorized."""
        if connection.execute(
            "SELECT 1 FROM warm_retirements WHERE id=?", (revision_id,)
        ).fetchone():
            raise ControlConflict("backend revision is retired")

    def admit(self, deployment_id: str) -> AdmissionLease:
        """Select route and persist its stream obligation in one retirement-fenced transaction."""
        with self.transaction() as connection:
            route = self._snapshot(connection, deployment_id)
            if (
                read_admission_protocol(connection) != ADMISSION_PROTOCOL
                or route.admission_protocol != ADMISSION_PROTOCOL
            ):
                raise ControlConflict("legacy route cannot use durable admission")
            self._require_available(connection, route.revision_id)
            row = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?", (route.revision_id,)
            ).fetchone()
            if row is None:
                raise ControlConflict("admission backend is missing")
            backend = WarmBackend.model_validate_json(row[0])
            if backend.revision.digest() != route.revision_digest:
                raise ControlConflict("admission backend identity changed")
            return self._insert_admission(connection, route, backend)

    def _insert_admission(
        self, connection: sqlite3.Connection, route: RouteSnapshot, backend: WarmBackend
    ) -> AdmissionLease:
        """Gateway and collector obligations share the same atomic retirement fence and cap."""
        if connection.execute("SELECT COUNT(*) FROM warm_admissions").fetchone()[0] >= 4096:
            raise ControlConflict("durable admission limit reached")
        lease = AdmissionLease(uuid4().hex, route, backend)
        connection.execute(
            "INSERT INTO warm_admissions VALUES(?,?,?)",
            (
                lease.admission_id,
                route.revision_id,
                json.dumps(
                    {
                        "route": route.model_dump(mode="json"),
                        "backend": backend.model_dump(mode="json"),
                    },
                    sort_keys=True,
                ),
            ),
        )
        return lease

    def finish_admission(self, lease: AdmissionLease) -> None:
        """Only an exact locally closed response releases its durable stream obligation."""
        if not lease.verified_closed:
            raise ControlConflict("backend transport closure is unresolved")
        expected = json.dumps(
            {
                "route": lease.snapshot.model_dump(mode="json"),
                "backend": lease.backend.model_dump(mode="json"),
            },
            sort_keys=True,
        )
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT payload FROM warm_admissions WHERE admission_id=?", (lease.admission_id,)
            ).fetchone()
            if row is None or row[0] != expected:
                raise ControlConflict("admission identity changed or already released")
            connection.execute(
                "DELETE FROM warm_admissions WHERE admission_id=?", (lease.admission_id,)
            )

    def retire_drained(self, control: DeploymentStore, backend: WarmBackend) -> bool:
        """Permanently fence an inactive revision and prove no durable stream obligations remain."""
        if self.path == control.path:
            raise ValueError("route and control stores require separate database files")
        revision = backend.revision
        with self.transaction() as connection, control.transaction() as controller:
            if read_admission_protocol(connection) != ADMISSION_PROTOCOL:
                return False
            for (payload,) in connection.execute("SELECT payload FROM warm_routes"):
                if RouteSnapshot.model_validate_json(payload).revision_id == revision.revision_id:
                    return False
            for (payload,) in controller.execute("SELECT payload FROM deployments"):
                state = DeploymentState.model_validate_json(payload)
                if revision.revision_id in {state.active_revision, state.known_good_revision}:
                    return False
            row = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?", (revision.revision_id,)
            ).fetchone()
            if row is None or WarmBackend.model_validate_json(row[0]) != backend:
                raise ControlConflict("drain backend identity changed")
            for (payload,) in connection.execute("SELECT payload FROM warm_events"):
                if (
                    RouteSnapshot.model_validate_json(payload).admission_protocol
                    != ADMISSION_PROTOCOL
                ):
                    return False
            prior = connection.execute(
                "SELECT digest FROM warm_retirements WHERE id=?", (revision.revision_id,)
            ).fetchone()
            if prior is not None and prior[0] != revision.digest():
                raise ControlConflict("drain retirement identity changed")
            connection.execute(
                "INSERT OR IGNORE INTO warm_retirements VALUES(?,?)",
                (revision.revision_id, revision.digest()),
            )
            if connection.execute(
                "SELECT 1 FROM warm_admissions WHERE revision_id=? LIMIT 1", (revision.revision_id,)
            ).fetchone():
                return False
            receipt = WarmDrainReceipt(
                store_identity=self.identity,
                revision_id=revision.revision_id,
                revision_digest=revision.digest(),
                observed_at=time.time(),
            )
            connection.execute(
                "INSERT OR IGNORE INTO warm_drain_receipts VALUES(?,?)",
                (revision.revision_id, receipt.model_dump_json()),
            )
            return True

    def drain_pending(self, backend: WarmBackend) -> bool:
        """A retired revision without terminal drain evidence must not count as settled cleanup."""
        with self.transaction() as connection:
            if read_admission_protocol(connection) != ADMISSION_PROTOCOL:
                return False
            return (
                connection.execute(
                    "SELECT 1 FROM warm_retirements r WHERE r.id=? AND r.digest=? "
                    "AND NOT EXISTS (SELECT 1 FROM warm_drain_receipts d WHERE d.revision_id=r.id)",
                    (backend.revision.revision_id, backend.revision.digest()),
                ).fetchone()
                is not None
            )

    def retire_unserved(self, control: DeploymentStore, backend: WarmBackend) -> bool:
        """Fence never-served revisions before stopping their exact producer runtime.

        Historical routes remain protected because pinned in-flight requests have no durable
        drain receipt. Route->control lock order prevents activation racing this decision.
        """
        if self.path == control.path:
            raise ValueError("route and control stores require separate database files")
        revision = backend.revision
        with self.transaction() as connection, control.transaction() as controller:
            for (payload,) in connection.execute("SELECT payload FROM warm_events"):
                if RouteSnapshot.model_validate_json(payload).revision_id == revision.revision_id:
                    return False
            for (payload,) in controller.execute("SELECT payload FROM deployments"):
                state = DeploymentState.model_validate_json(payload)
                if revision.revision_id in {state.active_revision, state.known_good_revision}:
                    return False
            registered: tuple[str] | None = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?", (revision.revision_id,)
            ).fetchone()
            if registered is not None and WarmBackend.model_validate_json(registered[0]) != backend:
                raise ControlConflict("retirement differs from registered backend")
            if registered is None:
                try:
                    connection.execute(
                        "INSERT INTO warm_backends VALUES(?,?,?)",
                        (
                            revision.revision_id,
                            backend.configuration.base_url,
                            backend.model_dump_json(),
                        ),
                    )
                except sqlite3.IntegrityError:
                    raise ControlConflict(
                        "cleanup endpoint is reserved by another revision"
                    ) from None
            prior: tuple[str] | None = connection.execute(
                "SELECT digest FROM warm_retirements WHERE id=?", (revision.revision_id,)
            ).fetchone()
            if prior is not None and prior[0] != revision.digest():
                raise ControlConflict("retirement revision identity changed")
            connection.execute(
                "INSERT OR IGNORE INTO warm_retirements VALUES(?,?)",
                (revision.revision_id, revision.digest()),
            )
            return True

    def release_retired_endpoint(self, backend: WarmBackend) -> None:
        """After exact stop verification, release only this retired revision's endpoint slot."""
        with self.transaction() as connection:
            row: tuple[str, str] | None = connection.execute(
                "SELECT b.payload,r.digest FROM warm_backends b JOIN warm_retirements r "
                "ON b.id=r.id WHERE b.id=?",
                (backend.revision.revision_id,),
            ).fetchone()
            if (
                row is None
                or WarmBackend.model_validate_json(row[0]) != backend
                or row[1] != backend.revision.digest()
            ):
                raise ControlConflict("endpoint release differs from retired backend")
            connection.execute(
                "UPDATE warm_backends SET base_url=? WHERE id=?",
                ("retired:" + backend.revision.revision_id, backend.revision.revision_id),
            )

    def snapshot(self, deployment_id: str) -> RouteSnapshot:
        """Read one coherent active route; historical revision identity never comes from a cache."""
        with self.transaction() as connection:
            return self._snapshot(connection, deployment_id)

    def _snapshot(self, connection: sqlite3.Connection, deployment_id: str) -> RouteSnapshot:
        """Use the caller's transaction while fencing a switch or bootstrapping traffic."""
        row: tuple[str] | None = connection.execute(
            "SELECT payload FROM warm_routes WHERE id=?", (deployment_id,)
        ).fetchone()
        if row is None:
            raise KeyError("warm deployment not initialized")
        return RouteSnapshot.model_validate_json(row[0])

    def _save(self, connection: sqlite3.Connection, state: RouteSnapshot) -> None:
        """Append route history in the same commit that changes future request selection."""
        connection.execute(
            "INSERT INTO warm_routes VALUES(?,?) "
            "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
            (state.deployment_id, state.model_dump_json()),
        )
        connection.execute(
            "INSERT INTO warm_events(deployment_id,payload) VALUES(?,?)",
            (state.deployment_id, state.model_dump_json()),
        )

    def bootstrap(self, deployment_id: str, revision_id: str) -> RouteSnapshot:
        """Select the initial route; real traffic must subsequently verify readiness."""
        backend = self.backend(revision_id)
        with self.transaction() as connection:
            self._require_available(connection, revision_id)
            try:
                existing = self._snapshot(connection, deployment_id)
            except KeyError:
                existing = None
            if existing is not None:
                if existing.generation != 0 or existing.revision_id != revision_id:
                    raise ControlConflict("route already initialized differently")
                return existing
            state = RouteSnapshot(
                deployment_id=deployment_id,
                revision_id=revision_id,
                revision_digest=backend.revision.digest(),
                generation=0,
                changed_at=time.time(),
                admission_protocol=self.admission_protocol,
            )
            self._save(connection, state)
            return state

    def require_stable_baseline(
        self,
        control: DeploymentStore,
        deployment_id: str,
        backend: WarmBackend,
        generation: int,
        *,
        reserve: bool = False,
    ) -> AdmissionLease | None:
        """Optionally reserve a borrowed runtime in the same transaction as stable validation."""
        if self.path == control.path:
            raise ValueError("route and control stores require separate database files")
        revision = backend.revision
        with self.transaction() as connection, control.transaction() as controller:
            self._require_available(connection, revision.revision_id)
            route = self._snapshot(connection, deployment_id)
            state = load_record(controller, "deployments", deployment_id, DeploymentState)
            registered = load_record(controller, "revisions", revision.revision_id, Revision)
            row: tuple[str] | None = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?",
                (revision.revision_id,),
            ).fetchone()
            if (
                row is None
                or WarmBackend.model_validate_json(row[0]) != backend
                or registered != revision
                or route.revision_digest != revision.digest()
                or route.revision_id != revision.revision_id
                or state.active_revision != revision.revision_id
                or state.known_good_revision != revision.revision_id
                or state.generation != generation
                or route.generation != generation
                or state.rollback_id is not None
            ):
                raise ControlConflict("borrowed baseline is not the current stable deployment")
            if reserve:
                if (
                    read_admission_protocol(connection) != ADMISSION_PROTOCOL
                    or route.admission_protocol != ADMISSION_PROTOCOL
                ):
                    raise ControlConflict("legacy store cannot reserve a collector")
                return self._insert_admission(connection, route, backend)
            return None

    def acknowledge_baseline(
        self,
        control: DeploymentStore,
        deployment_id: str,
        revision: Revision,
        generation: int,
        health: HealthObservation,
    ) -> DeploymentState:
        """Fence current baseline traffic while verifying or initializing controller truth."""
        if control.path == self.path:
            raise ValueError("route and control stores require separate database files")
        with self.transaction() as connection:
            route = self._snapshot(connection, deployment_id)
            if (
                route.revision_id != revision.revision_id
                or route.revision_digest != revision.digest()
                or route.generation != generation
                or not health.verifies(revision)
            ):
                raise ControlConflict("baseline acknowledgment differs from current traffic")
            control.register_revision(revision)
            try:
                state = control.deployment(deployment_id)
            except KeyError:
                if generation != 0:
                    raise ControlConflict("missing controller for noninitial route") from None
                state = control.bootstrap(deployment_id, revision.revision_id, health)
            if (
                state.generation != generation
                or state.active_revision != revision.revision_id
                or state.known_good_revision != revision.revision_id
                or state.rollback_id is not None
            ):
                raise ControlConflict("baseline is not the stable controller revision")
            return state

    def apply(self, request: ApplyRequest) -> RouteSnapshot:
        """The shared route CAS fences promotion and rollback from separate intent stores."""
        backend = self.backend(request.target.revision_id)
        if backend.revision != request.target or not request.idempotency_key:
            raise ControlConflict("unregistered target identity or empty action key")
        fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        with self.transaction() as connection:
            self._require_available(connection, request.target.revision_id)
            receipt: tuple[str, str] | None = connection.execute(
                "SELECT fingerprint,payload FROM warm_actions WHERE id=?",
                (request.idempotency_key,),
            ).fetchone()
            if receipt is not None:
                if receipt[0] != fingerprint:
                    raise ControlConflict("idempotency key reused for another action")
                return RouteSnapshot.model_validate_json(receipt[1])
            current = self._snapshot(connection, request.deployment_id)
            if (
                current.revision_id != request.expected_revision
                or current.generation != request.expected_generation
            ):
                raise ControlConflict("stale external route generation")
            current_backend: tuple[str] = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?", (current.revision_id,)
            ).fetchone()
            if (
                WarmBackend.model_validate_json(current_backend[0]).configuration.model
                != backend.configuration.model
            ):
                raise ControlConflict("cutover cannot change the public model alias")
            updated = RouteSnapshot(
                deployment_id=current.deployment_id,
                revision_id=backend.revision.revision_id,
                revision_digest=backend.revision.digest(),
                generation=current.generation + 1,
                changed_at=time.time(),
                admission_protocol=self.admission_protocol,
            )
            self._save(connection, updated)
            connection.execute(
                "INSERT INTO warm_actions VALUES(?,?,?)",
                (request.idempotency_key, fingerprint, updated.model_dump_json()),
            )
            return updated

    def _verified_action(
        self,
        connection: sqlite3.Connection,
        request: ApplyRequest,
        decision: PromotionDecision,
        health: HealthObservation,
    ) -> RouteSnapshot:
        """Compare the action receipt and current route while its write lock is held."""
        fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        action: tuple[str, str] | None = connection.execute(
            "SELECT fingerprint,payload FROM warm_actions WHERE id=?",
            (request.idempotency_key,),
        ).fetchone()
        route = self._snapshot(connection, request.deployment_id)
        if (
            action is None
            or action[0] != fingerprint
            or RouteSnapshot.model_validate_json(action[1]) != route
            or route.revision_id != request.target.revision_id
            or route.revision_digest != request.target.digest()
            or route.generation != request.expected_generation + 1
            or decision.candidate_revision != request.target.revision_id
            or decision.candidate_digest != request.target.digest()
            or not health.verifies(request.target)
        ):
            raise ControlConflict("activation acknowledgment differs from current route action")
        return route

    def acknowledge_candidate(
        self,
        control: DeploymentStore,
        request: ApplyRequest,
        decision: PromotionDecision,
        health: HealthObservation,
    ) -> DeploymentState:
        """Fence route writers while acknowledging the exact persisted lifecycle action.

        Lock order is route then control. No network work runs under either lock. A crash
        after control commit replays its activation receipt without advancing generation again.
        """
        if control.path == self.path:
            raise ValueError("route and control stores require separate database files")
        with self.transaction() as connection:
            route = self._verified_action(connection, request, decision, health)
            current = control.deployment(request.deployment_id)
            if current.rollback_id is not None or (
                (current.generation, current.active_revision)
                not in {
                    (request.expected_generation, request.expected_revision),
                    (route.generation, route.revision_id),
                }
            ):
                raise ControlConflict(
                    "activation acknowledgment differs from controller generation"
                )
            updated = control.activate_candidate(
                request.deployment_id, request.expected_generation, decision, health
            )
            if (
                updated.generation != route.generation
                or updated.active_revision != route.revision_id
            ):
                raise ControlConflict("activation acknowledgment no longer names current traffic")
            return updated

    def stabilize_candidate(
        self,
        control: DeploymentStore,
        request: ApplyRequest,
        decision: PromotionDecision,
        health: HealthObservation,
        evidence_digest: str,
    ) -> DeploymentState:
        """Keep the route generation fixed while committing verified probation evidence."""
        if control.path == self.path:
            raise ValueError("route and control stores require separate database files")
        with self.transaction() as connection:
            route = self._verified_action(connection, request, decision, health)
            return control.mark_stable(
                request.deployment_id, route.generation, decision, health, evidence_digest
            )

    def history(self, deployment_id: str) -> list[RouteSnapshot]:
        """Keep every cutover, including later restoration to an older revision."""
        with self.transaction() as connection:
            rows: list[tuple[str]] = connection.execute(
                "SELECT payload FROM warm_events WHERE deployment_id=? ORDER BY sequence",
                (deployment_id,),
            ).fetchall()
        return [RouteSnapshot.model_validate_json(row[0]) for row in rows]


class WarmRouteAdapter:
    """Only a bounded successful smoke through the active inference URL establishes health."""

    supports_idempotency = True

    def __init__(
        self,
        store: WarmRouteStore,
        traffic_url: str,
        client: httpx.AsyncClient,
        timeout_seconds: float = 5,
    ) -> None:
        """The caller owns authenticated HTTP client lifetime; secrets remain in memory."""
        if not 0 < timeout_seconds <= 30:
            raise ValueError("finite health timeout in (0,30] required")
        address = httpx.URL(traffic_url)
        if (
            address.scheme not in {"http", "https"}
            or not address.host
            or address.username
            or address.password
            or address.query
            or address.fragment
        ):
            raise ValueError("invalid trusted traffic URL")
        self.store, self.traffic_url, self.client = store, traffic_url.rstrip("/"), client
        self.timeout_seconds = timeout_seconds

    async def apply(self, request: ApplyRequest) -> None:
        """Offload SQLite work; cancellation can be reconciled with the same action key."""
        await owned_disk(lambda: self.store.apply(request))

    async def health(self, deployment_id: str) -> HealthObservation:
        """One deadline covers route reads, HTTP smoke and the final generation fence."""
        async with asyncio.timeout(self.timeout_seconds):
            return await self._health(deployment_id)

    async def _health(self, deployment_id: str) -> HealthObservation:
        """Reject failed SSE and route races before claiming that configured traffic is healthy."""
        before = await owned_disk(lambda: self.store.snapshot(deployment_id))
        backend = await owned_disk(lambda: self.store.backend(before.revision_id))
        valid = False
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with self.client.stream(
                    "POST",
                    self.traffic_url + "/v1/completions",
                    json={
                        "model": backend.configuration.model,
                        "prompt": "health",
                        "max_tokens": 1,
                        "temperature": 0,
                        "stream": True,
                        "timeout_seconds": self.timeout_seconds,
                    },
                    follow_redirects=False,
                ) as response:
                    own_response(response)
                    media_type = response.headers.get("content-type", "").split(";", 1)[0]
                    if (
                        response.status_code != 200
                        or media_type.strip().lower() != "text/event-stream"
                        or response.headers.get("content-encoding", "identity").lower()
                        != "identity"
                    ):
                        raise ValueError("health response must be an uncompressed successful SSE")
                    state = CompletionState(maximum_tokens=1)
                    done = False
                    async for event in sse_events(response, maximum_bytes=65536):
                        if event == "[DONE]":
                            state.final_token()
                            done = True
                            break
                        state.consume(event)
                    valid = (
                        state.visible_text
                        and done
                        and state.completion_tokens == 1
                        and response.headers.get("x-finserve-revision") == before.revision_id
                        and response.headers.get("x-finserve-revision-digest")
                        == before.revision_digest
                        and response.headers.get("x-finserve-route-generation")
                        == str(before.generation)
                    )
            valid = valid and await owned_disk(lambda: self.store.snapshot(deployment_id)) == before
        except HTTPClosureError:
            raise
        except Exception:
            valid = False
        return HealthObservation(
            revision_id=before.revision_id,
            revision_digest=before.revision_digest,
            ready=valid,
            smoke_passed=valid,
        )
