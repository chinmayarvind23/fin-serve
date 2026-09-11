"""Durable local traffic cutover between already-running immutable backend bindings."""

import asyncio
import hashlib
import sqlite3
import time
from collections.abc import Generator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Self

import httpx
from pydantic import Field, model_validator

from finserve.contracts.deployment import HealthObservation, ImmutableModel, Revision
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.engines.openai_adapter import CompletionState, sse_events
from finserve.reliability.rollback import ApplyRequest, ControlConflict


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


class WarmRouteStore:
    """External traffic truth is separate from orchestration intent but owns the final CAS fence."""

    def __init__(self, path: Path) -> None:
        """Keep SQLite WAL/FULL state outside source, with short transactions around cutovers."""
        self.path = path.resolve()
        repository = Path(__file__).resolve().parents[3]
        if self.path == repository or repository in self.path.parents:
            raise ValueError("routing truth must be outside the source repository")
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
            """)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection]:
        """BEGIN IMMEDIATE commits route selection and its action receipt atomically."""
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

    def register(self, backend: WarmBackend) -> None:
        """Bound registered endpoints and forbid revision reuse with changed configuration."""
        with self.transaction() as connection:
            found: tuple[str] | None = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?", (backend.revision.revision_id,)
            ).fetchone()
            if found is not None:
                if WarmBackend.model_validate_json(found[0]) != backend:
                    raise ControlConflict("backend revision is immutable")
                return
            count: tuple[int] = connection.execute("SELECT COUNT(*) FROM warm_backends").fetchone()
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
            row: tuple[str] | None = connection.execute(
                "SELECT payload FROM warm_backends WHERE id=?", (revision_id,)
            ).fetchone()
        if row is None:
            raise KeyError("warm backend not registered")
        return WarmBackend.model_validate_json(row[0])

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
            )
            self._save(connection, state)
            return state

    def apply(self, request: ApplyRequest) -> RouteSnapshot:
        """The shared route CAS fences promotion and rollback from separate intent stores."""
        backend = self.backend(request.target.revision_id)
        if backend.revision != request.target or not request.idempotency_key:
            raise ControlConflict("unregistered target identity or empty action key")
        fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        with self.transaction() as connection:
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
            )
            self._save(connection, updated)
            connection.execute(
                "INSERT INTO warm_actions VALUES(?,?,?)",
                (request.idempotency_key, fingerprint, updated.model_dump_json()),
            )
            return updated

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
        await asyncio.to_thread(self.store.apply, request)

    async def health(self, deployment_id: str) -> HealthObservation:
        """One deadline covers route reads, HTTP smoke and the final generation fence."""
        async with asyncio.timeout(self.timeout_seconds):
            return await self._health(deployment_id)

    async def _health(self, deployment_id: str) -> HealthObservation:
        """Reject failed SSE and route races before claiming that configured traffic is healthy."""
        before = await asyncio.to_thread(self.store.snapshot, deployment_id)
        backend = await asyncio.to_thread(self.store.backend, before.revision_id)
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
            valid = valid and await asyncio.to_thread(self.store.snapshot, deployment_id) == before
        except Exception:
            valid = False
        return HealthObservation(
            revision_id=before.revision_id,
            revision_digest=before.revision_digest,
            ready=valid,
            smoke_passed=valid,
        )
