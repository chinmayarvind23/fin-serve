"""SQLite pool selection and global slot ownership share the route retirement transaction."""

import hashlib
import json
import sqlite3
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from finserve.contracts.capacity import CAPACITY_PROTOCOL
from finserve.contracts.inference import InferenceRequest
from finserve.reliability.rollback import ControlConflict
from finserve.reliability.warm_drain import AdmissionLease

if TYPE_CHECKING:
    from finserve.reliability.warm_routes import RouteSnapshot, WarmRouteStore


def initialize_capacity(connection: sqlite3.Connection, *, new_store: bool, enabled: bool) -> None:
    """Opt in before any old gateway history; immutable payload fields reject old readers."""
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS warm_capacity_protocol(protocol TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS warm_capacity_plans(
            id TEXT PRIMARY KEY,payload TEXT NOT NULL,state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS warm_capacity_authority(
            deployment_id TEXT PRIMARY KEY,plan_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS warm_capacity_generations(
            deployment_id TEXT PRIMARY KEY,generation INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS warm_capacity_history(
            deployment_id TEXT NOT NULL,generation INTEGER NOT NULL,reason TEXT NOT NULL,
            PRIMARY KEY(deployment_id,generation));
        CREATE TABLE IF NOT EXISTS warm_capacity_slots(
            deployment_id TEXT PRIMARY KEY,plan_id TEXT NOT NULL,cycle INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS warm_capacity_members(
            revision_id TEXT PRIMARY KEY,deployment_id TEXT NOT NULL,plan_id TEXT NOT NULL,
            anchor_revision TEXT NOT NULL,anchor_generation INTEGER NOT NULL,
            ready INTEGER NOT NULL,receipt TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS warm_capacity_probes(
            id TEXT PRIMARY KEY,deployment_id TEXT NOT NULL,
            route TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS warm_capacity_events(
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,plan_id TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS warm_capacity_demand(
            deployment_id TEXT PRIMARY KEY,rejected INTEGER NOT NULL);
    """)
    connection.execute("BEGIN IMMEDIATE")
    try:
        current = read_capacity_protocol(connection)
        if enabled and current is None:
            if not new_store or connection.execute("SELECT 1 FROM warm_events LIMIT 1").fetchone():
                raise ControlConflict("capacity requires a new store; legacy enrollment rejected")
            connection.execute("INSERT INTO warm_capacity_protocol VALUES(?)", (CAPACITY_PROTOCOL,))
        if read_capacity_protocol(connection):
            for table in ("warm_routes", "warm_events", "warm_actions"):
                for action in ("INSERT", "UPDATE"):
                    connection.execute(f"""
                        CREATE TRIGGER IF NOT EXISTS capacity_{table}_{action}
                        BEFORE {action} ON {table}
                        WHEN json_extract(NEW.payload,'$.capacity_protocol')
                             IS NOT '{CAPACITY_PROTOCOL}'
                        BEGIN SELECT RAISE(ABORT,'legacy capacity protocol rejected'); END
                    """)
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def read_capacity_protocol(connection: sqlite3.Connection) -> Literal["local-capacity-v1"] | None:
    """Existing non-capacity stores stay byte-compatible and cannot be silently upgraded."""
    if not connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='warm_capacity_protocol'"
    ).fetchone():
        return None
    rows = connection.execute("SELECT protocol FROM warm_capacity_protocol").fetchall()
    if not rows:
        return None
    if rows != [(CAPACITY_PROTOCOL,)]:
        raise ControlConflict("unsupported capacity protocol")
    return CAPACITY_PROTOCOL


def advance_pool(connection: sqlite3.Connection, deployment_id: str, reason: str) -> None:
    """Every eligibility change has a durable identity, including removal and route cutover."""
    connection.execute(
        "INSERT INTO warm_capacity_generations VALUES(?,1) ON CONFLICT(deployment_id) "
        "DO UPDATE SET generation=generation+1",
        (deployment_id,),
    )
    generation = connection.execute(
        "SELECT generation FROM warm_capacity_generations WHERE deployment_id=?", (deployment_id,)
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO warm_capacity_history VALUES(?,?,?)", (deployment_id, generation, reason)
    )


def remove_members(connection: sqlite3.Connection, plan_id: str, deployment_id: str) -> None:
    """Publish removal and its new pool generation in the same admission transaction."""
    changed = connection.execute(
        "UPDATE warm_capacity_members SET ready=0 WHERE plan_id=? AND ready=1", (plan_id,)
    ).rowcount
    if changed:
        advance_pool(connection, deployment_id, "remove:" + plan_id)


def retirement_blocked(connection: sqlite3.Connection, revision_id: str, *, unserved: bool) -> bool:
    """A selected pool member is served history even though it is not the logical route."""
    if read_capacity_protocol(connection) is None:
        return False
    member = connection.execute(
        "SELECT m.ready,m.anchor_revision,m.anchor_generation,r.payload "
        "FROM warm_capacity_members m LEFT JOIN warm_routes r ON r.id=m.deployment_id "
        "WHERE m.revision_id=?",
        (revision_id,),
    ).fetchone()
    if member is None:
        return False
    if unserved:
        return True
    current: dict[str, object] = json.loads(member[3]) if member[3] else {}
    return bool(
        member[0]
        and current.get("revision_id") == member[1]
        and current.get("generation") == member[2]
    )


def probe_token(store: "WarmRouteStore", route: "RouteSnapshot", request: InferenceRequest) -> str:
    """Mint bounded one-use authority for a normal authenticated primary-only HTTP smoke."""
    token = uuid4().hex + uuid4().hex
    with store.transaction() as connection:
        if read_capacity_protocol(connection) != CAPACITY_PROTOCOL:
            raise ControlConflict("primary probe requires capacity protocol")
        if connection.execute("SELECT COUNT(*) FROM warm_capacity_probes").fetchone()[0] >= 64:
            raise ControlConflict("primary probe token limit reached")
        connection.execute(
            "INSERT INTO warm_capacity_probes VALUES(?,?,?,?)",
            (
                hashlib.sha256(token.encode()).hexdigest(),
                route.deployment_id,
                route.model_dump_json(),
                hashlib.sha256(request.model_dump_json().encode()).hexdigest(),
            ),
        )
    return token


def revoke_probe(store: "WarmRouteStore", token: str) -> None:
    """Revocation removes unused probe authority, never any live stream obligation."""
    with store.transaction() as connection:
        connection.execute(
            "DELETE FROM warm_capacity_probes WHERE id=?",
            (hashlib.sha256(token.encode()).hexdigest(),),
        )


def select_admission(
    store: "WarmRouteStore",
    expected: "RouteSnapshot",
    request: InferenceRequest,
    token: str | None,
) -> AdmissionLease:
    """After auth/app admission, atomically choose a physical member and reserve its budget."""
    from finserve.contracts.capacity import CapacityPlan
    from finserve.reliability.warm_routes import WarmBackend

    rejected = False
    lease = None
    with store.transaction() as connection:
        route = store._snapshot(connection, expected.deployment_id)  # pyright: ignore[reportPrivateUsage]
        if route != expected or route.capacity_protocol != CAPACITY_PROTOCOL:
            raise ControlConflict("capacity route changed before dispatch")
        kind: Literal["serving", "collector", "probe"] = "serving"
        probe_id = None
        if token is not None:
            probe_id = hashlib.sha256(token.encode()).hexdigest()
            probe = connection.execute(
                "SELECT deployment_id,route,payload FROM warm_capacity_probes WHERE id=?",
                (probe_id,),
            ).fetchone()
            if probe != (
                route.deployment_id,
                route.model_dump_json(),
                hashlib.sha256(request.model_dump_json().encode()).hexdigest(),
            ):
                raise ControlConflict("primary probe authority is missing or substituted")
            kind = "probe"
        candidates = [route.revision_id]
        authority = connection.execute(
            "SELECT p.payload FROM warm_capacity_authority a "
            "JOIN warm_capacity_plans p ON p.id=a.plan_id WHERE a.deployment_id=?",
            (route.deployment_id,),
        ).fetchone()
        limit = 128
        version = connection.execute(
            "SELECT generation FROM warm_capacity_generations WHERE deployment_id=?",
            (route.deployment_id,),
        ).fetchone()
        generation = version[0] if version else 0
        if authority:
            plan = CapacityPlan.model_validate_json(json.loads(authority[0])["plan"])
            if (plan.primary.revision.revision_id, plan.expected_generation) == (
                route.revision_id,
                route.generation,
            ):
                limit = plan.per_member_requests
                if kind == "serving":
                    candidates.extend(
                        row[0]
                        for row in connection.execute(
                            "SELECT revision_id FROM warm_capacity_members WHERE deployment_id=? "
                            "AND anchor_revision=? AND anchor_generation=? AND ready=1",
                            (route.deployment_id, route.revision_id, route.generation),
                        )
                    )
        loads: list[tuple[int, str]] = []
        for revision_id in candidates:
            store._require_available(connection, revision_id)  # pyright: ignore[reportPrivateUsage]
            count = connection.execute(
                "SELECT COUNT(*) FROM warm_admissions WHERE revision_id=?", (revision_id,)
            ).fetchone()[0]
            if count < limit:
                loads.append((count, revision_id))
        if not loads:
            rejected = True
            if kind == "serving":
                connection.execute(
                    "INSERT INTO warm_capacity_demand VALUES(?,1) ON CONFLICT(deployment_id) "
                    "DO UPDATE SET rejected=rejected+1",
                    (route.deployment_id,),
                )
        else:
            revision_id = min(loads)[1]
            backend = WarmBackend.model_validate_json(
                connection.execute(
                    "SELECT payload FROM warm_backends WHERE id=?", (revision_id,)
                ).fetchone()[0]
            )
            snapshot = route.model_copy(
                update={"revision_id": revision_id, "revision_digest": backend.revision.digest()}
            )
            lease = store._insert_admission(  # pyright: ignore[reportPrivateUsage]
                connection, snapshot, backend, kind=kind, anchor=route, pool_generation=generation
            )
            if probe_id is not None:
                connection.execute("DELETE FROM warm_capacity_probes WHERE id=?", (probe_id,))
    if rejected or lease is None:
        raise ControlConflict("capacity serving admission limit reached")
    return lease
