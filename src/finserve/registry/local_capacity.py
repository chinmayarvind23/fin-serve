"""Local demand drives receipt-bound model launches and durable pool retirement."""

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from uuid import uuid4

import httpx

from finserve.contracts.capacity import CAPACITY_PROTOCOL, CapacityPlan, CapacityState
from finserve.contracts.deployment import DeploymentState
from finserve.contracts.managed_runtime import RuntimeLaunchSpec
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.registry.managed_runtime import DockerRuntime, owned_directory
from finserve.registry.model_assets import owned_disk
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import declare_input
from finserve.registry.release_activation import approved_activation
from finserve.registry.release_gate import GateRequest
from finserve.registry.runtime_fence import attempt_fence
from finserve.registry.runtime_stages import (
    abort_runtime_stage,
    launch_runtime_stage,
    load_launch,
    stop_runtime_stage,
    verify_upstream,
)
from finserve.reliability.capacity_policy import (
    CapacityMemory,
    CapacityObservation,
    decide_capacity,
)
from finserve.reliability.capacity_store import advance_pool, remove_members
from finserve.reliability.rollback import ControlConflict, DeploymentStore, load_record
from finserve.reliability.warm_routes import BackendConfiguration, WarmBackend, WarmRouteStore


def runtime_backend(spec: RuntimeLaunchSpec) -> WarmBackend:
    """Keep the full physical profile; a replica endpoint never inherits an anchor digest."""
    return WarmBackend(
        revision=spec.revision,
        serving_profile=spec.profile,
        configuration=BackendConfiguration(
            base_url=spec.profile.base_url, model=spec.profile.served_model
        ),
    )


def stable_anchor(
    connection: sqlite3.Connection,
    controller: sqlite3.Connection,
    routes: WarmRouteStore,
    plan: CapacityPlan,
) -> bool:
    """Route then control transactions prevent promotion from racing enrollment/publication."""
    route = routes._snapshot(connection, plan.deployment_id)  # pyright: ignore[reportPrivateUsage]
    control = load_record(controller, "deployments", plan.deployment_id, DeploymentState)
    return (
        route.capacity_protocol == CAPACITY_PROTOCOL
        and route.revision_id == plan.primary.revision.revision_id
        and route.revision_digest == plan.primary.revision.digest()
        and route.generation == plan.expected_generation
        and control.active_revision == control.known_good_revision == route.revision_id
        and control.generation == route.generation
        and control.rollback_id is None
    )


def freeze_capacity(
    journal: ProducerStages,
    routes: WarmRouteStore,
    control: DeploymentStore,
    plan: CapacityPlan,
) -> CapacityState:
    """Enroll only a canonically approved, promoted stable release and immutable model assets."""
    plan = CapacityPlan.model_validate_json(plan.model_dump_json())
    if routes.capacity_protocol != CAPACITY_PROTOCOL or routes.path == control.path:
        raise ControlConflict("capacity requires a new protocol store and separate control truth")
    request, decision = approved_activation(
        journal.registry, journal.artifacts, plan.approval_job_id
    )
    gate = GateRequest.model_validate_json(journal.registry.gate_input(plan.approval_job_id))
    profile = ServingProfileV1.model_validate_json(journal.artifacts.get(gate.candidate_profile))
    if (
        request.target != plan.primary.revision
        or profile != plan.primary.profile
        or request.deployment_id != plan.deployment_id
        or request.expected_generation + 1 != plan.expected_generation
    ):
        raise ControlConflict("capacity primary differs from approved canonical activation")
    primary_state = journal.state(plan.primary_launch_stage)
    load_launch(journal, primary_state, plan.primary)
    model, image = verify_upstream(journal, plan.model_stage_id, plan.build_stage_id, plan.primary)
    owned_directory(plan.workspace, create=True)
    envelope = json.dumps(
        {
            "plan": plan.model_dump_json(),
            "routes": routes.identity,
            "control": control.identity,
            "primary": primary_state.output.model_dump() if primary_state.output else None,
            "model": model.model_dump(),
            "image": image.model_dump(),
            "decision": hashlib.sha256(decision.model_dump_json().encode()).hexdigest(),
        },
        sort_keys=True,
    )
    # The journal independently makes this plan immutable across route-store replacement.
    declare_input(journal, plan.plan_id + ":capacity-input", json.loads(envelope))
    state = CapacityState(plan_id=plan.plan_id)
    with routes.transaction() as connection, control.transaction() as controller:
        if not stable_anchor(connection, controller, routes, plan):
            raise ControlConflict("capacity enrollment requires the exact stable primary")
        previous = connection.execute(
            "SELECT payload,state FROM warm_capacity_plans WHERE id=?", (plan.plan_id,)
        ).fetchone()
        if previous is not None:
            if previous[0] != envelope:
                raise ControlConflict("capacity input is immutable")
            state = CapacityState.model_validate_json(previous[1])
        authority = connection.execute(
            "SELECT plan_id FROM warm_capacity_authority WHERE deployment_id=?",
            (plan.deployment_id,),
        ).fetchone()
        if authority and authority[0] != plan.plan_id:
            prior = CapacityState.model_validate_json(
                connection.execute(
                    "SELECT state FROM warm_capacity_plans WHERE id=?", (authority[0],)
                ).fetchone()[0]
            )
            if (
                prior.phase != "closed"
                or connection.execute(
                    "SELECT 1 FROM warm_capacity_slots WHERE deployment_id=?", (plan.deployment_id,)
                ).fetchone()
            ):
                raise ControlConflict("another capacity plan owns the deployment slot")
        connection.execute(
            "INSERT OR IGNORE INTO warm_capacity_plans VALUES(?,?,?)",
            (plan.plan_id, envelope, state.model_dump_json()),
        )
        if authority != (plan.plan_id,):
            advance_pool(connection, plan.deployment_id, "enroll:" + plan.plan_id)
        connection.execute(
            "INSERT INTO warm_capacity_authority VALUES(?,?) "
            "ON CONFLICT(deployment_id) DO UPDATE SET plan_id=excluded.plan_id",
            (plan.deployment_id, plan.plan_id),
        )
    return state


class CapacityController:
    """A global store/deployment authority plus process fence bounds every actual allocation."""

    def __init__(
        self,
        journal: ProducerStages,
        routes: WarmRouteStore,
        control: DeploymentStore,
        plan: CapacityPlan,
        runtime: DockerRuntime,
        client: httpx.AsyncClient,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Production supplies DockerRuntime; tests replace only its actual command boundary."""
        self.journal, self.routes, self.control = journal, routes, control
        self.plan, self.runtime, self.client, self.clock = plan, runtime, client, clock
        self.epoch = uuid4().hex
        self.state()

    def _state(self, connection: sqlite3.Connection) -> CapacityState:
        """Every operation checks immutable input and the cross-plan deployment authority."""
        row = connection.execute(
            "SELECT payload,state FROM warm_capacity_plans WHERE id=?", (self.plan.plan_id,)
        ).fetchone()
        authority = connection.execute(
            "SELECT plan_id FROM warm_capacity_authority WHERE deployment_id=?",
            (self.plan.deployment_id,),
        ).fetchone()
        if row is None or authority != (self.plan.plan_id,):
            raise ControlConflict("capacity plan does not own deployment authority")
        frozen = json.loads(row[0])
        declared = self.journal.state(self.plan.plan_id + ":capacity-input")
        if json.loads(self.journal.artifacts.get(declared.input)) != frozen:
            raise ControlConflict("capacity journal differs from enrolled input")
        if (
            frozen["plan"] != self.plan.model_dump_json()
            or frozen["routes"] != self.routes.identity
            or frozen["control"] != self.control.identity
        ):
            raise ControlConflict("capacity executor input or store identity changed")
        return CapacityState.model_validate_json(row[1])

    def state(self) -> CapacityState:
        """Expose durable progress without inferring absence from daemon or process age."""
        with self.routes.transaction() as connection:
            return self._state(connection)

    def _save(self, connection: sqlite3.Connection, state: CapacityState, detail: object) -> None:
        """Phase and its audit event commit together under the controller's process fence."""
        self._state(connection)
        connection.execute(
            "UPDATE warm_capacity_plans SET state=? WHERE id=?",
            (state.model_dump_json(), self.plan.plan_id),
        )
        connection.execute(
            "INSERT INTO warm_capacity_events(plan_id,payload) VALUES(?,?)",
            (
                self.plan.plan_id,
                json.dumps({"state": state.model_dump(), "detail": detail}, sort_keys=True),
            ),
        )

    def _observation(
        self, connection: sqlite3.Connection, state: CapacityState
    ) -> CapacityObservation:
        """Count positive HTTP dispatch; unresolved reservations prevent removal."""
        revisions = [self.plan.primary.revision.revision_id]
        revisions.extend(
            row[0]
            for row in connection.execute(
                "SELECT revision_id FROM warm_capacity_members WHERE plan_id=? AND ready=1",
                (self.plan.plan_id,),
            )
        )
        active, unresolved = 0, False
        for revision in revisions:
            for (payload,) in connection.execute(
                "SELECT payload FROM warm_admissions WHERE revision_id=?", (revision,)
            ):
                value = json.loads(payload)
                if value.get("kind") == "serving" and value.get("phase") == "dispatched":
                    active += 1
                elif value.get("kind") == "collector" or value.get("phase") != "dispatched":
                    unresolved = True
        rejected = connection.execute(
            "SELECT rejected FROM warm_capacity_demand WHERE deployment_id=?",
            (self.plan.deployment_id,),
        ).fetchone()
        return CapacityObservation(
            sequence=state.samples,
            epoch=self.epoch,
            observed_seconds=self.clock(),
            serving_active=active,
            serving_capacity=len(revisions) * self.plan.per_member_requests,
            members=2 if len(revisions) == 2 else 1,
            unresolved=unresolved,
            rejected_total=rejected[0] if rejected else 0,
        )

    def observation(self) -> CapacityObservation:
        """Read actual tagged gateway state; callers cannot inject a desired load sample."""
        with self.routes.transaction() as connection:
            return self._observation(connection, self._state(connection))

    def _advance(self) -> CapacityState:
        """Reserve the sole global slot before a launch can issue any daemon command."""
        with self.routes.transaction() as connection, self.control.transaction() as controller:
            state = self._state(connection)
            if state.phase == "closed":
                return state
            stable = stable_anchor(connection, controller, self.routes, self.plan)
            if not stable or state.samples >= self.plan.max_samples:
                if state.phase == "one":
                    state = state.model_copy(update={"phase": "closed"})
                else:
                    state = state.model_copy(update={"phase": "draining"})
                    remove_members(connection, self.plan.plan_id, self.plan.deployment_id)
                self._save(connection, state, "anchor_changed_or_sample_budget_exhausted")
                return state
            if state.phase in {"warming", "draining", "blocked"}:
                return state
            observation = self._observation(connection, state)
            memory, action = decide_capacity(
                self.plan.policy, observation, CapacityMemory.model_validate_json(state.memory)
            )
            state = state.model_copy(
                update={"memory": memory.model_dump_json(), "samples": state.samples + 1}
            )
            if action == "up" and state.phase == "one":
                if state.cycle >= len(self.plan.replicas):
                    state = state.model_copy(update={"phase": "closed"})
                else:
                    # This unique deployment key is stronger than a per-plan process lock.
                    connection.execute(
                        "INSERT INTO warm_capacity_slots VALUES(?,?,?)",
                        (self.plan.deployment_id, self.plan.plan_id, state.cycle),
                    )
                    state = state.model_copy(update={"phase": "warming"})
            elif action == "down" and state.phase == "two":
                remove_members(connection, self.plan.plan_id, self.plan.deployment_id)
                state = state.model_copy(update={"phase": "draining"})
            self._save(
                connection, state, {"observation": observation.model_dump(), "action": action}
            )
            return state

    def _publish_ready(self, state: CapacityState, receipt: str) -> bool:
        """A receipt alone cannot publish traffic after promotion invalidates the anchor."""
        spec = self.plan.replicas[state.cycle]
        with self.routes.transaction() as connection, self.control.transaction() as controller:
            current = self._state(connection)
            if current != state or connection.execute(
                "SELECT plan_id,cycle FROM warm_capacity_slots WHERE deployment_id=?",
                (self.plan.deployment_id,),
            ).fetchone() != (self.plan.plan_id, state.cycle):
                raise ControlConflict("capacity warming slot changed")
            if not stable_anchor(connection, controller, self.routes, self.plan):
                self._save(
                    connection,
                    state.model_copy(update={"phase": "draining"}),
                    "anchor_changed_while_warming",
                )
                return False
            self.routes._require_available(connection, spec.revision.revision_id)  # pyright: ignore[reportPrivateUsage]
            connection.execute(
                "INSERT OR REPLACE INTO warm_capacity_members VALUES(?,?,?,?,?,?,?)",
                (
                    spec.revision.revision_id,
                    self.plan.deployment_id,
                    self.plan.plan_id,
                    self.plan.primary.revision.revision_id,
                    self.plan.expected_generation,
                    1,
                    receipt,
                ),
            )
            advance_pool(connection, self.plan.deployment_id, "ready:" + receipt)
            self._save(
                connection,
                state.model_copy(update={"phase": "two", "error": None}),
                "exact_runtime_ready",
            )
        return True

    async def _launch(self, state: CapacityState) -> None:
        """Drive the existing executor and publish only its actual observed container start."""
        spec = self.plan.replicas[state.cycle]
        primary = await owned_disk(
            lambda: load_launch(
                self.journal, self.journal.state(self.plan.primary_launch_stage), self.plan.primary
            )
        )
        await self.runtime.observe(self.plan.primary, primary, self.client)
        await owned_disk(lambda: self.routes.register(runtime_backend(spec)))
        receipt = await launch_runtime_stage(
            self.journal,
            f"{self.plan.plan_id}:replica-{state.cycle}-launch",
            self.plan.model_stage_id,
            self.plan.build_stage_id,
            spec,
            self.plan.workspace,
            self.client,
            self.runtime,
        )
        await owned_disk(lambda: self._publish_ready(state, receipt.model_dump_json()))

    def _blocked(self, error: str) -> None:
        """Failure retains the global allocation and removes eligibility before reconciliation."""
        with self.routes.transaction() as connection:
            state = self._state(connection)
            remove_members(connection, self.plan.plan_id, self.plan.deployment_id)
            self._save(
                connection,
                state.model_copy(update={"phase": "blocked", "error": error}),
                "owned_operation_unresolved",
            )

    def _settled(self, state: CapacityState) -> None:
        """Only terminal lifecycle cleanup releases the global slot and allows port reuse."""
        with self.routes.transaction() as connection, self.control.transaction() as controller:
            current = self._state(connection)
            if current.cycle != state.cycle:
                raise ControlConflict("capacity cleanup cycle changed")
            connection.execute(
                "DELETE FROM warm_capacity_slots WHERE deployment_id=? AND plan_id=? AND cycle=?",
                (self.plan.deployment_id, self.plan.plan_id, state.cycle),
            )
            cycle = state.cycle + 1
            closed = (
                cycle >= len(self.plan.replicas)
                or current.samples >= self.plan.max_samples
                or not stable_anchor(connection, controller, self.routes, self.plan)
            )
            self._save(
                connection,
                current.model_copy(
                    update={
                        "phase": "closed" if closed else "one",
                        "cycle": cycle,
                        "memory": CapacityMemory().model_dump_json(),
                    }
                ),
                "exact_owned_cleanup_complete",
            )

    async def _cleanup(self, state: CapacityState) -> None:
        """Unpublished attempts use abort; published starts wait for a durable zero-row drain."""
        spec = self.plan.replicas[state.cycle]
        backend = runtime_backend(spec)
        launch_id = f"{self.plan.plan_id}:replica-{state.cycle}-launch"
        try:
            launch = await owned_disk(lambda: self.journal.state(launch_id))
        except KeyError:
            launch = None
        if launch is not None and launch.status == "completed":
            if not await owned_disk(lambda: self.routes.retire_drained(self.control, backend)):
                return
            await stop_runtime_stage(
                self.journal,
                f"{self.plan.plan_id}:replica-{state.cycle}-stop",
                launch_id,
                spec,
                self.runtime,
            )
        elif launch is not None and launch.attempt_id is not None:
            if not await owned_disk(lambda: self.routes.retire_unserved(self.control, backend)):
                raise ControlConflict("incomplete capacity runtime gained traffic ownership")
            await abort_runtime_stage(
                self.journal,
                f"{self.plan.plan_id}:replica-{state.cycle}-abort",
                launch_id,
                launch.attempt_id,
                self.runtime,
            )
        else:
            # No attempt exists: this protocol issues no daemon call before the journal attempt.
            await owned_disk(lambda: self.routes.retire_unserved(self.control, backend))
        await owned_disk(lambda: self.routes.release_retired_endpoint(backend))
        await owned_disk(lambda: self._settled(state))

    async def close(self) -> CapacityState:
        """Stop new extra admissions at runner exit and reconcile only owned replica work."""

        def retire() -> None:
            """Persist shutdown before any wait; crashes keep the slot and obligations owned."""
            with self.routes.transaction() as connection:
                state = self._state(connection)
                if state.phase == "closed":
                    return
                phase = "closed" if state.phase == "one" else "draining"
                remove_members(connection, self.plan.plan_id, self.plan.deployment_id)
                self._save(
                    connection,
                    state.model_copy(
                        update={
                            "phase": phase,
                            "samples": self.plan.max_samples,
                        }
                    ),
                    "capacity_runner_shutdown",
                )

        directory = await owned_disk(
            lambda: owned_directory(
                self.plan.workspace
                / (
                    "capacity-"
                    + self.routes.identity
                    + "-"
                    + hashlib.sha256(self.plan.deployment_id.encode()).hexdigest()[:16]
                ),
                create=True,
            )
        )
        async with attempt_fence(directory):
            await owned_disk(retire)
            return await self.tick()

    async def tick(self) -> CapacityState:
        """Reconcile one durable sample/action; cancellation drains owned I/O before unlock."""
        directory = await owned_disk(
            lambda: owned_directory(
                self.plan.workspace
                / (
                    "capacity-"
                    + self.routes.identity
                    + "-"
                    + hashlib.sha256(self.plan.deployment_id.encode()).hexdigest()[:16]
                ),
                create=True,
            )
        )
        async with attempt_fence(directory):
            state = await owned_disk(self._advance)
            try:
                if state.phase == "warming":
                    await self._launch(state)
                    state = await owned_disk(self.state)
                if state.phase in {"draining", "blocked"}:
                    await self._cleanup(state)
            except asyncio.CancelledError:
                await owned_disk(lambda: self._blocked("CancelledError"))
                raise
            except Exception as error:
                error_code = type(error).__name__
                await owned_disk(lambda: self._blocked(error_code))
            return await owned_disk(self.state)
