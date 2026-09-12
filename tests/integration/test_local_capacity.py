"""Actual HTTP capacity cycles use canonical approval and production lifecycle executors."""

import asyncio
import os
import socket
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from capacity_approval_fixture import approve_primary, declare_approval
from capacity_http_fixture import CapacityDaemon, CapacityHTTPBackend
from test_managed_runtime import specification, upstream
from test_warm_rollback import live_server

from finserve.benchmark.runner import RunConfig, request_one
from finserve.benchmark.workload import WorkItem
from finserve.contracts.capacity import CapacityPlan
from finserve.contracts.inference import InferenceRequest
from finserve.contracts.managed_runtime import RuntimeLaunchSpec
from finserve.gateway.warm_route_app import create_warm_app
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.local_capacity import CapacityController, freeze_capacity
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.runtime_stages import launch_runtime_stage, stop_runtime_stage
from finserve.reliability.capacity_policy import CapacityPolicy
from finserve.reliability.capacity_store import select_admission
from finserve.reliability.rollback import ApplyRequest, ControlConflict, DeploymentStore
from finserve.reliability.warm_routes import (
    BackendConfiguration,
    WarmBackend,
    WarmRouteAdapter,
    WarmRouteStore,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="runtime ownership uses POSIX fences")


def unused_port() -> int:
    """Reserve ephemeral loopback identities without touching a configured production port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def relocate(spec: RuntimeLaunchSpec, name: str, port: int) -> RuntimeLaunchSpec:
    """Recompute canonical profile identity for each explicitly synthetic physical endpoint."""
    profile = spec.profile.model_copy(update={"base_url": f"http://127.0.0.1:{port}/v1"})
    revision = spec.revision.model_copy(
        update={"revision_id": name, "config_digest": profile.digest()}
    )
    return RuntimeLaunchSpec.model_validate(
        spec.model_copy(update={"profile": profile, "revision": revision}).model_dump()
    )


@dataclass
class CapacityFixture:
    """Keep every fixture resource and attribution available for lifecycle assertions."""

    plan: CapacityPlan
    journal: ProducerStages
    routes: WarmRouteStore
    control: DeploymentStore
    primary: CapacityHTTPBackend
    extra: CapacityHTTPBackend
    daemon: CapacityDaemon
    client: httpx.AsyncClient
    url: str
    adapter: WarmRouteAdapter


@asynccontextmanager
async def capacity_fixture(root: Path) -> AsyncGenerator[CapacityFixture]:
    """Launch a real primary listener, earn canonical stability, and borrow it for capacity."""
    registry = Registry("sqlite:///" + str(root / "registry.db"))
    artifacts = LocalArtifactStore(root / "objects")
    journal = ProducerStages(registry, artifacts)
    original = upstream(journal, specification(root), root)
    baseline, primary, extra = (
        relocate(original, name, unused_port()) for name in ("baseline", "primary", "extra")
    )
    previous = CapacityHTTPBackend("baseline", int(httpx.URL(baseline.profile.base_url).port or 0))
    serving = CapacityHTTPBackend("primary", int(httpx.URL(primary.profile.base_url).port or 0))
    replica = CapacityHTTPBackend("extra", int(httpx.URL(extra.profile.base_url).port or 0))
    primary_daemon = CapacityDaemon(primary, serving)
    primary_runtime = DockerRuntime(primary_daemon)
    daemon = CapacityDaemon(extra, replica)
    daemon.borrowed = primary_daemon
    routes = WarmRouteStore(root / "routes.sqlite", capacity_enabled=True)
    control = DeploymentStore(root / "control.sqlite")
    approval = declare_approval(root, registry, artifacts, baseline, primary)
    for launch in (baseline, primary):
        routes.register(
            WarmBackend(
                revision=launch.revision,
                serving_profile=launch.profile,
                configuration=BackendConfiguration(
                    base_url=launch.profile.base_url, model=launch.profile.served_model
                ),
            )
        )
        control.register_revision(launch.revision)
    routes.bootstrap(approval.deployment_id, baseline.revision.revision_id)
    await previous.start()
    try:
        async with httpx.AsyncClient(
            timeout=5, trust_env=False, headers={"authorization": "Bearer synthetic-capacity-key"}
        ) as client:
            await launch_runtime_stage(
                journal,
                "job:primary",
                "job:model",
                "job:build",
                primary,
                root / "runtime",
                client,
                primary_runtime,
            )
            async with live_server(
                create_warm_app(routes, approval.deployment_id, api_key="synthetic-capacity-key")
            ) as url:
                adapter = WarmRouteAdapter(routes, url, client)
                await approve_primary(registry, artifacts, approval, control, adapter)
                plan = CapacityPlan(
                    plan_id="http-capacity",
                    deployment_id=approval.deployment_id,
                    expected_generation=1,
                    approval_job_id=approval.job_id,
                    primary_launch_stage="job:primary",
                    model_stage_id="job:model",
                    build_stage_id="job:build",
                    primary=primary,
                    replicas=(extra,),
                    workspace=root / "runtime",
                    per_member_requests=2,
                    policy=CapacityPolicy(
                        sample_seconds=1,
                        high_load=0.5,
                        low_load=0.3,
                        high_samples=1,
                        low_samples=1,
                        cooldown_seconds=0,
                    ),
                )
                yield CapacityFixture(
                    plan, journal, routes, control, serving, replica, daemon, client, url, adapter
                )
            await stop_runtime_stage(
                journal, "job:primary-stop", "job:primary", primary, primary_runtime
            )
    finally:
        for backend in (previous, serving, replica):
            for hold in backend.holds.values():
                hold.set()
            await backend.stop()
        registry.close()


async def completion(fixture: CapacityFixture, prompt: str) -> httpx.Response:
    """Consume actual gateway SSE so dispatch and transport release retain production semantics."""
    return await fixture.client.post(
        fixture.url + "/v1/completions",
        json={"model": "fixture", "prompt": prompt, "max_tokens": 1, "stream": True},
    )


async def entered(backend: CapacityHTTPBackend, prompt: str) -> None:
    """Bound synchronization by the actual backend observing the named dispatched request."""
    await asyncio.wait_for(backend.entered[prompt].wait(), 5)


async def occupancy(controller: CapacityController, expected: int) -> None:
    """Wait for real transport dispatch/closure persistence after the backend emits its event."""
    async with asyncio.timeout(3):
        # Persistence deliberately has no test-only event hook.
        while controller.observation().serving_active != expected:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_http_load_launches_and_drained_replica_stops(tmp_path: Path) -> None:
    """Actual held work causes scale-up; low occupancy never permits stopping a held replica."""
    async with capacity_fixture(tmp_path) as fixture:
        now = [100.0]
        freeze_capacity(fixture.journal, fixture.routes, fixture.control, fixture.plan)
        controller = CapacityController(
            fixture.journal,
            fixture.routes,
            fixture.control,
            fixture.plan,
            DockerRuntime(fixture.daemon),
            fixture.client,
            clock=lambda: now[0],
        )
        fixture.primary.hold("primary-held")
        first = asyncio.create_task(completion(fixture, "primary-held"))
        await entered(fixture.primary, "primary-held")
        try:
            await occupancy(controller, 1)
            assert controller.observation().serving_active == 1
            assert (await controller.tick()).phase == "one"
            now[0] += 1
            expanded = await controller.tick()
            assert expanded.phase == "two", (expanded, fixture.daemon.calls)
            launch_stage = fixture.plan.plan_id + ":replica-0-launch"
            assert fixture.journal.state(launch_stage).status == "completed"
            assert fixture.extra.server is not None and fixture.extra.server.started
            record = await request_one(
                fixture.client,
                fixture.url + "/v1/completions",
                WorkItem(case_id="capacity-extra", prompt="collected-extra", max_tokens=1),
                0,
                time.perf_counter(),
                RunConfig(requests=1, concurrency=1, warmup=0, model="fixture"),
                "capacity-fixture",
            )
            assert record.success and record.routing is not None
            assert record.routing.physical is not None and record.routing.anchor is not None
            assert record.routing.physical.revision_id == "extra"
            assert record.routing.anchor.revision_id == "primary"
            assert record.routing.pool_generation is not None
            probe = await fixture.adapter.health(fixture.plan.deployment_id)
            assert probe.revision_id == fixture.plan.primary.revision.revision_id
            fixture.extra.hold("replica-held")
            second = asyncio.create_task(completion(fixture, "replica-held"))
            await entered(fixture.extra, "replica-held")
            try:
                await occupancy(controller, 2)
                assert controller.observation().serving_active == 2
                fixture.primary.holds["primary-held"].set()
                primary_response = await first
                await occupancy(controller, 1)
                assert primary_response.status_code == 200
                assert primary_response.headers["x-finserve-revision"] == "primary"
                assert controller.observation().serving_active == 1
                now[0] += 1
                assert (await controller.tick()).phase == "two"
                now[0] += 1
                assert (await controller.tick()).phase == "draining"
                assert not any(call[2] == "stop" for call in fixture.daemon.calls)
                after_retire = await completion(fixture, "after-retirement")
                assert after_retire.headers["x-finserve-revision"] == "primary"
                assert (
                    int(after_retire.headers["x-finserve-pool-generation"])
                    > record.routing.pool_generation
                )
                assert "after-retirement" not in fixture.extra.received
                now[0] += 1
                assert (await controller.tick()).phase == "draining"
                assert not any(call[2] == "stop" for call in fixture.daemon.calls)
                fixture.extra.holds["replica-held"].set()
                replica_response = await second
                assert replica_response.status_code == 200
                assert replica_response.headers["x-finserve-revision"] == "extra"
                assert (
                    primary_response.headers["x-finserve-revision-digest"]
                    != replica_response.headers["x-finserve-revision-digest"]
                )
                now[0] += 1
                terminal = await controller.tick()
                assert terminal.phase in {"one", "closed"}
                assert (
                    fixture.journal.state(fixture.plan.plan_id + ":replica-0-stop").status
                    == "completed"
                )
                assert fixture.daemon.container is None
                assert len([call for call in fixture.daemon.calls if call[2] == "create"]) == 1
                now[0] += 1
                await controller.tick()
                assert len([call for call in fixture.daemon.calls if call[2] == "create"]) == 1
            finally:
                fixture.extra.holds["replica-held"].set()
                await second
        finally:
            fixture.primary.holds["primary-held"].set()
            await first


async def test_nonserving_obligations_and_rejected_requests_are_not_occupancy(
    tmp_path: Path,
) -> None:
    """Durable collector/reserved rows retain ownership without manufacturing dispatched load."""
    async with capacity_fixture(tmp_path) as fixture:
        freeze_capacity(fixture.journal, fixture.routes, fixture.control, fixture.plan)
        controller = CapacityController(
            fixture.journal,
            fixture.routes,
            fixture.control,
            fixture.plan,
            DockerRuntime(fixture.daemon),
            fixture.client,
        )
        collector = fixture.routes.require_stable_baseline(
            fixture.control,
            fixture.plan.deployment_id,
            fixture.routes.backend("primary"),
            1,
            reserve=True,
        )
        assert collector is not None
        reserved = select_admission(
            fixture.routes,
            fixture.routes.snapshot(fixture.plan.deployment_id),
            InferenceRequest(model="fixture", prompt="never-dispatched"),
            None,
        )
        try:
            with fixture.routes.transaction() as connection:
                kinds = connection.execute(
                    "SELECT json_extract(payload,'$.kind'),json_extract(payload,'$.phase') "
                    "FROM warm_admissions ORDER BY 1"
                ).fetchall()
            assert kinds == [("collector", "reserved"), ("serving", "reserved")]
            before = controller.observation()
            assert before.serving_active == 0
            unauthorized = await fixture.client.post(
                fixture.url + "/v1/completions",
                headers={"authorization": "Bearer wrong"},
                json={"model": "fixture", "prompt": "unauthorized", "max_tokens": 1},
            )
            malformed = await fixture.client.post(
                fixture.url + "/v1/completions",
                json={"model": "fixture", "prompt": ""},
            )
            assert unauthorized.status_code == 401
            assert malformed.status_code == 422
            after = controller.observation()
            assert after.serving_active == 0
            assert after.rejected_total == before.rejected_total
            assert "unauthorized" not in fixture.primary.received
            assert "never-dispatched" not in fixture.primary.received
            assert not fixture.daemon.calls
        finally:
            fixture.routes.finish_admission(reserved)
            fixture.routes.finish_admission(collector)


async def test_competing_plan_and_stale_anchor_cannot_launch(tmp_path: Path) -> None:
    """Frozen authorization is exclusive and route replacement invalidates later allocation."""
    async with capacity_fixture(tmp_path) as fixture:
        freeze_capacity(fixture.journal, fixture.routes, fixture.control, fixture.plan)
        rival = fixture.plan.model_copy(update={"plan_id": "competing-plan"})
        with pytest.raises((RegistryConflict, ControlConflict, ValueError)):
            freeze_capacity(fixture.journal, fixture.routes, fixture.control, rival)
        fixture.routes.apply(
            ApplyRequest(
                deployment_id=fixture.plan.deployment_id,
                expected_revision="primary",
                expected_generation=1,
                target=fixture.routes.backend("baseline").revision,
                idempotency_key="capacity-stale-anchor-fixture",
            )
        )
        controller = CapacityController(
            fixture.journal,
            fixture.routes,
            fixture.control,
            fixture.plan,
            DockerRuntime(fixture.daemon),
            fixture.client,
        )
        result = await controller.tick()
        assert result.phase in {"closed", "blocked"}
        assert not fixture.daemon.calls
        assert fixture.primary.server is not None and fixture.primary.server.started


async def test_concurrent_controllers_share_one_launch_attempt(tmp_path: Path) -> None:
    """Two actual executors racing the same high-load sample cannot allocate two replicas."""
    async with capacity_fixture(tmp_path) as fixture:
        now = [100.0]
        freeze_capacity(fixture.journal, fixture.routes, fixture.control, fixture.plan)
        controllers = [
            CapacityController(
                fixture.journal,
                fixture.routes,
                fixture.control,
                fixture.plan,
                DockerRuntime(fixture.daemon),
                fixture.client,
                clock=lambda: now[0],
            )
            for _ in range(2)
        ]
        fixture.primary.hold("race-held")
        task = asyncio.create_task(completion(fixture, "race-held"))
        await entered(fixture.primary, "race-held")
        try:
            await occupancy(controllers[0], 1)
            assert (await controllers[0].tick()).phase == "one"
            now[0] += 1
            started, release = threading.Event(), threading.Event()
            fixture.daemon.hold_create = (started, release)
            owner = asyncio.create_task(controllers[0].tick())
            try:
                assert await asyncio.to_thread(started.wait, 3)
                contender = asyncio.create_task(controllers[1].tick())
                release.set()
                results = await asyncio.gather(owner, contender)
            finally:
                release.set()
                await owner
            assert all(result.phase == "two" for result in results), results
            assert len([call for call in fixture.daemon.calls if call[2] == "create"]) == 1
            history = fixture.journal.history(fixture.plan.plan_id + ":replica-0-launch")
            attempts = {state.attempt_id for state in history if state.attempt_id is not None}
            assert len(attempts) == 1
        finally:
            fixture.primary.holds["race-held"].set()
            await task
        for _ in range(4):
            now[0] += 1
            await controllers[0].tick()
        assert fixture.daemon.container is None
        assert fixture.primary.server is not None and fixture.primary.server.started


async def test_lost_create_response_blocks_then_aborts_exact_attempt(tmp_path: Path) -> None:
    """An ambiguous create retains its slot until fenced abort proves exact owned cleanup."""
    async with capacity_fixture(tmp_path) as fixture:
        now = [100.0]
        freeze_capacity(fixture.journal, fixture.routes, fixture.control, fixture.plan)
        controller = CapacityController(
            fixture.journal,
            fixture.routes,
            fixture.control,
            fixture.plan,
            DockerRuntime(fixture.daemon),
            fixture.client,
            clock=lambda: now[0],
        )
        fixture.daemon.fail_create = True
        fixture.primary.hold("failure-held")
        task = asyncio.create_task(completion(fixture, "failure-held"))
        await entered(fixture.primary, "failure-held")
        try:
            await occupancy(controller, 1)
            await controller.tick()
            now[0] += 1
            failed = await controller.tick()
            assert failed.phase == "blocked" and failed.error is not None
            assert fixture.daemon.container is not None
            assert fixture.extra.server is None
            assert controller.observation().members == 1
            now[0] += 1
            cleaned = await controller.tick()
            assert cleaned.phase == "closed", cleaned
            assert fixture.daemon.container is None
            assert (
                fixture.journal.state(fixture.plan.plan_id + ":replica-0-abort").status
                == "completed"
            )
            now[0] += 1
            await controller.tick()
            assert len([call for call in fixture.daemon.calls if call[2] == "create"]) == 1
            assert fixture.primary.server is not None and fixture.primary.server.started
        finally:
            fixture.primary.holds["failure-held"].set()
            await task
