"""Borrowed collectors own durable obligations through task and HTTP-client completion."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_managed_runtime import Daemon, handler
from test_runtime_abort import setup_launch
from test_warm_drain import obligations
from test_warm_rollback import bound_backend

from finserve.benchmark.runner import RunConfig
from finserve.benchmark.workload import WorkItem, Workload
from finserve.contracts.deployment import DeploymentState, HealthObservation
from finserve.evaluation.quality import default_suite
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.producer_pipeline import (
    FrozenExecution,
    ProducerExecution,
    borrowed_collection,
)
from finserve.registry.producer_runtime import (
    ProducerEngine,
    ProducerInput,
    freeze_producer,
    require_collector_lease,
)
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import declare_input
from finserve.registry.runtime_stages import launch_runtime_stage
from finserve.reliability.promotion import PromotionPolicy
from finserve.reliability.rollback import ApplyRequest, DeploymentStore
from finserve.reliability.warm_drain import ADMISSION_PROTOCOL
from finserve.reliability.warm_routes import BackendConfiguration, WarmBackend, WarmRouteStore

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
async def borrowed(
    tmp_path: Path,
) -> AsyncIterator[
    tuple[
        ProducerStages, FrozenExecution, WarmRouteStore, DeploymentStore, WarmBackend, WarmBackend
    ]
]:
    """Bind a completed owned fixture runtime to a separate borrower's frozen execution."""
    journal, spec, _ = setup_launch(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await launch_runtime_stage(
                journal,
                "job:launch",
                "job:model",
                "job:build",
                spec,
                tmp_path / "runtime",
                client,
                DockerRuntime(Daemon(spec)),
            )
        routes, control = WarmRouteStore(tmp_path / "routes"), DeploymentStore(tmp_path / "control")
        backend = WarmBackend(
            revision=spec.revision,
            serving_profile=spec.profile,
            configuration=BackendConfiguration(base_url=spec.profile.base_url, model="fixture"),
        )
        alternate = bound_backend("new", "http://127.0.0.1:8061")
        configuration = alternate.configuration.model_copy(update={"model": "fixture"})
        alternate = WarmBackend(
            configuration=configuration,
            revision=alternate.revision.model_copy(
                update={"config_digest": configuration.digest()}
            ),
        )
        routes.register(backend)
        routes.register(alternate)
        routes.bootstrap("service", backend.revision.revision_id)
        control.register_revision(backend.revision)
        control.bootstrap(
            "service",
            backend.revision.revision_id,
            HealthObservation(
                revision_id=backend.revision.revision_id,
                revision_digest=backend.revision.digest(),
                ready=True,
                smoke_passed=True,
            ),
        )
        producer = ProducerInput(
            job_id="borrower",
            deployment_id="service",
            expected_generation=0,
            source_revision=spec.image.specification.source_revision,
            collector_revision="c" * 40,
            model=spec.model,
            existing_baseline_stage="job:launch",
            baseline=ProducerEngine(port=8060, parameters=VLLMParameters()),
            candidate=ProducerEngine(port=8061, parameters=VLLMParameters()),
            workload=Workload(
                suite_id="fixture",
                version=1,
                items=(WorkItem(case_id="one", prompt="hello", max_tokens=1),),
            ),
            load=RunConfig(model="fixture", hardware="fixture-cpu", requests=1, warmup=0),
            suite=default_suite(),
            policy=PromotionPolicy(version="fixture"),
            repository=REPOSITORY,
            workspace=tmp_path / "borrower",
        )
        freeze_producer(journal, producer)
        frozen = FrozenExecution(
            execution=ProducerExecution(
                producer=producer, routes=routes.path, control=control.path
            ),
            routes_identity=routes.identity,
            control_identity=control.identity,
            collection_protocol=ADMISSION_PROTOCOL,
        )
        declare_input(journal, "borrower:execution", frozen.model_dump(mode="json"))
        yield journal, frozen, routes, control, backend, alternate
    finally:
        journal.registry.close()


def stabilize_fixture(
    routes: WarmRouteStore, control: DeploymentStore, old: WarmBackend, new: WarmBackend
) -> None:
    """Simulate another fully stabilized release; this test concerns collection ownership only."""
    routes.apply(
        ApplyRequest(
            deployment_id="service",
            expected_revision=old.revision.revision_id,
            expected_generation=0,
            target=new.revision,
            idempotency_key="switch",
        )
    )
    state = DeploymentState(
        deployment_id="service",
        active_revision=new.revision.revision_id,
        known_good_revision=new.revision.revision_id,
        generation=1,
    )
    with control.transaction() as connection:
        connection.execute(
            "UPDATE deployments SET payload=? WHERE id='service'", (state.model_dump_json(),)
        )


@pytest.mark.parametrize("failed", [False, True])
async def test_collector_obligation_outlives_other_job_promotion(
    borrowed: Any, failed: bool
) -> None:
    """Inactive/known-good movement does not override an outstanding direct collector pin."""
    journal, frozen, routes, control, old, new = borrowed

    async def collect() -> None:
        """Only successful completion acknowledges the exact collector obligation."""
        async with borrowed_collection(journal, frozen):
            assert obligations(routes) == 1
            require_collector_lease(journal, frozen.execution.producer, asyncio.current_task())
            stabilize_fixture(routes, control, old, new)
            assert not routes.retire_drained(control, old)
            if failed:
                raise RuntimeError("collector close is uncertain")

    if failed:
        with pytest.raises(RuntimeError, match="uncertain"):
            await collect()
    else:
        await collect()
    assert obligations(routes) == int(failed)
    assert routes.retire_drained(control, old) is (not failed)


async def test_collector_context_cannot_be_bypassed_or_inherited(borrowed: Any) -> None:
    """New task helpers bind the exact frozen input, store, backend and owning asyncio task."""
    journal, frozen, routes, _, _, _ = borrowed
    with pytest.raises(ValueError, match="exact collector lease"):
        require_collector_lease(journal, frozen.execution.producer, asyncio.current_task())
    async with borrowed_collection(journal, frozen):

        async def child() -> None:
            """Copied task context does not authorize a second executor's borrowed network work."""
            require_collector_lease(journal, frozen.execution.producer, asyncio.current_task())

        with pytest.raises(ValueError, match="exact collector lease"):
            await asyncio.create_task(child())
    assert obligations(routes) == 0


async def test_collector_cancellation_retains_durable_obligation(borrowed: Any) -> None:
    """A vanished producer task cannot expire into evidence that its native/HTTP work drained."""
    journal, frozen, routes, control, old, new = borrowed
    entered = asyncio.Event()

    async def collect() -> None:
        """Hold task lifetime after lease publication to exercise cancellation at its owner."""
        async with borrowed_collection(journal, frozen):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(collect())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stabilize_fixture(routes, control, old, new)
    assert obligations(routes) == 1 and not routes.retire_drained(control, old)


async def test_collector_lease_includes_http_client_close(borrowed: Any) -> None:
    """Collection success waits for pool closure before acknowledging its obligation."""
    journal, frozen, routes, control, old, new = borrowed
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldClient(httpx.AsyncBaseTransport):
        """An idle collector client's close still belongs to the borrowed task lifetime."""

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            """This ownership test performs no model requests."""
            raise AssertionError("unexpected request")

        async def aclose(self) -> None:
            """Expose the exact post-collection/pre-client-close boundary."""
            entered.set()
            await release.wait()

    async def collect() -> None:
        """Match collection_stage's nesting so the obligation outlives the client context."""
        async with borrowed_collection(journal, frozen):
            async with httpx.AsyncClient(transport=HeldClient()):
                pass

    task = asyncio.create_task(collect())
    await entered.wait()
    stabilize_fixture(routes, control, old, new)
    assert obligations(routes) == 1 and not routes.retire_drained(control, old)
    release.set()
    await task
    assert routes.retire_drained(control, old)
