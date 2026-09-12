"""Airflow-callable producer tasks reopen frozen inputs and trusted local stores per task."""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.rollout import RolloutSettings
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.model_assets import owned_disk
from finserve.registry.pipeline import runtime
from finserve.registry.producer_runtime import (
    ProducerInput,
    ProducerStep,
    cleanup_unserved,
    existing_baseline,
    freeze_producer,
    produce_step,
    producer_input,
)
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import declare_input
from finserve.reliability.rollback import ControlConflict, DeploymentStore
from finserve.reliability.store_identity import existing_identity
from finserve.reliability.warm_drain import ADMISSION_PROTOCOL, collector_admission
from finserve.reliability.warm_routes import BackendConfiguration, WarmBackend, WarmRouteStore


def existing_store(path: Path) -> Path:
    """Missing or redirected stores must not turn a cleanup decision into an empty-state read."""
    repository = Path(__file__).resolve().parents[3]
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or not path.is_file()
        or repository in path.parents
    ):
        raise ValueError("canonical existing external store file required")
    return path


class ProducerExecution(ImmutableModel):
    """The worker freezes route/control destinations together with the producer specification."""

    producer: ProducerInput
    routes: Path
    control: Path
    rollout: RolloutSettings | None = None

    @model_validator(mode="after")
    def separate_stores(self) -> "ProducerExecution":
        """The route-to-controller transaction order requires distinct canonical database files."""
        if self.routes == self.control:
            raise ValueError("route and control stores must be distinct")
        return self

    def verify_stores(self) -> None:
        """Recheck existence on every task without silently recreating lost traffic state."""
        existing_store(self.routes)
        existing_store(self.control)
        if self.routes.samefile(self.control):
            raise ValueError("route and control stores must be distinct files")


class FrozenExecution(ImmutableModel):
    """Bind the trusted request to persistent database identities, not only file locations."""

    execution: ProducerExecution
    routes_identity: str = Field(pattern=r"^[0-9a-f]{32}$")
    control_identity: str = Field(pattern=r"^[0-9a-f]{32}$")
    collection_protocol: Literal["durable-http-close-v1"] | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Old stores retain frozen execution bytes; new executions reject old task parsers."""
        result: dict[str, Any] = handler(self)
        if self.collection_protocol is None:
            result.pop("collection_protocol", None)
        return result

    def stores(self) -> tuple[WarmRouteStore, DeploymentStore]:
        """Open without initialization; each subsequent transaction rechecks the same identity."""
        self.execution.verify_stores()
        return (
            WarmRouteStore(self.execution.routes, expected_identity=self.routes_identity),
            DeploymentStore(self.execution.control, expected_identity=self.control_identity),
        )


@contextmanager
def journal_runtime() -> Generator[ProducerStages]:
    """Each task owns its registry connection; task arguments never select storage endpoints."""
    registry, artifacts = runtime()
    try:
        yield ProducerStages(registry, artifacts)
    finally:
        registry.close()


def execution_input(journal: ProducerStages, job_id: str) -> FrozenExecution:
    """Cross-check both immutable inputs before acquiring HTTP or Docker resources."""
    state = journal.state(job_id + ":execution")
    execution = FrozenExecution.model_validate_json(journal.artifacts.get(state.input))
    if execution.execution.producer != producer_input(journal, job_id):
        raise ValueError("execution differs from frozen producer input")
    execution.stores()
    return execution


def verify_borrowed_baseline(journal: ProducerStages, frozen: FrozenExecution) -> None:
    """A referenced launch may be borrowed only while it remains this deployment's stable route."""
    spec = frozen.execution.producer
    if spec.existing_baseline_stage is None:
        return
    launch = existing_baseline(journal, spec)
    routes, control = frozen.stores()
    routes.require_stable_baseline(
        control,
        spec.deployment_id,
        WarmBackend(
            revision=launch.revision,
            serving_profile=launch.profile,
            configuration=BackendConfiguration(
                base_url=launch.profile.base_url,
                model=launch.profile.served_model,
            ),
        ),
        spec.expected_generation,
    )


def freeze_stage(*, require_rollout: bool = False) -> str:
    """Read one bounded server-owned request; retries reject any changed job or store mapping."""
    with Path(os.environ["FINSERVE_PRODUCER_REQUEST"]).open("rb") as source:
        payload = source.read(4 * 1024**2 + 1)
    if len(payload) > 4 * 1024**2:
        raise ValueError("producer execution input exceeds four MiB")
    execution = ProducerExecution.model_validate_json(payload)
    if require_rollout and execution.rollout is None:
        raise ValueError("producer DAG requires frozen rollout settings")
    execution.verify_stores()
    with journal_runtime() as journal:
        if execution.rollout is not None:
            if (
                execution.producer.expected_generation != 0
                and execution.producer.existing_baseline_stage is None
            ):
                raise ValueError("noninitial deployment requires an existing baseline stage")
            if execution.producer.existing_baseline_stage is None and not journal.history(
                execution.producer.job_id + ":execution"
            ):
                routes = WarmRouteStore(
                    execution.routes, expected_identity=existing_identity(execution.routes)
                )
                control = DeploymentStore(
                    execution.control, expected_identity=existing_identity(execution.control)
                )
                for lookup in (routes.snapshot, control.deployment):
                    try:
                        lookup(execution.producer.deployment_id)
                    except KeyError:
                        continue
                    raise ValueError("producer rollout requires an unused deployment ID")
        job_id = freeze_producer(journal, execution.producer)
        # freeze_producer canonicalizes the workspace/repository before declaring its input.
        execution = execution.model_copy(update={"producer": producer_input(journal, job_id)})
        frozen = FrozenExecution(
            execution=execution,
            routes_identity=existing_identity(execution.routes),
            control_identity=existing_identity(execution.control),
            collection_protocol=WarmRouteStore(
                execution.routes, expected_identity=existing_identity(execution.routes)
            ).admission_protocol,
        )
        if not journal.history(job_id + ":execution"):
            verify_borrowed_baseline(journal, frozen)
        declare_input(journal, job_id + ":execution", frozen.model_dump(mode="json"))
        return job_id


@asynccontextmanager
async def borrowed_collection(
    journal: ProducerStages, frozen: FrozenExecution
) -> AsyncGenerator[None]:
    """Protect each borrowed task through collection and client close; failed tasks retain proof."""
    spec = frozen.execution.producer
    routes, control = await owned_disk(frozen.stores)
    if spec.existing_baseline_stage is None or routes.admission_protocol is None:
        yield
        return
    if frozen.collection_protocol != ADMISSION_PROTOCOL or collector_admission.get() is not None:
        raise ControlConflict("borrowed collection requires its immutable execution protocol")
    launch = await owned_disk(lambda: existing_baseline(journal, spec))
    backend = WarmBackend(
        revision=launch.revision,
        serving_profile=launch.profile,
        configuration=BackendConfiguration(
            base_url=launch.profile.base_url, model=launch.profile.served_model
        ),
    )
    lease = await owned_disk(
        lambda: routes.require_stable_baseline(
            control,
            spec.deployment_id,
            backend,
            spec.expected_generation,
            reserve=True,
        )
    )
    assert lease is not None
    reference = await owned_disk(lambda: journal.state(spec.job_id + ":execution").input)
    token = collector_admission.set(
        (lease, reference.sha256, routes.identity, asyncio.current_task())
    )
    try:
        yield
    except BaseException:
        # Stage-specific cleanup may be uncertain; a generic wrapper cannot infer a
        # successful native/HTTP drain from exception type or executor age.
        raise
    else:
        await owned_disk(lambda: routes.finish_admission(lease))
    finally:
        collector_admission.reset(token)


def collection_client() -> httpx.AsyncClient:
    """Collection workers ignore proxy environment and never forward credentials on redirects."""
    return httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=30)


def collection_stage(job_id: str, step: ProducerStep) -> str:
    """Run one owned producer action to completion before closing the per-task resources."""
    with journal_runtime() as journal:
        frozen = execution_input(journal, job_id)
        verify_borrowed_baseline(journal, frozen)

        async def collect() -> str:
            """Use one client lifetime per task; inner stages retain cancellation ownership."""
            async with borrowed_collection(journal, frozen):
                async with collection_client() as client:
                    return await produce_step(journal, job_id, step, client, DockerRuntime())

        return asyncio.run(collect())


def cleanup_stage(job_id: str) -> dict[str, str]:
    """An all-done task uses the frozen stores and preserves ambiguous or traffic-owned work."""
    with journal_runtime() as journal:
        execution = execution_input(journal, job_id)
        routes, control = execution.stores()
        return asyncio.run(
            cleanup_unserved(
                journal,
                job_id,
                routes,
                control,
                DockerRuntime(),
            )
        )
