"""Airflow-callable producer tasks reopen frozen inputs and trusted local stores per task."""

import asyncio
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import httpx
from pydantic import Field, model_validator

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.rollout import RolloutSettings
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.pipeline import runtime
from finserve.registry.producer_runtime import (
    ProducerInput,
    ProducerStep,
    cleanup_unserved,
    freeze_producer,
    produce_step,
    producer_input,
)
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import declare_input
from finserve.reliability.rollback import DeploymentStore
from finserve.reliability.store_identity import existing_identity
from finserve.reliability.warm_routes import WarmRouteStore


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
            if execution.producer.expected_generation != 0:
                raise ValueError("producer rollout currently requires an initial deployment")
            if not journal.history(execution.producer.job_id + ":execution"):
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
        )
        declare_input(journal, job_id + ":execution", frozen.model_dump(mode="json"))
        return job_id


def collection_client() -> httpx.AsyncClient:
    """Collection workers ignore proxy environment and never forward credentials on redirects."""
    return httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=30)


def collection_stage(job_id: str, step: ProducerStep) -> str:
    """Run one owned producer action to completion before closing the per-task resources."""
    with journal_runtime() as journal:
        execution_input(journal, job_id)

        async def collect() -> str:
            """Use one client lifetime per task; inner stages retain cancellation ownership."""
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
