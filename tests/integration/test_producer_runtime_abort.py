"""Verify all-done producer cleanup safely retires and reconciles failed runtime startups."""

import asyncio
from pathlib import Path

import httpx
import pytest
from test_managed_runtime import Daemon
from test_runtime_abort import fail_readiness, setup_launch

from finserve.benchmark.runner import RunConfig
from finserve.benchmark.workload import WorkItem, Workload
from finserve.evaluation.quality import default_suite
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.metadata import RegistryConflict
from finserve.registry.producer_runtime import (
    ProducerEngine,
    ProducerInput,
    cleanup_unserved,
    freeze_producer,
    prepared_cohorts,
)
from finserve.registry.runtime_stages import launch_runtime_stage
from finserve.reliability.promotion import PromotionPolicy
from finserve.reliability.rollback import ControlConflict, DeploymentStore
from finserve.reliability.warm_routes import BackendConfiguration, WarmBackend, WarmRouteStore

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fault", ["readiness", "pending_create", "traffic"])
async def test_all_done_cleanup_reconciles_only_retired_owned_attempts(
    tmp_path: Path, fault: str
) -> None:
    """Only terminal abort releases an endpoint; mutable model damage never prevents cleanup."""
    journal, template, _ = setup_launch(tmp_path)
    producer = ProducerInput(
        job_id="job",
        deployment_id="fixture",
        expected_generation=0,
        source_revision=template.image.specification.source_revision,
        collector_revision="c" * 40,
        model=template.model,
        baseline=ProducerEngine(port=8061, parameters=VLLMParameters()),
        candidate=ProducerEngine(port=8060, parameters=VLLMParameters()),
        workload=Workload(
            suite_id="fixture",
            version=1,
            items=(
                WorkItem(
                    case_id="one",
                    prompt="hello",
                    max_tokens=1,
                ),
            ),
        ),
        load=RunConfig(model="fixture", hardware="fixture-cpu", requests=1, warmup=0),
        suite=default_suite(),
        policy=PromotionPolicy(version="fixture"),
        repository=REPOSITORY,
        workspace=tmp_path,
    )
    freeze_producer(journal, producer)
    _, runtimes = prepared_cohorts(journal, producer)
    spec = runtimes[1]
    daemon = Daemon(spec)

    def command(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """An accepted create can remain unresolved after its local CLI times out."""
        if fault == "pending_create" and arguments[2] == "create":
            raise TimeoutError("pending daemon create")
        daemon(arguments, directory, output, timeout)

    runtime = DockerRuntime(command)
    runtime.probe_endpoint = fail_readiness
    routes = WarmRouteStore(tmp_path / "routes.sqlite")
    control = DeploymentStore(tmp_path / "control.sqlite")
    backend = WarmBackend(
        revision=spec.revision,
        serving_profile=spec.profile,
        configuration=BackendConfiguration(
            base_url=spec.profile.base_url, model=spec.profile.served_model
        ),
    )
    other = backend.model_copy(
        update={"revision": spec.revision.model_copy(update={"revision_id": "another-job"})}
    )
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises((RuntimeError, TimeoutError)):
                await launch_runtime_stage(
                    journal,
                    "job:candidate-launch",
                    "job:model",
                    "job:build",
                    spec,
                    tmp_path / "candidate" / "runtime",
                    client,
                    runtime,
                )
        if fault == "traffic":
            routes.register(backend)
            routes.bootstrap("fixture", spec.revision.revision_id)
        await asyncio.to_thread((spec.model_directory / "config.json").unlink)
        before = len(daemon.calls)
        expected = {
            "readiness": "aborted",
            "pending_create": "needs_reconciliation",
            "traffic": "preserved_for_traffic",
        }[fault]
        assert await cleanup_unserved(journal, "job", routes, control, runtime) == {
            "baseline": "not_launched",
            "candidate": expected,
        }
        if fault == "readiness":
            assert daemon.container is None
            routes.register(other)
            assert await cleanup_unserved(journal, "job", routes, control, runtime) == {
                "baseline": "not_launched",
                "candidate": "aborted",
            }
            # Restoring model bytes must not turn task replay into a relaunch of a retired revision.
            (spec.model_directory / "config.json").write_bytes(b'{"model_type":"fixture"}')
            async with httpx.AsyncClient() as client:
                with pytest.raises(RegistryConflict, match="new producer identity"):
                    await launch_runtime_stage(
                        journal,
                        "job:candidate-launch",
                        "job:model",
                        "job:build",
                        spec,
                        tmp_path / "candidate" / "runtime",
                        client,
                        runtime,
                    )
        else:
            assert not any(call[2] in {"stop", "rm"} for call in daemon.calls[before:])
            with pytest.raises(ControlConflict):
                routes.register(other)
    finally:
        journal.registry.close()
