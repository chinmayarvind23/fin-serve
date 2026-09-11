"""Server-configured Airflow stage entry points, with immutable job IDs as the only task handoff."""

import asyncio
import importlib
import os
from pathlib import Path
from typing import cast

from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.lifecycle import LifecycleService, LifecycleSpec
from finserve.registry.metadata import Registry
from finserve.reliability.promotion import PromotionPolicy
from finserve.reliability.rollback import DeploymentAdapter


class PipelineRequest(ImmutableModel):
    """This server-owned input identifies existing verified build/run artifacts, not client
    assertions.
    """

    job_id: str
    deployment_id: str
    expected_generation: int
    baseline_directory: Path
    candidate_directory: Path
    quality_file: Path
    suite_file: Path
    baseline_revision: Revision
    target_revision: Revision
    policy: PromotionPolicy


def runtime() -> tuple[Registry, LocalArtifactStore]:
    """Resolve database and artifact endpoints only at task runtime, never during DAG parsing."""
    return Registry(os.environ["FINSERVE_REGISTRY_URL"]), LocalArtifactStore(
        Path(os.environ["FINSERVE_ARTIFACT_ROOT"])
    )


def register_stage() -> str:
    """Register frozen runs and quality inputs, then persist the full immutable lifecycle
    specification.
    """
    request = PipelineRequest.model_validate_json(
        Path(os.environ["FINSERVE_PIPELINE_REQUEST"]).read_text()
    )
    registry, artifacts = runtime()
    try:
        baseline = registry.register_run(
            request.baseline_directory, artifacts, request.baseline_revision
        )
        candidate = registry.register_run(
            request.candidate_directory, artifacts, request.target_revision
        )
        spec = LifecycleSpec(
            job_id=request.job_id,
            deployment_id=request.deployment_id,
            expected_revision=request.baseline_revision.revision_id,
            expected_generation=request.expected_generation,
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            quality=artifacts.put(request.quality_file.read_bytes()),
            suite=artifacts.put(request.suite_file.read_bytes()),
            policy=request.policy,
            target=request.target_revision,
        )
        registry.create_job(spec.job_id, spec.canonical())
        return spec.job_id
    finally:
        registry.close()


def evaluate_stage(job_id: str) -> str:
    """Recompute gates from registered bytes; rejected jobs fail the task before deployment."""
    registry, artifacts = runtime()
    try:
        specification = LifecycleSpec.model_validate_json(registry.specification(job_id))
        state = asyncio.run(
            LifecycleService(registry, artifacts).run(specification, evaluate_only=True)
        )
        if state.status not in {"evaluated", "promoted"} or state.last_error is not None:
            raise RuntimeError("lifecycle evaluation did not pass")
        return job_id
    finally:
        registry.close()


def deploy_stage(job_id: str) -> str:
    """Load a server-configured adapter and rely on durable lifecycle reconciliation across
    retries.
    """
    module_name, factory_name = os.environ["FINSERVE_DEPLOYMENT_ADAPTER"].split(":", maxsplit=1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    adapter = cast(DeploymentAdapter, factory())
    registry, artifacts = runtime()
    try:
        specification = LifecycleSpec.model_validate_json(registry.specification(job_id))
        state = asyncio.run(LifecycleService(registry, artifacts).run(specification, adapter))
        if state.status != "promoted":
            raise RuntimeError("lifecycle deployment is not verified healthy")
        return job_id
    finally:
        registry.close()
