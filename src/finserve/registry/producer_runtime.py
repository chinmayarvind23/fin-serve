"""Server-owned task runtime deriving collection identities from actual model and image receipts."""

import json
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self, get_args

import httpx
from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from finserve.benchmark.runner import RunConfig
from finserve.benchmark.workload import Workload
from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.contracts.managed_runtime import RuntimeLaunchSpec
from finserve.contracts.model_assets import ModelFetchSpec
from finserve.contracts.performance import PerformanceCollectionSpec
from finserve.contracts.producer import QualityCollectionSpec
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.evaluation.quality import GoldenSuite
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.managed_quality import managed_quality_stage
from finserve.registry.managed_runtime import DockerRuntime, owned_directory
from finserve.registry.model_assets import owned_disk
from finserve.registry.performance_stages import performance_stage
from finserve.registry.produced_release import (
    ProducedReleasePlan,
    ReleaseCohort,
    freeze_produced_release,
)
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import (
    build_stage,
    declare_input,
    fetch_stage,
    verified_model_receipt,
)
from finserve.registry.runtime_build import RuntimeBuildSpec, RuntimeImage
from finserve.registry.runtime_stages import launch_runtime_stage, load_launch, stop_runtime_stage
from finserve.reliability.promotion import PromotionPolicy
from finserve.reliability.rollback import DeploymentStore
from finserve.reliability.warm_routes import BackendConfiguration, WarmBackend, WarmRouteStore


class ProducerEngine(ImmutableModel):
    """Only typed engine knobs and a fixed loopback port vary between cohorts."""

    port: int = Field(ge=1024, le=65535, strict=True)
    parameters: VLLMParameters


class ProducerInput(ImmutableModel):
    """Freeze source, weights, workload and policy before fetching or observing outputs."""

    job_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    deployment_id: str = Field(min_length=1, max_length=128)
    expected_generation: int = Field(ge=0, strict=True)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    collector_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model: ModelFetchSpec
    baseline: ProducerEngine
    candidate: ProducerEngine
    existing_baseline_stage: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+:[A-Za-z0-9_-]+$", max_length=128
    )
    workload: Workload
    load: RunConfig
    suite: GoldenSuite
    quality_max_tokens: int = Field(default=128, ge=1, le=2048, strict=True)
    policy: PromotionPolicy
    repository: Path
    workspace: Path
    readiness_timeout_seconds: float = Field(default=300, gt=0, le=900)

    @model_serializer(mode="wrap")
    def preserve_initial_input(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Adding baseline reuse must not change canonical bytes of already frozen initial jobs."""
        result: dict[str, Any] = handler(self)
        if self.existing_baseline_stage is None:
            result.pop("existing_baseline_stage", None)
        return result

    @model_validator(mode="after")
    def bounded_input(self) -> Self:
        """Reject ambiguous destinations and oversized work before creating a producer namespace."""
        if (
            self.baseline.port == self.candidate.port
            or not self.repository.is_absolute()
            or not self.workspace.is_absolute()
            or len(self.workload.items) > 1024
            or len(self.suite.cases) > 1024
            or self.load.requests + self.load.warmup > 65536
            or self.load.concurrency > 128
            or self.load.hardware == "undeclared"
            or len(self.workload.model_dump_json().encode()) > 1024**2
            or len(self.suite.model_dump_json().encode()) > 1024**2
            or (
                self.existing_baseline_stage is not None
                and self.existing_baseline_stage.startswith(self.job_id + ":")
            )
        ):
            raise ValueError("producer input exceeds local bounds or has ambiguous paths/ports")
        if any(
            getattr(self.load, name) != "undeclared"
            for name in (
                "revision",
                "model_revision",
                "tokenizer_revision",
                "engine",
                "engine_config",
                "image_digest",
                "config_digest",
            )
        ):
            raise ValueError(
                "producer derives runtime load identities from verified build receipts"
            )
        if len(self.model_dump_json().encode()) > 4 * 1024**2:
            raise ValueError("producer input exceeds four MiB")
        return self


def freeze_producer(journal: ProducerStages, specification: ProducerInput) -> str:
    """Freeze input before any task; absolute resolved paths are part of its immutable namespace."""
    spec = ProducerInput.model_validate_json(specification.model_dump_json())
    workspace = owned_directory(spec.workspace, create=True)
    repository = spec.repository.resolve(strict=True)
    spec = ProducerInput.model_validate(
        {**spec.model_dump(), "workspace": workspace, "repository": repository}
    )
    declare_input(journal, spec.job_id + ":producer", spec.model_dump(mode="json"))
    return spec.job_id


def producer_input(journal: ProducerStages, job_id: str) -> ProducerInput:
    """Tasks accept only an existing frozen job ID; task arguments cannot replace runtime inputs."""
    state = journal.state(job_id + ":producer")
    spec = ProducerInput.model_validate_json(journal.artifacts.get(state.input))
    if spec.job_id != job_id:
        raise ValueError("producer job differs from frozen namespace")
    return spec


def prepared_cohorts(
    journal: ProducerStages,
    spec: ProducerInput,
) -> tuple[ProducedReleasePlan, tuple[RuntimeLaunchSpec, RuntimeLaunchSpec]]:
    """Bind actual model and image outputs into revisions before any collection begins."""
    model = verified_model_receipt(journal, journal.state(spec.job_id + ":model"))
    image_state = journal.state(spec.job_id + ":build")
    if image_state.status != "completed" or image_state.output is None:
        raise ValueError("producer image build is not completed")
    image = RuntimeImage.model_validate_json(journal.artifacts.get(image_state.output))
    frozen = json.loads(journal.artifacts.get(image_state.input))
    if (
        model.specification_sha256 != spec.model.digest()
        or model.directory != spec.workspace / "models" / spec.model.digest()
        or image.specification.source_revision != spec.source_revision
        or image.specification.model_manifest_sha256 != model.manifest.sha256
        or frozen.get("model_stage_id") != spec.job_id + ":model"
        or frozen.get("kind") != "runtime-image-v1"
        or frozen.get("model_manifest") != model.manifest.model_dump()
        or frozen.get("specification") != image.specification.model_dump(mode="json")
    ):
        raise ValueError("producer build differs from frozen source/model inputs")
    cohorts: list[ReleaseCohort] = []
    runtimes: list[RuntimeLaunchSpec] = []
    for name, engine in (("baseline", spec.baseline), ("candidate", spec.candidate)):
        profile = ServingProfileV1(
            engine="vllm",
            engine_version="0.29.0",
            engine_parameters_json=engine.parameters.model_dump_json(),
            model_revision=spec.model.revision,
            tokenizer_revision=spec.model.revision,
            model_manifest_sha256=model.manifest.sha256,
            tokenizer_manifest_sha256=model.manifest.sha256,
            base_url=f"http://127.0.0.1:{engine.port}/v1",
            served_model=spec.load.model,
        )
        revision = Revision(
            revision_id=spec.job_id + "-" + name,
            source_revision=spec.source_revision,
            model_revision=spec.model.revision,
            tokenizer_revision=spec.model.revision,
            image_digest=image.image_manifest_digest,
            config_digest=profile.digest(),
            engine=profile.engine,
            engine_config=profile.engine_parameters_json,
        )
        launch = RuntimeLaunchSpec(
            image=image,
            profile=profile,
            revision=revision,
            model=spec.model,
            model_directory=model.directory,
            readiness_timeout_seconds=spec.readiness_timeout_seconds,
        )
        if name == "baseline" and spec.existing_baseline_stage is not None:
            launch = existing_baseline(journal, spec)
            profile, revision = launch.profile, launch.revision
        configuration = RunConfig.model_validate(
            {
                **spec.load.model_dump(),
                "revision": revision.source_revision,
                "model_revision": revision.model_revision,
                "tokenizer_revision": revision.tokenizer_revision,
                "image_digest": revision.image_digest,
                "config_digest": revision.config_digest,
                "engine": revision.engine,
                "engine_config": revision.engine_config,
            }
        )
        cohorts.append(
            ReleaseCohort(
                performance_stage=spec.job_id + ":" + name + "-performance",
                quality_stage=spec.job_id + ":" + name + "-quality",
                performance=PerformanceCollectionSpec(
                    collection_id=spec.job_id + "-" + name,
                    collector_revision=spec.collector_revision,
                    profile=profile,
                    revision=revision,
                    workload=spec.workload,
                    configuration=configuration,
                ),
                quality=QualityCollectionSpec(
                    collection_id=spec.job_id + "-" + name,
                    profile=profile,
                    revision=revision,
                    suite=spec.suite,
                    configuration=configuration,
                    max_tokens=spec.quality_max_tokens,
                ),
            )
        )
        runtimes.append(launch)
    return ProducedReleasePlan(
        job_id=spec.job_id,
        deployment_id=spec.deployment_id,
        expected_generation=spec.expected_generation,
        baseline=cohorts[0],
        candidate=cohorts[1],
        policy=spec.policy,
    ), (runtimes[0], runtimes[1])


ProducerStep = Literal[
    "fetch",
    "build",
    "freeze",
    "baseline_launch",
    "baseline_quality",
    "baseline_performance",
    "candidate_launch",
    "candidate_quality",
    "candidate_performance",
]


async def produce_step(
    journal: ProducerStages,
    job_id: str,
    step: ProducerStep,
    client: httpx.AsyncClient,
    runtime: DockerRuntime,
) -> str:
    """Execute one resumable stage; actual resources and outcomes stay in the existing journal."""
    if step not in get_args(ProducerStep):
        raise ValueError("unsupported producer action")
    spec = await owned_disk(lambda: producer_input(journal, job_id))
    if step == "fetch":
        await fetch_stage(journal, job_id + ":model", client, spec.model, spec.workspace / "models")
        return job_id
    if step == "build":
        model = await owned_disk(
            lambda: verified_model_receipt(journal, journal.state(job_id + ":model"))
        )
        await build_stage(
            journal,
            job_id + ":build",
            job_id + ":model",
            RuntimeBuildSpec(
                source_revision=spec.source_revision, model_manifest_sha256=model.manifest.sha256
            ),
            spec.repository,
            spec.workspace / "build",
        )
        return job_id
    plan, runtimes = await owned_disk(lambda: prepared_cohorts(journal, spec))
    if step == "freeze":
        await owned_disk(lambda: freeze_produced_release(journal, plan))
        return job_id
    frozen = await owned_disk(lambda: journal.state(job_id + ":release-plan"))
    if frozen.status != "completed" or frozen.output != frozen.input:
        raise ValueError("collection requires a completed frozen release plan")
    recorded = ProducedReleasePlan.model_validate_json(
        await owned_disk(lambda: journal.artifacts.get(frozen.input))
    )
    if recorded != plan:
        raise ValueError("derived collections differ from frozen release plan")
    name, action = step.split("_", maxsplit=1)
    cohort, launch = (
        (plan.baseline, runtimes[0]) if name == "baseline" else (plan.candidate, runtimes[1])
    )
    launch_id = (spec.existing_baseline_stage if name == "baseline" else None) or (
        job_id + ":" + name + "-launch"
    )
    root = spec.workspace / name
    if action == "launch":
        if name == "baseline" and spec.existing_baseline_stage is not None:
            receipt = await owned_disk(
                lambda: load_launch(journal, journal.state(launch_id), launch)
            )
            await runtime.observe(launch, receipt, client)
            return job_id
        await launch_runtime_stage(
            journal,
            launch_id,
            job_id + ":model",
            job_id + ":build",
            launch,
            root / "runtime",
            client,
            runtime,
        )
    elif action == "quality":
        await managed_quality_stage(
            journal,
            cohort.quality_stage,
            launch_id,
            cohort.quality,
            root / "quality",
            client,
            runtime,
        )
    elif action == "performance":
        await performance_stage(
            journal,
            cohort.performance_stage,
            launch_id,
            cohort.performance,
            root / "performance",
            client,
            runtime,
        )
    else:
        raise ValueError("unsupported producer action")
    return job_id


async def cleanup_unserved(
    journal: ProducerStages,
    job_id: str,
    routes: WarmRouteStore,
    control: DeploymentStore,
    runtime: DockerRuntime,
) -> dict[str, str]:
    """Retire unused revisions before exact receipt-bound cleanup; preserve ambiguous launches."""
    spec = await owned_disk(lambda: producer_input(journal, job_id))
    launch_ids = [job_id + ":" + name + "-launch" for name in ("baseline", "candidate")]
    histories = [await owned_disk(lambda key=key: journal.history(key)) for key in launch_ids]
    outcomes: dict[str, str] = {}
    for name, launch_id, history in zip(
        ("baseline", "candidate"), launch_ids, histories, strict=True
    ):
        if name == "baseline" and spec.existing_baseline_stage is not None:
            outcomes[name] = "borrowed_baseline"
            continue
        if not history:
            outcomes[name] = "not_launched"
            continue
        state = await owned_disk(partial(journal.state, launch_id))
        if state.status != "completed":
            outcomes[name] = "needs_reconciliation"
            continue
        launch_spec = await owned_disk(partial(cleanup_launch, journal, spec, name, launch_id))
        backend = WarmBackend(
            revision=launch_spec.revision,
            serving_profile=launch_spec.profile,
            configuration=BackendConfiguration(
                base_url=launch_spec.profile.base_url, model=launch_spec.profile.served_model
            ),
        )
        retired = await owned_disk(partial(routes.retire_unserved, control, backend))
        if not retired:
            outcomes[name] = "preserved_for_traffic"
            continue
        await stop_runtime_stage(
            journal, job_id + ":" + name + "-cleanup", launch_id, launch_spec, runtime
        )
        await owned_disk(partial(routes.release_retired_endpoint, backend))
        outcomes[name] = "stopped"
    return outcomes


def existing_baseline(journal: ProducerStages, spec: ProducerInput) -> RuntimeLaunchSpec:
    """Reuse the original completed launch identity; never relabel its image, model or source."""
    if spec.existing_baseline_stage is None:
        raise ValueError("existing baseline stage is required")
    state = journal.state(spec.existing_baseline_stage)
    frozen = json.loads(journal.artifacts.get(state.input))
    launch = RuntimeLaunchSpec.model_validate(frozen["specification"])
    load_launch(journal, state, launch)
    if (
        launch.profile.base_url != f"http://127.0.0.1:{spec.baseline.port}/v1"
        or launch.profile.served_model != spec.load.model
        or json.loads(launch.profile.engine_parameters_json)
        != spec.baseline.parameters.model_dump()
        or launch.revision.revision_id == spec.job_id + "-candidate"
    ):
        raise ValueError("existing baseline differs from frozen engine inputs")
    return launch


def cleanup_launch(
    journal: ProducerStages,
    spec: ProducerInput,
    name: str,
    launch_id: str,
) -> RuntimeLaunchSpec:
    """Validate immutable launch/build linkage even when the live model volume is damaged."""
    state = journal.state(launch_id)
    frozen = json.loads(journal.artifacts.get(state.input))
    launch = RuntimeLaunchSpec.model_validate(frozen["specification"])
    load_launch(journal, state, launch)
    build = journal.state(spec.job_id + ":build")
    if build.status != "completed" or build.output is None:
        raise ValueError("cleanup requires the completed immutable build receipt")
    image = RuntimeImage.model_validate_json(journal.artifacts.get(build.output))
    engine = spec.baseline if name == "baseline" else spec.candidate
    if (
        launch.image != image
        or image.specification.source_revision != spec.source_revision
        or launch.model != spec.model
        or launch.model_directory != spec.workspace / "models" / spec.model.digest()
        or launch.revision.revision_id != spec.job_id + "-" + name
        or launch.profile.base_url != f"http://127.0.0.1:{engine.port}/v1"
        or launch.profile.served_model != spec.load.model
        or json.loads(launch.profile.engine_parameters_json) != engine.parameters.model_dump()
        or frozen.get("model_stage_id") != spec.job_id + ":model"
        or frozen.get("build_stage_id") != spec.job_id + ":build"
    ):
        raise ValueError("cleanup launch differs from frozen producer inputs")
    return launch
