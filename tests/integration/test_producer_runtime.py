"""Run producer task dispatch through actual collectors with explicit synthetic build scope."""

import time
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from test_managed_runtime import Daemon, handler, specification

from finserve.benchmark.gpu import TelemetrySample
from finserve.benchmark.runner import RunConfig
from finserve.benchmark.workload import WorkItem, Workload
from finserve.contracts.model_assets import ModelManifest
from finserve.evaluation.quality import default_suite
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.produced_release import register_produced_release
from finserve.registry.producer_runtime import (
    ProducerEngine,
    ProducerInput,
    ProducerStep,
    freeze_producer,
    prepared_cohorts,
    produce_step,
)
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.release_gate import evaluate_gate
from finserve.registry.runtime_build import RuntimeBuildSpec, RuntimeImage
from finserve.reliability.promotion import PromotionPolicy

REPOSITORY = Path(__file__).resolve().parents[2]


async def test_producer_derives_actual_build_and_runs_collectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze->fetch->build->launch/collect->gate rejects wrong fixture answers without cutover."""
    template = specification(tmp_path)
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
        spec = ProducerInput(
            job_id="producer",
            deployment_id="fixture",
            expected_generation=0,
            source_revision=template.image.specification.source_revision,
            collector_revision="c" * 40,
            model=template.model,
            baseline=ProducerEngine(port=9000, parameters=VLLMParameters()),
            candidate=ProducerEngine(port=9001, parameters=VLLMParameters(enforce_eager=False)),
            workload=Workload(
                suite_id="fixture",
                version=1,
                items=(WorkItem(case_id="one", prompt="hello", max_tokens=1),),
            ),
            load=RunConfig(
                model="fixture", hardware="fixture-cpu", requests=4, warmup=1, concurrency=2
            ),
            suite=default_suite(),
            policy=PromotionPolicy(),
            repository=REPOSITORY,
            workspace=tmp_path / "work",
        )
        assert freeze_producer(journal, spec) == spec.job_id
        with pytest.raises(RegistryConflict):
            freeze_producer(journal, spec.model_copy(update={"expected_generation": 1}))
        calls = 0

        def response(request: httpx.Request) -> httpx.Response:
            """Serve original model fixture bytes or real SSE protocol responses."""
            nonlocal calls
            calls += 1
            if request.url.path.endswith("/config.json"):
                return httpx.Response(200, content=b'{"model_type":"fixture"}')
            return handler(request)

        def build(
            repository: Path, build_spec: RuntimeBuildSpec, manifest: ModelManifest, output: Path
        ) -> RuntimeImage:
            """Only Docker is a fixture; its expected model digest comes from fetched file bytes."""
            assert build_spec.model_manifest_sha256 == manifest.digest()
            return template.image.model_copy(update={"specification": build_spec})

        def client(config: RunConfig) -> httpx.AsyncClient:
            """Use the same deterministic protocol fixture on the isolated performance loop."""
            return httpx.AsyncClient(transport=httpx.MockTransport(response))

        def git(arguments: list[str], **kwargs: object) -> str:
            """Mark collector provenance synthetic; this test is not a real clean-clone run."""
            return "" if arguments[1] == "status" else "c" * 40 + "\n"

        monkeypatch.setattr("finserve.registry.producer_tasks.build_runtime", build)
        monkeypatch.setattr("finserve.benchmark.experiment.benchmark_client", client)
        monkeypatch.setattr("finserve.benchmark.experiment.subprocess.check_output", git)
        monkeypatch.setattr(
            "finserve.benchmark.experiment.collect",
            lambda: TelemetrySample(
                epoch_s=time.time(), collection_seconds=0, devices=[], error="FixtureNoGPU"
            ),
        )

        class CohortDaemon(Daemon):
            """The two fixture listeners have independently declared ports instead of legacy8060."""

            def inspection(self, attempt: str, directory: Path) -> dict[str, Any]:
                """Retain existing ownership fields and report this cohort's declared listener."""
                result = super().inspection(attempt, directory)
                port = "9000" if self.spec.revision.revision_id.endswith("baseline") else "9001"
                result["HostConfig"]["PortBindings"] = {
                    "8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": port}]
                }
                return result

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as http:
            with pytest.raises(ValueError, match="unsupported producer action"):
                await produce_step(
                    journal,
                    spec.job_id,
                    cast(ProducerStep, "unexpected_launch"),
                    http,
                    DockerRuntime(),
                )
            assert calls == 0
            for step in ("fetch", "build", "freeze"):
                assert (
                    await produce_step(
                        journal, spec.job_id, cast(ProducerStep, step), http, DockerRuntime()
                    )
                    == spec.job_id
                )
            plan, runtimes = prepared_cohorts(journal, spec)
            assert (
                plan.candidate.performance.revision.image_digest
                == template.image.image_manifest_digest
            )
            for name, runtime_spec in zip(("baseline", "candidate"), runtimes, strict=True):
                runtime = DockerRuntime(CohortDaemon(runtime_spec))
                for action in ("launch", "quality", "performance"):
                    await produce_step(
                        journal, spec.job_id, cast(ProducerStep, name + "_" + action), http, runtime
                    )
            before = calls
            await produce_step(journal, spec.job_id, "freeze", http, DockerRuntime())
            assert register_produced_release(journal, spec.job_id + ":release-plan") == spec.job_id
            assert evaluate_gate(registry, journal.artifacts, spec.job_id).status == "rejected"
            assert calls == before
    finally:
        registry.close()
