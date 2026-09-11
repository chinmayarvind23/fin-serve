"""Run producer task dispatch through actual collectors with explicit synthetic build scope."""

import asyncio
import json
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
from finserve.contracts.rollout import RolloutSettings
from finserve.evaluation.quality import default_suite
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.produced_release import register_produced_release
from finserve.registry.producer_pipeline import (
    ProducerExecution,
    execution_input,
    freeze_stage,
    verify_borrowed_baseline,
)
from finserve.registry.producer_rollout import rollout_stage
from finserve.registry.producer_runtime import (
    ProducerEngine,
    ProducerInput,
    ProducerStep,
    cleanup_unserved,
    freeze_producer,
    prepared_cohorts,
    produce_step,
)
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.release_gate import evaluate_gate
from finserve.registry.runtime_build import RuntimeBuildSpec, RuntimeImage
from finserve.reliability.promotion import PromotionPolicy
from finserve.reliability.rollback import ControlConflict, DeploymentStore
from finserve.reliability.warm_routes import WarmRouteStore

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("rollout", [False, True])
async def test_producer_derives_actual_build_and_runs_collectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rollout: bool,
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
            policy=PromotionPolicy(
                version="synthetic-orchestration-fixture",
                minimum_requests_per_second_ratio=0.001,
                minimum_tokens_per_second_ratio=0.001,
                maximum_client_ttft_ratio=1000,
                maximum_e2e_p95_ratio=1000,
            ),
            repository=REPOSITORY,
            workspace=tmp_path / "work",
        )
        assert freeze_producer(journal, spec) == spec.job_id
        routes = WarmRouteStore(tmp_path / "routes.sqlite")
        control = DeploymentStore(tmp_path / "control.sqlite")
        execution = ProducerExecution(
            producer=spec,
            routes=routes.path,
            control=control.path,
            rollout=RolloutSettings(
                traffic_url="http://traffic",
                maximum_probes=3,
                interval_seconds=0.1,
                slow_probe_seconds=5,
            ),
        )
        request_file = tmp_path / "producer.json"
        request_file.write_text(execution.model_dump_json())
        monkeypatch.setenv("FINSERVE_PRODUCER_REQUEST", str(request_file))
        monkeypatch.setenv(
            "FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.sqlite")
        )
        monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
        assert await asyncio.to_thread(freeze_stage) == spec.job_id
        with pytest.raises(RegistryConflict):
            freeze_producer(journal, spec.model_copy(update={"expected_generation": 1}))
        calls = 0

        def response(request: httpx.Request) -> httpx.Response:
            """Serve original model fixture bytes or real SSE protocol responses."""
            nonlocal calls
            calls += 1
            if request.url.path.endswith("/config.json"):
                return httpx.Response(200, content=b'{"model_type":"fixture"}')
            if rollout and request.method == "POST":
                prompt = json.loads(request.content).get("prompt", "")
                # Deterministic oracle is only a transport fixture, never model quality evidence.
                answer = next(
                    (case.expected for case in spec.suite.cases if case.prompt == prompt), "hello"
                )
                headers = {"content-type": "text/event-stream"}
                if request.url.host == "traffic":
                    route = routes.snapshot(spec.deployment_id)
                    headers.update(
                        {
                            "x-finserve-revision": route.revision_id,
                            "x-finserve-revision-digest": route.revision_digest,
                            "x-finserve-route-generation": str(route.generation),
                        }
                    )
                event = json.dumps(
                    {
                        "choices": [{"index": 0, "text": answer, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 1},
                    }
                )
                return httpx.Response(
                    200, headers=headers, content="data: " + event + "\n\ndata: [DONE]\n\n"
                )
            return handler(request)

        def build(
            repository: Path, build_spec: RuntimeBuildSpec, manifest: ModelManifest, output: Path
        ) -> RuntimeImage:
            """Only Docker is a fixture; its expected model digest comes from fetched file bytes."""
            assert build_spec.model_manifest_sha256 == manifest.digest()
            changed = build_spec.source_revision != template.image.specification.source_revision
            return template.image.model_copy(
                update={
                    "specification": build_spec,
                    "image_local_id": "sha256:" + "d" * 64
                    if changed
                    else template.image.image_local_id,
                    "image_manifest_digest": "sha256:" + "d" * 64
                    if changed
                    else template.image.image_manifest_digest,
                }
            )

        def client(config: RunConfig) -> httpx.AsyncClient:
            """Use the same deterministic protocol fixture on the isolated performance loop."""
            return httpx.AsyncClient(transport=httpx.MockTransport(response))

        def git(arguments: list[str], **kwargs: object) -> str:
            """Mark collector provenance synthetic; this test is not a real clean-clone run."""
            return "" if arguments[1] == "status" else "c" * 40 + "\n"

        monkeypatch.setattr("finserve.registry.producer_tasks.build_runtime", build)

        def traffic_client(settings: RolloutSettings) -> httpx.AsyncClient:
            """Exercise full rollout tasks with route-aware SSE, without claiming a live gateway."""
            return httpx.AsyncClient(transport=httpx.MockTransport(response))

        monkeypatch.setattr("finserve.registry.producer_rollout.traffic_client", traffic_client)
        monkeypatch.setattr("finserve.benchmark.experiment.benchmark_client", client)
        monkeypatch.setattr("finserve.benchmark.experiment.subprocess.check_output", git)
        monkeypatch.setattr(
            "finserve.benchmark.experiment.collect",
            lambda: TelemetrySample(
                epoch_s=time.time(), collection_seconds=0, devices=[], error="FixtureNoGPU"
            ),
        )

        class CohortDaemon(Daemon):
            """Fixture listeners have independently declared ports and container identities."""

            def inspection(self, attempt: str, directory: Path) -> dict[str, Any]:
                """Retain existing ownership fields and report this cohort's declared listener."""
                result = super().inspection(attempt, directory)
                name = self.spec.revision.revision_id
                identity, port = {
                    "producer-baseline": ("e", "9000"),
                    "producer-candidate": ("f", "9001"),
                    "second-candidate": ("d", "9002"),
                }[name]
                result["Id"] = identity * 64
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
            daemons: dict[str, CohortDaemon] = {}
            for name, runtime_spec in zip(("baseline", "candidate"), runtimes, strict=True):
                daemons[name] = CohortDaemon(runtime_spec)
                runtime = DockerRuntime(daemons[name])
                for action in ("launch", "quality", "performance"):
                    await produce_step(
                        journal, spec.job_id, cast(ProducerStep, name + "_" + action), http, runtime
                    )
            before = calls
            await produce_step(journal, spec.job_id, "freeze", http, DockerRuntime())
            assert register_produced_release(journal, spec.job_id + ":release-plan") == spec.job_id
            assert evaluate_gate(registry, journal.artifacts, spec.job_id).status == (
                "approved" if rollout else "rejected"
            )
            assert calls == before
            if rollout:
                for action in ("prepare", "deploy", "acknowledge", "probation"):
                    assert (
                        await asyncio.to_thread(rollout_stage, spec.job_id, action) == spec.job_id
                    )
                assert (
                    control.deployment(spec.deployment_id).known_good_revision
                    == "producer-candidate"
                )
                assert (
                    await asyncio.to_thread(rollout_stage, spec.job_id, "probation") == spec.job_id
                )
                second = ProducerInput.model_validate(
                    {
                        **spec.model_dump(),
                        "job_id": "second",
                        "expected_generation": 1,
                        "existing_baseline_stage": "producer:candidate-launch",
                        "baseline": spec.candidate,
                        "candidate": ProducerEngine(port=9002, parameters=VLLMParameters()),
                        "source_revision": "d" * 40,
                        "workspace": tmp_path / "second-work",
                    }
                )
                request_file.write_text(
                    execution.model_copy(update={"producer": second}).model_dump_json()
                )
                assert await asyncio.to_thread(freeze_stage) == "second"
                assert await cleanup_unserved(
                    journal, "second", routes, control, DockerRuntime()
                ) == {
                    "baseline": "borrowed_baseline",
                    "candidate": "not_launched",
                }
                for step in ("fetch", "build", "freeze"):
                    await produce_step(journal, "second", step, http, DockerRuntime())
                second_plan, second_runtimes = prepared_cohorts(journal, second)
                assert second_runtimes[0] == runtimes[1]
                assert second_runtimes[1].image != runtimes[1].image
                assert second_plan.baseline.performance.revision == runtimes[1].revision
                assert (
                    second_plan.baseline.performance.configuration.revision == spec.source_revision
                )
                original_launch = journal.state("producer:candidate-launch")
                baseline_creates = sum(item[2] == "create" for item in daemons["candidate"].calls)
                fresh = CohortDaemon(second_runtimes[1])
                for name, daemon in (("baseline", daemons["candidate"]), ("candidate", fresh)):
                    for action in ("launch", "quality", "performance"):
                        await produce_step(
                            journal,
                            "second",
                            cast(ProducerStep, name + "_" + action),
                            http,
                            DockerRuntime(daemon),
                        )
                assert journal.history("second:baseline-launch") == []
                assert journal.state("producer:candidate-launch") == original_launch
                assert (
                    sum(item[2] == "create" for item in daemons["candidate"].calls)
                    == baseline_creates
                )
                assert register_produced_release(journal, "second:release-plan") == "second"
                assert evaluate_gate(registry, journal.artifacts, "second").status == "approved"
                for action in ("prepare", "deploy", "acknowledge", "probation"):
                    assert await asyncio.to_thread(rollout_stage, "second", action) == "second"
                current = control.deployment(spec.deployment_id)
                assert current.generation == 2 and current.known_good_revision == "second-candidate"
                assert await cleanup_unserved(
                    journal, "second", routes, control, DockerRuntime()
                ) == {
                    "baseline": "borrowed_baseline",
                    "candidate": "preserved_for_traffic",
                }
                assert daemons["candidate"].container is not None
                # A new collection task cannot continue against the now superseded baseline.
                with pytest.raises(ControlConflict, match="current stable"):
                    verify_borrowed_baseline(journal, execution_input(journal, "second"))
            else:
                with pytest.raises(ValueError, match="approved canonical"):
                    await asyncio.to_thread(rollout_stage, spec.job_id, "prepare")
                assert routes.history(spec.deployment_id) == []

            def cleanup_command(
                arguments: list[str], directory: Path, output: Path, timeout: float
            ) -> None:
                """Dispatch independently identified fixture containers by owned directory."""
                name = "baseline" if "baseline" in directory.parts else "candidate"
                daemons[name](arguments, directory, output, timeout)

            routes = WarmRouteStore(tmp_path / "routes.sqlite")
            control = DeploymentStore(tmp_path / "control.sqlite")
            await asyncio.to_thread((runtimes[0].model_directory / "config.json").unlink)
            cleaned = await cleanup_unserved(
                journal, spec.job_id, routes, control, DockerRuntime(cleanup_command)
            )
            expected = "preserved_for_traffic" if rollout else "stopped"
            assert cleaned == {"baseline": expected, "candidate": expected}
            assert all((daemon.container is None) != rollout for daemon in daemons.values())
            assert (
                await cleanup_unserved(
                    journal, spec.job_id, routes, control, DockerRuntime(cleanup_command)
                )
                == cleaned
            )
    finally:
        registry.close()
