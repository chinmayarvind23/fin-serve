"""Register recomputed benchmark/GPU artifacts bound to an observed managed runtime start."""

import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from finserve.benchmark.gpu import TelemetrySample, aggregate
from finserve.benchmark.runner import validate_evidence
from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.managed_runtime import RuntimeLaunchSpec, RuntimeReceipt
from finserve.contracts.performance import PerformanceCollectionSpec
from finserve.evaluation.quality import unique_object
from finserve.http_ownership import HTTPClosureError
from finserve.registry.annotations import Annotation, AnnotationStore
from finserve.registry.artifacts import ArtifactRef
from finserve.registry.managed_runtime import DockerRuntime, owned_directory
from finserve.registry.metadata import RunBundle
from finserve.registry.model_assets import owned_disk
from finserve.registry.performance_collection import (
    collect_performance,
    evidence_bytes,
    verify_local_http_closure,
)
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.producer_tasks import (
    attempt_directory,
    declare_input,
    failed_after_drain,
    start_owned,
)
from finserve.registry.quality_collection import bounded_file
from finserve.registry.runtime_stages import load_launch


class PerformanceReceipt(ImmutableModel):
    """Raw artifacts retain collector provenance separately from the engine image source."""

    kind: Literal["performance-receipt-v1"] = "performance-receipt-v1"
    specification: ArtifactRef
    runtime: RuntimeLaunchSpec
    launch: ArtifactRef
    before: RuntimeReceipt
    after: RuntimeReceipt
    run: RunBundle
    environment: ArtifactRef
    telemetry: ArtifactRef
    status: ArtifactRef
    gpu_summary: ArtifactRef


def document(path: Path, maximum_bytes: int) -> dict[str, Any]:
    """Reject duplicate evidence keys instead of silently choosing one contradictory value."""
    value = json.loads(bounded_file(path, maximum_bytes), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("performance document must be an object")
    return cast(dict[str, Any], value)


def verify_experiment(directory: Path, spec: PerformanceCollectionSpec) -> tuple[float, float]:
    """Recompute serving and GPU summaries while keeping collector and image revisions distinct."""
    if evidence_bytes(directory) > spec.maximum_raw_bytes:
        raise ValueError("performance artifacts exceed collection limit")
    verify_local_http_closure(directory)
    manifest = validate_evidence(directory / "run")
    raw_manifest = document(directory / "run" / "manifest.json", spec.maximum_raw_bytes)
    document(directory / "run" / "summary.json", spec.maximum_raw_bytes)
    environment = document(directory / "environment.json", spec.maximum_raw_bytes)
    status = document(directory / "experiment-status.json", spec.maximum_raw_bytes)
    if (
        manifest.configuration != spec.configuration
        or manifest.workload.digest() != spec.workload.digest()
        or raw_manifest["url"] != spec.endpoint()
        or environment["configuration"] != spec.configuration.model_dump()
        or environment["workload_hash"] != spec.workload.digest()
        or not re.fullmatch(r"[0-9a-f]{40}", environment["git_sha"])
        or environment["git_sha"] != spec.collector_revision
        or environment["git_status"] != []
        or not isinstance(environment["git_status"], list)
        or any(not isinstance(item, str) for item in cast(list[object], environment["git_status"]))
        or status["status"] != "completed"
    ):
        raise ValueError("performance artifacts differ from frozen collection")
    epoch, monotonic, drift = (
        float(environment["clock_epoch_anchor_s"]),
        float(environment["clock_monotonic_anchor_s"]),
        float(status["clock_drift_seconds"]),
    )
    if not all(math.isfinite(value) for value in (epoch, monotonic, drift)):
        raise ValueError("performance clock mapping is nonfinite")
    samples = [
        TelemetrySample.model_validate(json.loads(line, object_pairs_hook=unique_object))
        for line in bounded_file(directory / "gpu.jsonl", spec.maximum_raw_bytes).splitlines()
    ]
    start, end = (
        epoch + manifest.measured_started_s - monotonic,
        epoch + manifest.measured_finished_s - monotonic,
    )
    report = aggregate(samples, start, end)
    if abs(drift) > 0.1:
        report["average_gpu_utilization_percent"] = None
        report["clock_warning"] = "wall/monotonic clock drift exceeded 100ms"
    if document(directory / "gpu-summary.json", spec.maximum_raw_bytes) != report:
        raise ValueError("GPU summary differs from measured raw samples")
    return start, end


def verify_observations(receipt: PerformanceReceipt, launch: RuntimeReceipt) -> None:
    """Both probes must surround collection on the same exact immutable runtime start."""
    for observation in (receipt.before, receipt.after):
        if (
            observation.specification_sha256 != launch.specification_sha256
            or observation.attempt_id != launch.attempt_id
            or observation.container_id != launch.container_id
            or observation.container_started_at != launch.container_started_at
            or observation.output_directory != launch.output_directory
        ):
            raise ValueError("performance observations refer to different runtime starts")
    if receipt.before.observed_at > receipt.after.observed_at:
        raise ValueError("runtime observations moved backwards")
    if receipt.runtime.digest() != launch.specification_sha256:
        raise ValueError("performance runtime differs from launch receipt")


def load_performance_receipt(journal: ProducerStages, reference: ArtifactRef) -> PerformanceReceipt:
    """Completed replay reconstructs CAS bytes and summaries without offering another request."""
    receipt = PerformanceReceipt.model_validate_json(journal.artifacts.get(reference))
    spec = PerformanceCollectionSpec.model_validate_json(
        journal.artifacts.get(receipt.specification)
    )
    launch = RuntimeReceipt.model_validate_json(journal.artifacts.get(receipt.launch))
    verify_observations(receipt, launch)
    if receipt.runtime.profile != spec.profile or receipt.runtime.revision != spec.revision:
        raise ValueError("performance runtime differs from collection specification")
    if (
        journal.registry.run(receipt.run.run_id) != receipt.run
        or receipt.run.revision_id != spec.revision.revision_id
    ):
        raise ValueError("performance receipt differs from registered run")
    references = (
        receipt.run.manifest,
        receipt.run.requests,
        receipt.run.summary,
        receipt.environment,
        receipt.telemetry,
        receipt.status,
        receipt.gpu_summary,
    )
    if sum(item.size_bytes for item in references) > spec.maximum_raw_bytes:
        raise ValueError("performance artifact references exceed collection limit")
    with tempfile.TemporaryDirectory(prefix="finserve-performance-") as temporary:
        directory = Path(temporary)
        (directory / "run").mkdir()
        for name, artifact in (
            ("run/manifest.json", receipt.run.manifest),
            ("run/requests.jsonl", receipt.run.requests),
            ("run/summary.json", receipt.run.summary),
            ("environment.json", receipt.environment),
            ("gpu.jsonl", receipt.telemetry),
            ("experiment-status.json", receipt.status),
            ("gpu-summary.json", receipt.gpu_summary),
        ):
            (directory / name).write_bytes(journal.artifacts.get(artifact))
        start, end = verify_experiment(directory, spec)
        if not receipt.before.observed_at <= start <= end <= receipt.after.observed_at:
            raise ValueError("performance window falls outside runtime observations")
    return receipt


def register_managed_gpu(journal: ProducerStages, stage_id: str) -> Annotation:
    """Bind collector/GPU provenance through a verified runtime receipt instead of equating SHAs."""
    state = journal.state(stage_id)
    if state.status != "completed" or state.output is None:
        raise ValueError("managed GPU annotation requires a completed performance stage")
    receipt = load_performance_receipt(journal, state.output)
    frozen = json.loads(journal.artifacts.get(state.input))
    spec = PerformanceCollectionSpec.model_validate_json(
        journal.artifacts.get(receipt.specification)
    )
    if (
        frozen.get("kind") != "performance-collection-v1"
        or frozen.get("specification") != json.loads(spec.canonical())
        or frozen.get("launch") != receipt.launch.model_dump()
        or journal.registry.revision(spec.revision.revision_id) != receipt.runtime.revision
    ):
        raise ValueError("managed GPU stage or image revision linkage changed")
    report = json.loads(journal.artifacts.get(receipt.gpu_summary))
    inputs = [
        journal.artifacts.get(reference)
        for reference in (
            state.output,
            receipt.launch,
            receipt.specification,
            receipt.environment,
            receipt.telemetry,
            receipt.status,
            receipt.gpu_summary,
        )
    ]
    return AnnotationStore(journal.registry, journal.artifacts).save(
        receipt.run.run_id, "gpu", report, inputs
    )


async def performance_stage(
    journal: ProducerStages,
    stage_id: str,
    launch_stage_id: str,
    specification: PerformanceCollectionSpec,
    workspace: Path,
    client: httpx.AsyncClient,
    runtime: DockerRuntime,
) -> PerformanceReceipt:
    """Freeze launch linkage, collect once, recheck the same runtime, then register evidence."""
    spec = PerformanceCollectionSpec.model_validate_json(specification.model_dump_json())
    launch_state = await owned_disk(lambda: journal.state(launch_stage_id))
    frozen_launch = json.loads(await owned_disk(lambda: journal.artifacts.get(launch_state.input)))
    runtime_spec = RuntimeLaunchSpec.model_validate(frozen_launch["specification"])
    launch = await owned_disk(lambda: load_launch(journal, launch_state, runtime_spec))
    if runtime_spec.profile != spec.profile or runtime_spec.revision != spec.revision:
        raise ValueError("performance runtime differs from completed launch")
    assert launch_state.output is not None
    launch_ref = launch_state.output
    workspace = await owned_disk(lambda: owned_directory(workspace, create=True))
    state = await owned_disk(
        lambda: declare_input(
            journal,
            stage_id,
            {
                "kind": "performance-collection-v1",
                "specification": json.loads(spec.canonical()),
                "workspace": str(workspace),
                "launch_stage_id": launch_stage_id,
                "launch": launch_ref.model_dump(),
            },
        )
    )
    if state.status == "completed":
        assert state.output is not None
        output = state.output
        receipt = await owned_disk(lambda: load_performance_receipt(journal, output))
        if receipt.specification.sha256 != spec.digest() or receipt.launch != launch_ref:
            raise ValueError("performance completed receipt differs from stage input")
        return receipt
    running = await start_owned(journal, stage_id)
    assert running.attempt_id is not None
    try:
        directory = await owned_disk(lambda: attempt_directory(workspace, running))
        before = await runtime.observe(runtime_spec, launch, client)
        await collect_performance(spec, directory)
        after = await runtime.observe(runtime_spec, launch, client)

        def publish() -> PerformanceReceipt:
            """Verify bytes before linking the registered run and immutable attempt receipt."""
            start, end = verify_experiment(directory, spec)
            bundle = journal.registry.register_run(
                directory / "run", journal.artifacts, spec.revision
            )
            receipt = PerformanceReceipt(
                specification=journal.artifacts.put(spec.canonical().encode()),
                runtime=runtime_spec,
                launch=launch_ref,
                before=before,
                after=after,
                run=bundle,
                environment=journal.artifacts.put(
                    bounded_file(directory / "environment.json", spec.maximum_raw_bytes)
                ),
                telemetry=journal.artifacts.put(
                    bounded_file(directory / "gpu.jsonl", spec.maximum_raw_bytes)
                ),
                status=journal.artifacts.put(
                    bounded_file(directory / "experiment-status.json", spec.maximum_raw_bytes)
                ),
                gpu_summary=journal.artifacts.put(
                    bounded_file(directory / "gpu-summary.json", spec.maximum_raw_bytes)
                ),
            )
            verify_observations(receipt, launch)
            if not before.observed_at <= start <= end <= after.observed_at:
                raise ValueError("performance window falls outside runtime observations")
            reference = journal.artifacts.put(receipt.model_dump_json().encode())
            assert running.attempt_id is not None
            journal.finish(stage_id, running.attempt_id, reference)
            return receipt

        return await owned_disk(publish)
    except HTTPClosureError:
        raise
    except BaseException as exc:
        await failed_after_drain(journal, stage_id, running.attempt_id, type(exc).__name__)
        raise
