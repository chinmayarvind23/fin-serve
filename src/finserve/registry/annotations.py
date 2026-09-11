"""Immutable, recomputed GPU and correctness observations for the read-only evidence catalog."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Column, ForeignKey, String, Table, Text, insert, select
from sqlalchemy.exc import IntegrityError

from finserve.benchmark.gpu import TelemetrySample, aggregate
from finserve.contracts.deployment import ImmutableModel
from finserve.evaluation import quality as quality_module
from finserve.evaluation.quality import GoldenSuite, QualityConfig, evaluate_quality
from finserve.registry.artifacts import ArtifactRef, ArtifactStore, LocalArtifactStore
from finserve.registry.metadata import Registry, RegistryConflict, canonical_json, runs, schema

annotations = Table(
    "finserve_evidence_annotations",
    schema,
    Column("id", String(160), primary_key=True),
    Column("run_id", String(128), ForeignKey(runs.c.id), nullable=False),
    Column("digest", String(64), nullable=False),
    Column("payload", Text, nullable=False),
)


class Annotation(ImmutableModel):
    """Reports retain input references; a catalog association grants no release authorization."""

    run_id: str
    kind: Literal["gpu", "quality"]
    report: ArtifactRef
    inputs: tuple[ArtifactRef, ...]


class AnnotationStore:
    """Trusted offline registration verifies relationships before exposing any derived metric."""

    def __init__(self, registry: Registry, artifacts: ArtifactStore) -> None:
        """Create the additive table for registries initialized before this module was imported."""
        self.registry, self.artifacts = registry, artifacts
        annotations.create(registry.engine, checkfirst=True)

    def read(self, run_id: str, kind: Literal["gpu", "quality"]) -> Annotation | None:
        """Absent telemetry remains absent rather than turning into a zero utilization claim."""
        with self.registry.engine.connect() as connection:
            payload = connection.execute(
                select(annotations.c.payload).where(annotations.c.id == f"{run_id}:{kind}")
            ).scalar_one_or_none()
        if payload is None:
            return None
        return Annotation.model_validate_json(str(payload))

    def save(
        self,
        run_id: str,
        kind: Literal["gpu", "quality"],
        report: dict[str, Any],
        inputs: list[bytes],
    ) -> Annotation:
        """Only identical retries can reuse a run/kind association; changed reports conflict."""
        self.registry.run(run_id)
        item = Annotation(
            run_id=run_id,
            kind=kind,
            report=self.artifacts.put(canonical_json(report).encode()),
            inputs=tuple(self.artifacts.put(value) for value in inputs),
        )
        payload = canonical_json(item.model_dump())
        try:
            with self.registry.engine.begin() as connection:
                connection.execute(
                    insert(annotations).values(
                        id=f"{run_id}:{kind}",
                        run_id=run_id,
                        digest=hashlib.sha256(payload.encode()).hexdigest(),
                        payload=payload,
                    )
                )
        except IntegrityError:
            if self.read(run_id, kind) != item:
                raise RegistryConflict("immutable annotation identity reused") from None
        return item

    def register_gpu(self, run_id: str, directory: Path) -> Annotation:
        """Recompute the measured-window integral using retained clocks and raw device samples."""
        raw = {
            name: bounded_file(directory / name)
            for name in (
                "environment.json",
                "experiment-status.json",
                "gpu.jsonl",
                "gpu-summary.json",
            )
        }
        bundle = self.registry.run(run_id)
        manifest = json.loads(self.artifacts.get(bundle.manifest))
        environment = json.loads(raw["environment.json"])
        status = json.loads(raw["experiment-status.json"])
        if (
            environment["configuration"] != manifest["configuration"]
            or environment["git_sha"] != manifest["configuration"]["revision"]
            or environment["git_status"] != []
            or status["status"] != "completed"
        ):
            raise ValueError("GPU experiment does not bind the registered configuration")
        drift = status["clock_drift_seconds"]
        if not math.isfinite(drift) or abs(drift) > 0.1:
            raise ValueError("GPU clock mapping is unreliable")
        shift = environment["clock_epoch_anchor_s"] - environment["clock_monotonic_anchor_s"]
        start, end = (manifest[f"measured_{label}_s"] + shift for label in ("started", "finished"))
        samples = [
            TelemetrySample.model_validate_json(line) for line in raw["gpu.jsonl"].splitlines()
        ]
        saved = json.loads(raw["gpu-summary.json"])
        report = aggregate(samples, start, end)
        for key, value in report.items():
            observed = saved.get(key)
            if isinstance(value, float) and isinstance(observed, (float, int)):
                # Relative tolerance on epoch timestamps could admit a different sampling window.
                equal = math.isclose(value, observed, rel_tol=0, abs_tol=1e-6)
            else:
                equal = value == observed
            if not equal:
                raise ValueError("GPU summary differs from raw measured-window evidence")
        return self.save(run_id, "gpu", report, list(raw.values()))

    def register_quality(self, run_id: str, directory: Path, reference: Path) -> Annotation:
        """Regrade retained outputs and verify source/model/configuration and reference linkage."""
        candidate = quality_inputs(directory)
        baseline = quality_inputs(reference)
        manifest, ref_manifest = (
            json.loads(candidate["manifest.json"]),
            json.loads(baseline["manifest.json"]),
        )
        configuration = json.loads(self.artifacts.get(self.registry.run(run_id).manifest))[
            "configuration"
        ]
        for key in ("model", "model_revision", "tokenizer_revision", "engine", "engine_config"):
            if manifest[key] != configuration[key]:
                raise ValueError("quality configuration differs from registered run")
        if manifest["git_sha"] != configuration["revision"] or manifest["git_status"]:
            raise ValueError("quality source is not the registered clean revision")
        suite = GoldenSuite.model_validate(manifest["suite"])
        if suite.digest() != manifest["suite_hash"] or suite.digest() != ref_manifest["suite_hash"]:
            raise ValueError("quality suite mismatch")
        if GoldenSuite.model_validate(ref_manifest["suite"]).digest() != suite.digest():
            raise ValueError("reference embedded suite mismatch")
        if (
            manifest["evaluator_sha256"] != ref_manifest["evaluator_sha256"]
            or manifest["evaluator_sha256"] not in evaluator_hashes()
        ):
            raise ValueError("quality evaluator mismatch")
        if (
            directory.resolve() != reference.resolve()
            and manifest["reference_manifest_sha256"]
            != hashlib.sha256(baseline["manifest.json"]).hexdigest()
        ):
            raise ValueError("quality reference linkage mismatch")
        for values, identity in ((candidate, manifest), (baseline, ref_manifest)):
            if (
                identity["status"] != "completed"
                or hashlib.sha256(values["answers.json"]).hexdigest() != identity["answers_sha256"]
            ):
                raise ValueError("incomplete or changed quality answers")
            rows = [json.loads(line) for line in values["requests.jsonl"].splitlines()]
            ids = {case.case_id for case in suite.cases}
            if len(rows) != len(ids) or {row["case_id"] for row in rows} != ids:
                raise ValueError("quality request population mismatch")
            if {row["case_id"]: row["output"] for row in rows if "output" in row} != json.loads(
                values["answers.json"]
            ):
                raise ValueError("quality answers differ from raw requests")
        saved = json.loads(candidate["quality.json"])
        report = evaluate_quality(
            suite,
            json.loads(baseline["answers.json"]),
            json.loads(candidate["answers.json"]),
            QualityConfig(),
        )
        if any(saved.get(key) != value for key, value in report.items()):
            raise ValueError("quality report differs from recomputed outputs")
        report["reference_scope"] = (
            "self" if directory.resolve() == reference.resolve() else "external baseline"
        )
        report["quality_run_id"] = manifest["run_id"]
        return self.save(run_id, "quality", report, [*candidate.values(), *baseline.values()])


def evaluator_hashes() -> set[str]:
    """Only LF/CRLF checkout normalization is equivalent; any grader source change rejects."""
    source = Path(str(quality_module.__file__)).read_bytes().replace(b"\r\n", b"\n")
    return {hashlib.sha256(value).hexdigest() for value in (source, source.replace(b"\n", b"\r\n"))}


def bounded_file(path: Path) -> bytes:
    """Offline imports have the same finite artifact size as the evidence store."""
    with path.open("rb") as stream:
        data = stream.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("annotation input exceeds byte budget")
    return data


def quality_inputs(directory: Path) -> dict[str, bytes]:
    """Capture each input once so filesystem changes cannot alter a verified report afterward."""
    return {
        name: bounded_file(directory / name)
        for name in ("manifest.json", "answers.json", "requests.jsonl", "quality.json")
    }


def main() -> None:
    """Import a completed native experiment explicitly without assigning a deployment image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--reference-quality", type=Path)
    args = parser.parse_args()
    if (args.quality is None) != (args.reference_quality is None):
        parser.error("quality and reference-quality must be supplied together")
    registry, artifacts = Registry(args.database_url), LocalArtifactStore(args.artifact_root)
    try:
        bundle = registry.register_run(args.experiment / "run", artifacts)
        catalog = AnnotationStore(registry, artifacts)
        catalog.register_gpu(bundle.run_id, args.experiment)
        if args.quality is not None:
            catalog.register_quality(bundle.run_id, args.quality, args.reference_quality)
        print(json.dumps({"run_id": bundle.run_id, "registered": True}))
    finally:
        registry.close()


if __name__ == "__main__":
    main()
