"""A shared profile-aware evidence gate for Airflow and CI; it never deploys or rewrites runs."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from sqlalchemy.exc import SQLAlchemyError

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.evaluation.quality import GoldenSuite
from finserve.registry.artifacts import ArtifactRef, ArtifactStore, LocalArtifactStore
from finserve.registry.lifecycle import LifecycleSpec, materialize
from finserve.registry.metadata import Registry, canonical_json
from finserve.reliability.promotion import QualityEvidence, evaluate_promotion


class GateRequest(ImmutableModel):
    """Freeze actual profile artifact references separately from historical lifecycle schemas."""

    job_id: str = Field(min_length=1, max_length=128)
    baseline_profile: ArtifactRef
    candidate_profile: ArtifactRef


class GateOutcome(ImmutableModel):
    """Approval describes evidence consistency and thresholds, never deployment authority."""

    job_id: str = Field(min_length=1, max_length=128)
    specification_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_input_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    candidate_run_id: str = Field(min_length=1, max_length=128)
    status: Literal["approved", "rejected", "invalid_evidence"]
    decision_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reasons: tuple[str, ...] = ()
    scope: Literal["canonical-profile-evidence-gate-v1"] = "canonical-profile-evidence-gate-v1"

    @model_validator(mode="after")
    def consistent_decision(self) -> Self:
        """Typed approvals identify a decision and frozen input without rejection reasons."""
        if self.status != "invalid_evidence" and (
            self.decision_digest is None or self.gate_input_digest is None
        ):
            raise ValueError("measured outcome requires decision and gate input identities")
        if (self.status == "approved") == bool(self.reasons):
            raise ValueError("outcome reasons must agree with approval status")
        return self

    def exit_code(self) -> int:
        """CI distinguishes a measured rejection from absent or inconsistent evidence."""
        return {"approved": 0, "rejected": 2, "invalid_evidence": 3}[self.status]


def freeze_gate_request(registry: Registry, request: GateRequest) -> None:
    """Bind both immutable profile references before any threshold evaluation starts."""
    registry.freeze_gate_input(request.job_id, canonical_json(request.model_dump()))


def evaluate_gate(registry: Registry, artifacts: ArtifactStore, job_id: str) -> GateOutcome:
    """Recompute raw artifacts on every invocation, including a previously approved job."""
    specification_raw = registry.specification(job_id)
    specification = LifecycleSpec.model_validate_json(specification_raw)
    common = {
        "job_id": job_id,
        "specification_digest": hashlib.sha256(specification.canonical().encode()).hexdigest(),
        "candidate_run_id": specification.candidate_run_id,
    }
    try:
        if specification.gate_mode != "canonical-profile-v1":
            raise ValueError("historical drill is not a canonical release")
        request = GateRequest.model_validate_json(registry.gate_input(job_id))
        common["gate_input_digest"] = hashlib.sha256(
            canonical_json(request.model_dump()).encode()
        ).hexdigest()
        if request.job_id != job_id:
            raise ValueError("gate job identity mismatch")
        profiles = [
            ServingProfileV1.model_validate_json(artifacts.get(reference))
            for reference in (request.baseline_profile, request.candidate_profile)
        ]
        baseline = registry.run(specification.baseline_run_id)
        candidate = registry.run(specification.candidate_run_id)
        if (
            baseline.revision_id != specification.expected_revision
            or candidate.revision_id != specification.target.revision_id
        ):
            raise ValueError("run revision does not match lifecycle input")
        profiles[0].verify_revision(registry.revision(specification.expected_revision))
        profiles[1].verify_revision(specification.target)
        for bundle, profile in zip((baseline, candidate), profiles, strict=True):
            manifest = json.loads(artifacts.get(bundle.manifest))
            suffix = (
                "/chat/completions"
                if manifest["configuration"].get("request_api", "completions") == "chat"
                else "/completions"
            )
            if (
                manifest.get("url") != profile.base_url + suffix
                or manifest["configuration"]["model"] != profile.served_model
            ):
                raise ValueError("measured endpoint or served model differs from profile")
        if registry.revision(specification.target.revision_id) != specification.target:
            raise ValueError("target differs from registered revision")
        quality = QualityEvidence.model_validate_json(artifacts.get(specification.quality))
        suite = GoldenSuite.model_validate_json(artifacts.get(specification.suite))
        with tempfile.TemporaryDirectory(prefix="finserve-release-gate-") as temporary:
            directory = Path(temporary)
            materialize(baseline, artifacts, directory / "baseline")
            materialize(candidate, artifacts, directory / "candidate")
            decision = evaluate_promotion(
                directory / "baseline",
                directory / "candidate",
                quality,
                suite,
                specification.policy,
                specification.target,
            )
        decision_digest = registry.record_decision(decision, candidate.run_id)
        status = (
            "approved"
            if decision.approved
            else ("rejected" if decision.evidence_digest is not None else "invalid_evidence")
        )
        outcome = GateOutcome.model_validate(
            {
                **common,
                "status": status,
                "decision_digest": decision_digest,
                "reasons": decision.rejection_reasons,
            }
        )
    except (ValueError, KeyError, OSError, TypeError):
        outcome = GateOutcome.model_validate(
            {
                **common,
                "status": "invalid_evidence",
                "reasons": ("missing_or_invalid_profile_evidence",),
            }
        )
    registry.record_gate_outcome(job_id, canonical_json(outcome.model_dump()))
    return outcome


def run_cli(arguments: list[str] | None = None) -> int:
    """Read only server-configured storage, write a reviewable outcome and return a stable code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parsed = parser.parse_args(arguments)
    output = Path(parsed.output).resolve()
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        raise ValueError("gate evidence must be outside the source repository")
    registry: Registry | None = None
    try:
        try:
            registry = Registry(os.environ["FINSERVE_REGISTRY_URL"])
            artifacts = LocalArtifactStore(Path(os.environ["FINSERVE_ARTIFACT_ROOT"]))
            outcome = evaluate_gate(registry, artifacts, parsed.job_id)
            document, code = outcome.model_dump(), outcome.exit_code()
        except (KeyError, ValueError, OSError, SQLAlchemyError):
            document, code = (
                {"status": "invalid_evidence", "reasons": ["gate_input_unavailable"]},
                3,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as destination:
            destination.write(json.dumps(document, sort_keys=True, indent=2, allow_nan=False))
        return code
    finally:
        if registry is not None:
            registry.close()


if __name__ == "__main__":
    raise SystemExit(run_cli())
