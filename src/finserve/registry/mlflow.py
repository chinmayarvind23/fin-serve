"""Optional MLflow evidence mirror; relational registry remains promotion truth."""

import importlib
import tempfile
from pathlib import Path
from typing import Protocol, cast

from finserve.registry.artifacts import ArtifactStore
from finserve.registry.metadata import Registry
from finserve.reliability.promotion import PromotionDecision


class RunInfo(Protocol):
    """The adapter reads only the stable run ID from the SDK's returned entity."""

    run_id: str


class MlflowRun(Protocol):
    """Keep the optional SDK isolated from core serving and control-plane imports."""

    info: RunInfo


class MlflowClient(Protocol):
    """The narrow client surface is exercised against a real local MLflow backend."""

    def create_run(self, experiment_id: str, tags: dict[str, str]) -> MlflowRun:
        """Create an experiment record linked to immutable registry evidence."""
        ...

    def log_param(self, run_id: str, key: str, value: str) -> None:
        """Record immutable workload/model/evidence identities as run parameters."""
        ...

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        """Log only metrics actually present in verified evidence."""
        ...

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        """Upload verified bytes through the supported tracking API."""
        ...

    def set_terminated(self, run_id: str, status: str) -> None:
        """A partial mirror failure is recorded as failed, never as a completed experiment."""
        ...


def client_for_uri(tracking_uri: str) -> MlflowClient:
    """Construct a URI-scoped optional client without mutating MLflow's process-wide global
    settings.
    """
    module = importlib.import_module("mlflow")
    return cast(MlflowClient, module.MlflowClient(tracking_uri=tracking_uri))


def mirror_run(
    registry: Registry,
    artifacts: ArtifactStore,
    run_id: str,
    decision: PromotionDecision,
    client: MlflowClient,
    experiment_id: str,
) -> str:
    """Mirror verification artifacts; the returned real MLflow ID does not grant promotion
    authority.
    """
    bundle = registry.run(run_id)
    registry.verify_decision_run(decision, run_id)
    if bundle.revision_id != decision.candidate_revision:
        raise ValueError("MLflow mirror decision belongs to a different revision")
    run = client.create_run(
        experiment_id, tags={"finserve.run_id": run_id, "finserve.scope": "evidence mirror"}
    )
    identity = run.info.run_id
    try:
        client.log_param(identity, "workload_sha256", bundle.workload_hash)
        client.log_param(identity, "registry_revision", bundle.revision_id or "host-observation")
        client.log_param(
            identity, "evidence_digest", decision.evidence_digest or "rejected-missing-evidence"
        )
        if decision.quality_parity is not None:
            client.log_metric(identity, "quality_parity", decision.quality_parity)
        if decision.candidate_accuracy is not None:
            client.log_metric(identity, "candidate_accuracy", decision.candidate_accuracy)
        with tempfile.TemporaryDirectory(prefix="finserve-mlflow-") as temporary:
            for filename, reference in (
                ("manifest.json", bundle.manifest),
                ("requests.jsonl", bundle.requests),
                ("summary.json", bundle.summary),
            ):
                path = Path(temporary) / filename
                path.write_bytes(artifacts.get(reference))
                client.log_artifact(identity, str(path), "verified_evidence")
            decision_path = Path(temporary) / "decision.json"
            decision_path.write_text(decision.model_dump_json())
            client.log_artifact(identity, str(decision_path), "verified_evidence")
        client.set_terminated(identity, "FINISHED")
    except Exception:
        client.set_terminated(identity, "FAILED")
        raise
    return identity
