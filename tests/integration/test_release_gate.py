"""CLI and Airflow release gates consume the same newly generated synthetic evidence fixture."""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from test_lifecycle import CandidateAdapter
from test_promotion import artifact, quality

from finserve.contracts.deployment import Revision
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.evaluation.quality import default_suite
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.lifecycle import LifecycleService, LifecycleSpec
from finserve.registry.metadata import Registry, RegistryConflict, canonical_json, digest_text
from finserve.registry.pipeline import (
    PipelineRequest,
    deploy_release_stage,
    deploy_stage,
    evaluate_release_stage,
    register_stage,
)
from finserve.registry.release_gate import (
    GateOutcome,
    GateRequest,
    evaluate_gate,
    freeze_gate_request,
    run_cli,
)


def prepare_gate(
    tmp_path: Path, wrong: bool = False, wrong_endpoint: bool = False
) -> tuple[Registry, LocalArtifactStore, GateRequest]:
    """Construct fresh image/profile-labelled fixtures; never relabel an existing measured run."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.db"))
    store = LocalArtifactStore(tmp_path / "objects")
    profiles: list[ServingProfileV1] = []
    revisions: list[Revision] = []
    for name, duration, port in (("baseline", 4, 9000), ("candidate", 2, 9001)):
        value = ServingProfileV1(
            engine="fixture",
            engine_version="0.0.0",
            engine_parameters_json="{}",
            model_revision="a" * 40,
            tokenizer_revision="b" * 40,
            model_manifest_sha256="c" * 64,
            tokenizer_manifest_sha256="d" * 64,
            base_url=f"http://127.0.0.1:{port}/v1",
            served_model="reference",
        )
        revision = Revision(
            revision_id=name,
            model_revision=value.model_revision,
            tokenizer_revision=value.tokenizer_revision,
            source_revision="e" * 40,
            image_digest="sha256:" + "f" * 64,
            config_digest=value.digest(),
            engine=value.engine,
            engine_config=value.engine_parameters_json,
        )
        artifact(tmp_path / name, name + "-fixture", duration, 0.1 * duration)
        path = tmp_path / name / "manifest.json"
        document = json.loads(path.read_text())
        document["url"] = value.base_url + "/completions"
        if wrong_endpoint and name == "candidate":
            document["url"] = "http://127.0.0.1:9002/v1/completions"
        for field, source in (
            ("model_revision", "model_revision"),
            ("tokenizer_revision", "tokenizer_revision"),
            ("revision", "source_revision"),
            ("image_digest", "image_digest"),
            ("config_digest", "config_digest"),
            ("engine", "engine"),
            ("engine_config", "engine_config"),
        ):
            document["configuration"][field] = getattr(revision, source)
        path.write_text(json.dumps(document))
        registry.register_run(tmp_path / name, store, revision)
        profiles.append(value)
        revisions.append(revision)
    evidence = quality().model_copy(
        update={
            "reference_model_revision": profiles[0].model_revision,
            "candidate_model_revision": profiles[1].model_revision,
        }
    )
    if wrong:
        evidence.candidate["margin"] = "0.20"
    specification = LifecycleSpec(
        job_id="release-fixture",
        deployment_id="fixture-service",
        expected_revision="baseline",
        expected_generation=0,
        baseline_run_id="baseline-fixture",
        candidate_run_id="candidate-fixture",
        quality=store.put(evidence.model_dump_json().encode()),
        suite=store.put(default_suite().model_dump_json().encode()),
        target=revisions[1],
        gate_mode="canonical-profile-v1",
    )
    registry.create_job(specification.job_id, specification.canonical())
    request = GateRequest(
        job_id=specification.job_id,
        baseline_profile=store.put(profiles[0].model_dump_json().encode()),
        candidate_profile=store.put(profiles[1].model_dump_json().encode()),
    )
    freeze_gate_request(registry, request)
    return registry, store, request


@pytest.mark.parametrize("wrong", [False, True])
def test_cli_and_airflow_share_persisted_gate_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong: bool,
) -> None:
    """Twice the fixture speed still fails both public entry points when financial quality fails."""
    registry, store, request = prepare_gate(tmp_path, wrong)
    try:
        monkeypatch.setenv("FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.db"))
        monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(store.root))
        outcome = evaluate_gate(registry, store, request.job_id)
        assert outcome.status == ("rejected" if wrong else "approved")
        payload = canonical_json(outcome.model_dump())
        assert registry.gate_outcome(digest_text(payload)) == payload
        assert run_cli(
            ["--job-id", request.job_id, "--output", str(tmp_path / "outcome.json")]
        ) == (2 if wrong else 0)
        assert json.loads((tmp_path / "outcome.json").read_text()) == outcome.model_dump(
            mode="json"
        )
        if wrong:
            with pytest.raises(RuntimeError):
                evaluate_release_stage(request.job_id)
            monkeypatch.setenv("FINSERVE_DEPLOYMENT_ADAPTER", "must_not_import:create")
            for deploy in (deploy_release_stage, deploy_stage):
                with pytest.raises(RuntimeError, match="evidence"):
                    deploy(request.job_id)
        else:
            assert evaluate_release_stage(request.job_id) == request.job_id
    finally:
        registry.close()


def test_missing_corrupt_profiles_and_frozen_input_replacement_fail_closed(tmp_path: Path) -> None:
    """Neither a previous approval nor an alternate profile reference bypasses byte verification."""
    registry, store, request = prepare_gate(tmp_path)
    try:
        assert evaluate_gate(registry, store, request.job_id).status == "approved"
        with pytest.raises(RegistryConflict):
            freeze_gate_request(
                registry, request.model_copy(update={"candidate_profile": request.baseline_profile})
            )
        path = (
            store.root
            / "sha256"
            / request.candidate_profile.sha256[:2]
            / request.candidate_profile.sha256
        )
        path.write_bytes(b"tampered")
        outcome = evaluate_gate(registry, store, request.job_id)
        assert outcome.status == "invalid_evidence" and outcome.exit_code() == 3
    finally:
        registry.close()


@pytest.mark.parametrize("mismatch", ["missing", "job", "run", "revision", "endpoint", "legacy"])
def test_inconsistent_release_identity_cannot_pass(tmp_path: Path, mismatch: str) -> None:
    """Consistent performance alone cannot authorize a different job, revision or endpoint."""
    registry, store, request = prepare_gate(tmp_path, wrong_endpoint=mismatch == "endpoint")
    try:
        specification = LifecycleSpec.model_validate_json(registry.specification(request.job_id))
        alternate = specification.model_copy(update={"job_id": "alternate"})
        if mismatch == "run":
            alternate = alternate.model_copy(update={"expected_revision": "candidate"})
        elif mismatch == "legacy":
            alternate = alternate.model_copy(update={"gate_mode": "legacy-drill"})
        elif mismatch == "revision":
            alternate = alternate.model_copy(
                update={
                    "target": specification.target.model_copy(update={"source_revision": "d" * 40})
                }
            )
        registry.create_job(alternate.job_id, alternate.canonical())
        if mismatch != "missing":
            frozen = (
                request
                if mismatch == "job"
                else request.model_copy(update={"job_id": alternate.job_id})
            )
            registry.freeze_gate_input(alternate.job_id, canonical_json(frozen.model_dump()))
        result = evaluate_gate(registry, store, alternate.job_id)
        assert result.status == "invalid_evidence"
        assert result.exit_code() == 3
    finally:
        registry.close()


async def test_interrupted_registration_cannot_fall_back_to_legacy_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before profile publication preserves required mode at each deployment entry point."""
    registry, store, request = prepare_gate(tmp_path)
    try:
        original = LifecycleSpec.model_validate_json(registry.specification(request.job_id))
        pending = original.model_copy(update={"job_id": "interrupted"})
        registry.create_job(pending.job_id, pending.canonical())
        adapter = CandidateAdapter(registry.revision(pending.expected_revision))
        state = await LifecycleService(registry, store).run(pending, adapter)
        assert state.status == "registered" and state.last_error == "ValueError"
        assert adapter.calls == []
        monkeypatch.setenv("FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.db"))
        monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(store.root))
        monkeypatch.setenv("FINSERVE_DEPLOYMENT_ADAPTER", "must_not_import:create")
        for deploy in (deploy_stage, deploy_release_stage):
            with pytest.raises(RuntimeError, match="evidence"):
                deploy(pending.job_id)
        with pytest.raises(RegistryConflict):
            await LifecycleService(registry, store).run(
                pending.model_copy(update={"gate_mode": "legacy-drill"}), adapter
            )
        freeze_gate_request(registry, request.model_copy(update={"job_id": pending.job_id}))
        completed = await LifecycleService(registry, store).run(pending, adapter)
        assert completed.status == "promoted" and len(adapter.calls) == 1
    finally:
        registry.close()


def test_profile_registration_and_verified_deployment_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate profile pairs before side effects and preserve release mode through task retries."""
    registry, store, gate = prepare_gate(tmp_path)
    try:
        spec = LifecycleSpec.model_validate_json(registry.specification(gate.job_id))
        for name, reference in (
            ("quality", spec.quality),
            ("suite", spec.suite),
            ("baseline-profile", gate.baseline_profile),
            ("candidate-profile", gate.candidate_profile),
        ):
            (tmp_path / (name + ".json")).write_bytes(store.get(reference))
        request = PipelineRequest(
            job_id=gate.job_id,
            deployment_id=spec.deployment_id,
            expected_generation=0,
            baseline_directory=tmp_path / "baseline",
            candidate_directory=tmp_path / "candidate",
            quality_file=tmp_path / "quality.json",
            suite_file=tmp_path / "suite.json",
            baseline_revision=registry.revision(spec.expected_revision),
            target_revision=spec.target,
            policy=spec.policy,
            baseline_profile_file=tmp_path / "baseline-profile.json",
            candidate_profile_file=tmp_path / "candidate-profile.json",
        )
        partial = request.model_dump()
        partial["candidate_profile_file"] = None
        with pytest.raises(ValueError, match="both canonical"):
            PipelineRequest.model_validate(partial)
        path = tmp_path / "request.json"
        path.write_text(request.model_dump_json())
        monkeypatch.setenv("FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.db"))
        monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(store.root))
        monkeypatch.setenv("FINSERVE_PIPELINE_REQUEST", str(path))
        assert register_stage() == gate.job_id
        assert register_stage() == gate.job_id
        adapter = CandidateAdapter(request.baseline_revision)
        module = types.ModuleType("profile_fixture_adapter")
        module.create = lambda: adapter  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setenv("FINSERVE_DEPLOYMENT_ADAPTER", module.__name__ + ":create")
        assert deploy_release_stage(gate.job_id) == gate.job_id
        assert len(adapter.calls) == 1
    finally:
        registry.close()


@pytest.mark.parametrize("configured", [False, True])
def test_cli_missing_storage_or_job_has_bounded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    """Operational setup failures return the same invalid-evidence code without leaking paths."""
    monkeypatch.delenv("FINSERVE_REGISTRY_URL", raising=False)
    if configured:
        monkeypatch.setenv("FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "empty.db"))
        monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(tmp_path / "objects"))
    output = tmp_path / "failure.json"
    assert run_cli(["--job-id", "missing", "--output", str(output)]) == 3
    assert json.loads(output.read_text()) == {
        "status": "invalid_evidence",
        "reasons": ["gate_input_unavailable"],
    }


def test_module_cli_and_output_exclusivity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The executable module preserves outcomes and refuses to overwrite previous evidence."""
    registry, store, request = prepare_gate(tmp_path)
    registry.close()
    monkeypatch.setenv("FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.db"))
    monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(store.root))
    output = tmp_path / "module.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "finserve.registry.release_gate",
            "--job-id",
            request.job_id,
            "--output",
            str(output),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == "approved"
    with pytest.raises(FileExistsError):
        run_cli(["--job-id", request.job_id, "--output", str(output)])
    with pytest.raises(ValueError, match="outside"):
        run_cli(
            ["--job-id", request.job_id, "--output", str(Path(__file__).parent / "invalid.json")]
        )


@pytest.mark.parametrize(
    "status,reasons,decision",
    [
        ("approved", (), None),
        ("approved", ("failure",), "a" * 64),
        ("rejected", (), "a" * 64),
        ("invalid_evidence", (), None),
    ],
)
def test_gate_outcome_rejects_contradictory_status(
    status: str, reasons: tuple[str, ...], decision: str | None
) -> None:
    """Serialized outcomes cannot accidentally describe rejection and approval simultaneously."""
    with pytest.raises(ValueError):
        GateOutcome.model_validate(
            {
                "job_id": "fixture",
                "candidate_run_id": "candidate",
                "specification_digest": "b" * 64,
                "gate_input_digest": "c" * 64,
                "status": status,
                "reasons": reasons,
                "decision_digest": decision,
            }
        )
