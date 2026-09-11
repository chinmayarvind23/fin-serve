"""Task boundaries preserve frozen inputs while reopening real SQLite and artifact stores."""

import sqlite3
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from test_managed_runtime import specification

from finserve.benchmark.runner import RunConfig
from finserve.benchmark.workload import WorkItem, Workload
from finserve.contracts.deployment import HealthObservation
from finserve.contracts.rollout import RolloutSettings
from finserve.evaluation.quality import default_suite
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.metadata import RegistryConflict
from finserve.registry.producer_pipeline import (
    ProducerExecution,
    cleanup_stage,
    collection_stage,
    execution_input,
    freeze_stage,
    journal_runtime,
)
from finserve.registry.producer_runtime import ProducerEngine, ProducerInput
from finserve.registry.producer_tasks import verified_model_receipt
from finserve.reliability.promotion import PromotionPolicy
from finserve.reliability.rollback import DeploymentStore
from finserve.reliability.warm_routes import WarmRouteStore


def test_task_reopen_fetch_replay_and_frozen_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual model fetch survives process-style reopen; changed or lost stores fail closed."""
    template = specification(tmp_path)
    producer = ProducerInput(
        job_id="entry",
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
        load=RunConfig(model="fixture", hardware="fixture-cpu", requests=4, warmup=1),
        suite=default_suite(),
        policy=PromotionPolicy(),
        repository=Path(__file__).resolve().parents[2],
        workspace=tmp_path / "work",
    )
    routes, control = tmp_path / "routes.sqlite", tmp_path / "control.sqlite"
    WarmRouteStore(routes)
    DeploymentStore(control)
    execution = ProducerExecution(producer=producer, routes=routes, control=control)
    request = tmp_path / "request.json"
    request.write_text(execution.model_dump_json())
    monkeypatch.setenv("FINSERVE_PRODUCER_REQUEST", str(request))
    monkeypatch.setenv("FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.sqlite"))
    monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    calls = 0

    def response(request: httpx.Request) -> httpx.Response:
        """Fetch exact original fixture bytes; no Docker or GPU behavior is simulated here."""
        nonlocal calls
        calls += 1
        assert request.url.path.endswith("/config.json")
        return httpx.Response(200, content=b'{"model_type":"fixture"}')

    def client() -> httpx.AsyncClient:
        """Retain the task's real client lifetime with an explicit local model transport."""
        return httpx.AsyncClient(transport=httpx.MockTransport(response))

    monkeypatch.setattr("finserve.registry.producer_pipeline.collection_client", client)
    with pytest.raises(ValueError, match="requires frozen rollout"):
        freeze_stage(require_rollout=True)
    assert freeze_stage() == freeze_stage() == producer.job_id
    assert collection_stage(producer.job_id, "fetch") == producer.job_id
    first_calls = calls
    assert first_calls > 0
    assert collection_stage(producer.job_id, "fetch") == producer.job_id
    assert calls == first_calls
    with journal_runtime() as journal:
        assert execution_input(journal, producer.job_id).execution == execution
        receipt = verified_model_receipt(journal, journal.state(producer.job_id + ":model"))
        assert receipt.directory == producer.workspace / "models" / producer.model.digest()
    other = tmp_path / "other.sqlite"
    WarmRouteStore(other)
    request.write_text(execution.model_copy(update={"routes": other}).model_dump_json())
    with pytest.raises(RegistryConflict):
        freeze_stage()
    # Cleanup reads the original immutable input, not the modified environment request file.
    assert cleanup_stage(producer.job_id) == {
        "baseline": "not_launched",
        "candidate": "not_launched",
    }
    next_job = producer.model_copy(update={"job_id": "next", "expected_generation": 1})
    next_execution = execution.model_copy(
        update={
            "producer": next_job,
            "rollout": RolloutSettings(traffic_url="http://traffic"),
        }
    )
    request.write_text(next_execution.model_dump_json())
    with pytest.raises(ValueError, match="initial deployment"):
        freeze_stage()
    controller = DeploymentStore(control)
    controller.register_revision(template.revision)
    controller.bootstrap(
        producer.deployment_id,
        template.revision.revision_id,
        HealthObservation(
            revision_id=template.revision.revision_id,
            revision_digest=template.revision.digest(),
            ready=True,
            smoke_passed=True,
        ),
    )
    next_execution = next_execution.model_copy(
        update={
            "producer": next_job.model_copy(update={"expected_generation": 0}),
        }
    )
    request.write_text(next_execution.model_dump_json())
    with pytest.raises(ValueError, match="unused deployment"):
        freeze_stage()
    with journal_runtime() as journal:
        assert journal.history("next:producer") == []
    routes.unlink()
    with pytest.raises(FileNotFoundError):
        cleanup_stage(producer.job_id)
    assert not routes.exists()
    routes.touch()
    with pytest.raises(sqlite3.OperationalError, match="store_identity"):
        cleanup_stage(producer.job_id)
    with closing(sqlite3.connect(routes)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
    WarmRouteStore(routes)
    with pytest.raises(ValueError, match="identity changed"):
        cleanup_stage(producer.job_id)
    request.write_bytes(b" " * (4 * 1024**2 + 1))
    with pytest.raises(ValueError, match="four MiB"):
        freeze_stage()


@pytest.mark.parametrize("kind", ["routes", "control"])
def test_store_transactions_reject_replacement_after_open(tmp_path: Path, kind: str) -> None:
    """A long-lived object cannot use new empty state when its database disappears or changes."""
    factory = WarmRouteStore if kind == "routes" else DeploymentStore
    path = tmp_path / "state.sqlite"
    original = factory(path)
    identity = original.identity
    factory(path, expected_identity=identity)
    path.unlink()
    with pytest.raises(sqlite3.OperationalError):
        with original.transaction():
            pytest.fail("missing database was silently recreated")
    assert not path.exists()
    replacement = factory(path)
    assert replacement.identity != identity
    with pytest.raises(ValueError, match="identity changed"):
        with original.transaction():
            pytest.fail("replaced database became authoritative")
    with pytest.raises(ValueError, match="identity changed"):
        factory(path, expected_identity=identity)
