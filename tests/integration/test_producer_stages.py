"""Producer retries retain immutable inputs, interrupted attempts and verified output receipts."""

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.producer_stages import ProducerStages, StageState


@pytest.fixture
def journal(tmp_path: Path) -> Generator[ProducerStages]:
    """Use actual SQLAlchemy SQLite transactions and local immutable artifact bytes."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        yield ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
    finally:
        registry.close()


def test_stage_attempts_and_immutable_inputs(journal: ProducerStages) -> None:
    """A failed attempt remains visible; only an explicitly reconciled attempt permits retry."""
    reference = journal.artifacts.put(b'{"frozen":"input"}')
    planned = journal.declare("job:quality", reference)
    assert journal.declare("job:quality", reference) == planned
    with pytest.raises(RegistryConflict, match="input identity"):
        journal.declare("job:quality", journal.artifacts.put(b"changed"))
    running = journal.start("job:quality")
    assert running.attempt_id is not None
    with pytest.raises(RegistryConflict, match="reconciliation"):
        journal.start("job:quality")
    failed = journal.fail(
        "job:quality",
        running.attempt_id,
        "CancelledError",
        reconciliation=journal.artifacts.put(b'{"requests":"drained"}'),
    )
    assert failed.status == "failed" and failed.reconciliation is not None
    retried = journal.start("job:quality")
    assert retried.attempt_id is not None and retried.attempt_id != running.attempt_id
    receipt = journal.artifacts.put(b'{"raw_evidence":"verified"}')
    with pytest.raises(RegistryConflict, match="stale"):
        journal.finish("job:quality", running.attempt_id, receipt)
    completed = journal.finish("job:quality", retried.attempt_id, receipt)
    assert completed.status == "completed"
    assert journal.finish("job:quality", retried.attempt_id, receipt) == completed
    with pytest.raises(RegistryConflict):
        journal.finish("job:quality", retried.attempt_id, journal.artifacts.put(b"different"))
    with pytest.raises(RegistryConflict):
        journal.start("job:quality")
    assert [event.status for event in journal.history("job:quality")] == [
        "planned",
        "running",
        "failed",
        "running",
        "completed",
    ]


def test_concurrent_start_has_one_owner(journal: ProducerStages) -> None:
    """Two real SQL transactions cannot both acquire the same planned stage."""
    journal.declare("job:build", journal.artifacts.put(b"specification"))

    def claim() -> bool:
        """Return the ownership outcome without swallowing unexpected database failures."""
        try:
            journal.start("job:build")
            return True
        except RegistryConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim) for _ in range(2)]
        outcomes = [future.result() for future in futures]
    assert outcomes.count(True) == 1
    assert journal.state("job:build").attempt_number == 1


def test_artifact_verification_and_reopened_ambiguity(tmp_path: Path) -> None:
    """A process restart preserves running ambiguity and rejects altered receipt namespaces."""
    url = "sqlite:///" + str(tmp_path / "registry.sqlite")
    registry = Registry(url)
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = ProducerStages(registry, store)
    first.declare("job:runtime", store.put(b"runtime-input"))
    running = first.start("job:runtime")
    registry.close()
    reopened = Registry(url)
    try:
        second = ProducerStages(reopened, store)
        assert second.state("job:runtime") == running
        with pytest.raises(RegistryConflict):
            second.start("job:runtime")
        assert running.attempt_id is not None
        altered = store.put(b"output").model_copy(update={"namespace": "wrong"})
        with pytest.raises(ValueError):
            second.finish("job:runtime", running.attempt_id, altered)
        assert second.state("job:runtime").status == "running"
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "change",
    [
        {"attempt_number": 1},
        {"status": "running"},
        {"finished_at": 1},
        {"error_code": "Failure"},
    ],
)
def test_stage_state_rejects_invented_observations(
    journal: ProducerStages, change: dict[str, object]
) -> None:
    """Typed evidence cannot claim attempts or completion without the corresponding observations."""
    value: dict[str, object] = {"stage_id": "job:stage", "input": journal.artifacts.put(b"input")}
    value.update(change)
    with pytest.raises(ValueError):
        StageState.model_validate(value)


def test_clock_regression_cannot_publish_completion(journal: ProducerStages) -> None:
    """A backwards real-clock observation rolls back the SQL transaction and leaves work pending."""
    journal.declare("job:clock", journal.artifacts.put(b"input"))
    journal.clock = lambda: 20.0
    running = journal.start("job:clock")
    assert running.attempt_id is not None
    journal.clock = lambda: 19.0
    with pytest.raises(ValueError, match="finish"):
        journal.finish("job:clock", running.attempt_id, journal.artifacts.put(b"output"))
    assert journal.state("job:clock") == running


@pytest.mark.parametrize(
    "change",
    [
        {"finished_at": 3.0},
        {"status": "completed", "finished_at": 3.0},
        {"status": "failed", "finished_at": 3.0, "error_code": "Drained"},
    ],
)
def test_unproved_stage_outcomes_fail_validation(
    journal: ProducerStages, change: dict[str, object]
) -> None:
    """Finish times, output receipts and failure reconciliation require separate observations."""
    value: dict[str, object] = {
        "stage_id": "job:stage",
        "input": journal.artifacts.put(b"input"),
        "status": "running",
        "attempt_number": 1,
        "attempt_id": "a" * 32,
        "started_at": 2.0,
    }
    value.update(change)
    with pytest.raises(ValueError):
        StageState.model_validate(value)
