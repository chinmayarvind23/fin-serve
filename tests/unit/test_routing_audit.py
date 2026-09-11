"""Offline routing audit rejects altered bytes before trusting retained summaries."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from finserve.benchmark.routing_workload import frozen_workload


@pytest.fixture
def auditor() -> ModuleType:
    """Load the operator script without depending on any private measured artifact."""
    path = Path(__file__).resolve().parents[2] / "scripts/audit_routing_session.py"
    spec = importlib.util.spec_from_file_location("routing_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["requests.jsonl", "../outside.json"])
def test_artifact_integrity_precedes_summary_acceptance(
    auditor: ModuleType,
    tmp_path: Path,
    name: str,
) -> None:
    """A valid-looking completed manifest cannot authorize changed bytes or path escape."""
    workload = frozen_workload()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "policy": "least_load",
                "workload_sha256": workload.digest(),
                "artifacts": {name: "0" * 64},
            }
        )
    )
    (tmp_path / "workload.json").write_text(workload.model_dump_json())
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "requests.jsonl").write_text("altered bytes")
    with pytest.raises(ValueError, match="artifact"):
        auditor.audit_cohort(tmp_path, "least_load")


def test_missing_cohort_cannot_be_reported_as_full_comparison(
    auditor: ModuleType, tmp_path: Path
) -> None:
    """A partial run remains partial instead of being reduced to whichever policies completed."""
    (tmp_path / "cohort-1-least_load").mkdir()
    with pytest.raises(ValueError, match="exactly"):
        auditor.audit_session(tmp_path)


def test_complete_artifact_set_and_source_binding(auditor: ModuleType, tmp_path: Path) -> None:
    """Omitting a digest or relabelling archived source cannot retain acceptance."""
    names = auditor.DATA_FILES | {"source__" + name.replace("/", "__") for name in auditor.SOURCES}
    for name in names:
        (tmp_path / name).write_text("retained")
    sha = auditor.digest(tmp_path / "requests.jsonl")
    manifest = {
        "artifacts": dict.fromkeys(names, sha),
        "sources": dict.fromkeys(auditor.SOURCES, sha),
    }
    auditor.validate_artifacts(tmp_path, manifest)
    changed = deepcopy(manifest)
    changed["artifacts"] = {}
    with pytest.raises(ValueError, match="artifact set"):
        auditor.validate_artifacts(tmp_path, changed)
    changed = deepcopy(manifest)
    changed["sources"]["engines/ray_http.py"] = "0" * 64
    with pytest.raises(ValueError, match="source digest"):
        auditor.validate_artifacts(tmp_path, changed)


@pytest.mark.parametrize("mutation", ["identity", "missing", "policy", "replica", "duplicate"])
def test_actual_routing_decision_reconciliation(auditor: ModuleType, mutation: str) -> None:
    """Successful labels alone cannot substitute for a matching recorded routing decision."""
    raw = [
        {
            "request_id": "run-measured-0",
            "record": {
                "phase": "measured",
                "logical_id": 0,
                "success": True,
            },
        }
    ]
    manifest = {
        "run_id": "run",
        "policy": "adaptive",
        "topology": {"backends": [{"replica_id": "a"}]},
    }
    final: dict[str, Any] = {
        "recent_decisions": [
            {
                "request_id": "run-measured-0",
                "decision": {
                    "policy": "adaptive",
                    "replica_id": "a",
                },
            }
        ]
    }
    auditor.validate_decisions(raw, manifest, final)
    if mutation == "identity":
        raw[0]["request_id"] = "different-measured-0"
    elif mutation == "missing":
        final["recent_decisions"] = []
    elif mutation == "duplicate":
        final["recent_decisions"] *= 2
    else:
        key = "replica_id" if mutation == "replica" else "policy"
        final["recent_decisions"][0]["decision"][key] = "wrong"
    with pytest.raises(ValueError):
        auditor.validate_decisions(raw, manifest, final)


@pytest.mark.parametrize(
    "mutation", ["topology", "sources", "tracing", "url", "workload_sha256", "overlap"]
)
def test_mixed_or_overlapping_cohorts_rejected(auditor: ModuleType, mutation: str) -> None:
    """Comparable policy labels cannot conceal changed controls or simultaneous intervals."""
    first: dict[str, Any] = dict.fromkeys(
        ("topology", "sources", "tracing", "url", "workload_sha256"), "same"
    )
    first.update(
        run_id="one",
        started_epoch_s=1,
        measured_started_epoch_s=2,
        measured_ended_epoch_s=3,
        finished_epoch_s=4,
    )
    second = dict(
        first,
        run_id="two",
        started_epoch_s=5,
        measured_started_epoch_s=6,
        measured_ended_epoch_s=7,
        finished_epoch_s=8,
    )
    auditor.validate_group([first, second])
    if mutation == "overlap":
        second["started_epoch_s"] = 3
    else:
        second[mutation] = "different"
    with pytest.raises(ValueError):
        auditor.validate_group([first, second])
