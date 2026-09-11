"""The concurrency frontier must reject changed envelopes and misleading success subsets."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from finserve.evaluation import quality as quality_module
from finserve.evaluation.quality import default_suite, evaluate_quality


@pytest.fixture
def auditor() -> ModuleType:
    """Load the offline CLI without adding it to the serving runtime or importing plotting deps."""
    path = Path(__file__).resolve().parents[2] / "scripts/audit_concurrency_frontier.py"
    spec = importlib.util.spec_from_file_location("concurrency_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cells() -> list[dict[str, Any]]:
    """Keep a small comparable population with both dominated and nondominated observations."""
    base: dict[str, Any] = {
        "configuration": {"concurrency": 1, "requests": 256, "engine_config": "fixed"},
        "workload_hash": "same",
        "client_platform": "same",
        "python": "same",
        "endpoint": "same",
        "physical_device_ids": ["one"],
        "source_provenance": {"tracked_runtime": {"src": "same"}},
        "summary": {"failed_requests": 0, "requests_per_second": 1, "latency": 0.1},
    }
    result = [deepcopy(base) for _ in range(3)]
    for row, concurrency, rate, delay in zip(
        result, (1, 4, 8), (1, 2, 3), (0.1, 0.3, 0.2), strict=True
    ):
        row["configuration"]["concurrency"] = concurrency
        row["summary"].update(requests_per_second=rate, latency=delay)
    return result


@pytest.mark.parametrize(
    "field,value", [("requests", 3072), ("engine_config", "ngram"), ("concurrency", 1)]
)
def test_changed_population_profile_or_duplicate_cell_rejected(
    auditor: ModuleType,
    field: str,
    value: object,
) -> None:
    """A shared workload hash cannot hide more repetitions, a new engine profile or reruns."""
    rows = cells()
    rows[1]["configuration"][field] = value
    with pytest.raises(ValueError):
        auditor.require_comparable(rows)


@pytest.mark.parametrize("field", ["workload_hash", "endpoint", "physical_device_ids"])
def test_changed_workload_transport_or_device_rejected(auditor: ModuleType, field: str) -> None:
    """Hold the whole measurement boundary fixed before calculating a frontier."""
    rows = cells()
    rows[1][field] = "different"
    with pytest.raises(ValueError):
        auditor.require_comparable(rows)


def test_frontier_retains_low_latency_and_high_throughput_extremes(auditor: ModuleType) -> None:
    """Only the middle dominated point drops; there is no smoothing or invented optimum."""
    rows = cells()
    auditor.require_comparable(rows)
    assert auditor.frontier(rows, "latency") == [1, 8]
    rows[1]["summary"]["failed_requests"] = 1
    with pytest.raises(ValueError, match="hide failed"):
        auditor.frontier(rows, "latency")


def test_runtime_change_is_not_hidden_by_revision_labels(auditor: ModuleType) -> None:
    """Even identical declared configuration cannot authorize changed tracked serving bytes."""
    rows = cells()
    rows[1]["source_provenance"]["tracked_runtime"]["src"] = "changed"
    with pytest.raises(ValueError, match="runtime changed"):
        auditor.require_comparable(rows)


@pytest.mark.parametrize("fault", ["none", "evaluator", "case_ids"])
def test_quality_context_binds_grader_and_exact_case_set(
    auditor: ModuleType,
    tmp_path: Path,
    fault: str,
) -> None:
    """Equal case counts or coincidentally equal scores cannot conceal a different frozen grader."""
    suite = default_suite()
    answers = {case.case_id: case.expected for case in suite.cases}
    config = {
        key: "fixture"
        for key in ("model", "model_revision", "tokenizer_revision", "engine", "engine_config")
    }
    (tmp_path / "answers.json").write_text(json.dumps(answers))
    manifest: dict[str, Any] = {
        "suite": suite.model_dump(),
        "suite_hash": suite.digest(),
        "status": "completed",
        "reference_manifest_sha256": None,
        "answers_sha256": auditor.digest(tmp_path / "answers.json"),
        "evaluator_sha256": hashlib.sha256(
            Path(str(quality_module.__file__)).read_bytes()
        ).hexdigest(),
        "run_id": "fixture",
        "git_sha": "fixture",
        "git_status": [],
        **config,
    }
    raw = [{"case_id": key, "output": value} for key, value in answers.items()]
    if fault == "evaluator":
        manifest["evaluator_sha256"] = "0" * 64
    elif fault == "case_ids":
        raw[0]["case_id"] = "unknown-case-with-same-cardinality"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "quality.json").write_text(json.dumps(evaluate_quality(suite, answers, answers)))
    (tmp_path / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in raw))
    if fault == "none":
        assert auditor.quality_context(tmp_path, config)["passed"]
    else:
        with pytest.raises(ValueError):
            auditor.quality_context(tmp_path, config)


@pytest.mark.parametrize(
    "status",
    [
        [" M src/finserve/gateway/app.py"],
        ["?? src/finserve/engines/unknown.py"],
        [" M uv.lock"],
        ["R  src/finserve/gateway/app.py -> docs/moved.py"],
    ],
)
def test_dirty_runtime_rejected_by_default(auditor: ModuleType, status: list[str]) -> None:
    """Matching committed trees cannot hide tracked edits, added modules or renamed source."""
    with pytest.raises(ValueError, match="dirty runtime"):
        auditor.runtime_dirty_exclusions(status, frozenset())


def test_only_exact_present_reviewed_untracked_exclusions_are_allowed(auditor: ModuleType) -> None:
    """Explicit legacy exceptions stay finite and cannot authorize tracked versions or globs."""
    paths = frozenset(
        {
            "src/finserve/multimodal/jax_generator.py",
            "src/finserve/registry/__init__.py",
            "src/finserve/registry/artifacts.py",
        }
    )
    status = [f"?? {path}" for path in paths]
    assert auditor.runtime_dirty_exclusions(status, paths) == sorted(paths)
    for invalid in (status[:-1], [entry.replace("??", " M") for entry in status]):
        with pytest.raises(ValueError):
            auditor.runtime_dirty_exclusions(invalid, paths)
    with pytest.raises(ValueError):
        auditor.runtime_dirty_exclusions(status, frozenset({"src/finserve/*"}))
