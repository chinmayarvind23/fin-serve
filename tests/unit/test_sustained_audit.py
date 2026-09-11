"""Offline audit boundaries reject altered quality evidence without making model requests."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from finserve.evaluation.quality import default_suite, evaluate_quality


@pytest.fixture
def auditor() -> ModuleType:
    """Load the source-controlled CLI while keeping it outside the application package."""
    path = Path(__file__).resolve().parents[2] / "scripts/audit_sustained_comparison.py"
    spec = importlib.util.spec_from_file_location("sustained_audit_review", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantile_interpolates_between_adjacent_observations(auditor: ModuleType) -> None:
    """A two-point population distinguishes interpolation from rounding to a sample."""
    assert auditor.quantile([1.0, 0.0], 0.95) == 0.95
    with pytest.raises(ValueError, match="empty"):
        auditor.quantile([], 0.5)


@pytest.mark.parametrize("fault", ["none", "bytes", "source", "raw", "suite", "report"])
def test_quality_audit_rejects_tampering(auditor: ModuleType, tmp_path: Path, fault: str) -> None:
    """Reject plausible scores with altered answers, run identity or case population."""
    suite = default_suite()
    answers = {case.case_id: case.expected for case in suite.cases}
    configuration = {
        "revision": "declared-source",
        "model": "fixture",
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "engine": "fixture",
        "engine_config": "fixed",
    }
    (tmp_path / "answers.json").write_text(json.dumps(answers))
    manifest: dict[str, object] = {
        "suite": suite.model_dump(),
        "suite_hash": suite.digest(),
        "evaluator_sha256": "declared-evaluator",
        "status": "completed",
        "answers_sha256": auditor.digest(tmp_path / "answers.json"),
        "git_sha": configuration["revision"],
        "git_status": [],
        "run_id": "fixture",
        **{key: value for key, value in configuration.items() if key != "revision"},
    }
    rows = [{"case_id": key, "output": value} for key, value in answers.items()]
    score = evaluate_quality(suite, answers, answers)
    if fault == "bytes":
        (tmp_path / "answers.json").write_text(json.dumps(answers) + " ")
    elif fault == "source":
        manifest["git_sha"] = "different-source"
    elif fault == "raw":
        rows[0]["output"] = "altered-output"
    elif fault == "suite":
        manifest["suite_hash"] = "altered-suite"
    elif fault == "report":
        score["candidate_accuracy"] = 0.25
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "quality.json").write_text(json.dumps(score))
    if fault == "none":
        assert auditor.quality_run(tmp_path, tmp_path, {"configuration": configuration})["result"][
            "passed"
        ]
    else:
        with pytest.raises(ValueError):
            auditor.quality_run(tmp_path, tmp_path, {"configuration": configuration})
