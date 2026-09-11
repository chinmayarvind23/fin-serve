"""Audit one fixed-configuration concurrency sweep without contacting an engine."""

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from finserve.benchmark.runner import prepare_output, validate_evidence, write_json
from finserve.evaluation import quality as quality_module
from finserve.evaluation.quality import GoldenSuite, evaluate_quality

REVIEWED_UNTRACKED_RUNTIME = frozenset(
    {
        "src/finserve/multimodal/jax_generator.py",
        "src/finserve/registry/__init__.py",
        "src/finserve/registry/artifacts.py",
    }
)


def read(path: Path) -> Any:
    """Retained artifacts are trusted local inputs; production validators check their structure."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Bind aggregate observations to exact raw bytes without publishing prompts or host paths."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    """Reject altered or incomparable inputs instead of producing a plausible-looking frontier."""
    if not condition:
        raise ValueError(message)


def quantile(values: list[float], fraction: float) -> float:
    """Independently reconstruct the runner's linear percentile for the declared population."""
    require(bool(values), "empty latency population")
    ordered = sorted(values)
    low = math.floor(position := (len(values) - 1) * fraction)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def source_provenance(actual: str, declared: str) -> dict[str, Any]:
    """Keep revision mismatches explicit; only identical tracked runtime trees permit comparison."""
    repository = Path(__file__).resolve().parents[1]
    require(
        all(re.fullmatch(r"[a-f0-9]{40}", revision) for revision in (actual, declared)),
        "full source revisions required",
    )
    identities: dict[str, dict[str, str]] = {}
    for label, revision in (("actual", actual), ("declared", declared)):
        identities[label] = {
            path: subprocess.check_output(
                ["git", "rev-parse", f"{revision}:{path}"], cwd=repository, text=True
            ).strip()
            for path in ("src", "uv.lock", "pyproject.toml")
        }
    require(identities["actual"] == identities["declared"], "tracked runtime source differs")
    return {
        "actual_revision": actual,
        "declared_revision": declared,
        "revision_matches": actual == declared,
        "tracked_runtime": identities["actual"],
        "revision_difference_paths": subprocess.check_output(
            ["git", "diff", "--name-only", declared, actual], cwd=repository, text=True
        ).splitlines(),
    }


def runtime_dirty_exclusions(status: list[str], allowed: frozenset[str]) -> list[str]:
    """Reject runtime dirtiness unless exact, reviewed untracked legacy paths are opted in.

    Exclusion does not attest those file bytes. Tracked edits, renames, quoted paths and
    directory/glob exceptions cannot be used to smuggle a changed runtime into a sweep.
    """
    require(allowed <= REVIEWED_UNTRACKED_RUNTIME, "unreviewed runtime exclusion")
    require(
        {f"?? {path}" for path in allowed} <= set(status),
        "each exclusion must be present as an untracked file",
    )
    for entry in status:
        require(len(entry) >= 4 and entry[2] == " ", "unsupported dirty status syntax")
        for path in entry[3:].split(" -> "):
            require(
                not path.startswith('"') and "\\" not in path,
                "quoted or escaped dirty paths require separate review",
            )
            runtime = (
                path == "src" or path.startswith("src/") or path in {"uv.lock", "pyproject.toml"}
            )
            if runtime:
                require(entry[:2] == "??" and path in allowed, "unapproved dirty runtime path")
    return sorted(allowed)


def audit_run(root: Path, allowed: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Recompute each complete raw population while retaining the dirty-source qualification."""
    typed = validate_evidence(root / "run")
    manifest, environment, saved = (
        read(root / path) for path in ("run/manifest.json", "environment.json", "run/summary.json")
    )
    require(environment["configuration"] == manifest["configuration"], "configuration changed")
    provenance = source_provenance(environment["git_sha"], typed.configuration.revision)
    exclusions = runtime_dirty_exclusions(environment["git_status"], allowed)
    rows = [json.loads(line) for line in (root / "run/requests.jsonl").read_text().splitlines()]
    measured = [row for row in rows if row["phase"] == "measured"]
    success = [row for row in measured if row["success"]]
    duration = manifest["measured_finished_s"] - manifest["measured_started_s"]
    summary = {
        "offered_requests": len(measured),
        "successful_requests": len(success),
        "failed_requests": len(measured) - len(success),
        "measured_seconds": duration,
        "generated_tokens": sum(row["generated_tokens"] for row in success),
        "requests_per_second": len(success) / duration,
        "tokens_per_second": sum(row["generated_tokens"] for row in success) / duration,
        "client_ttft_median_s": quantile(
            [
                row["first_content_s"] - row["send_s"]
                for row in success
                if row["first_content_s"] is not None
            ],
            0.5,
        ),
        "e2e_p95_s": quantile([row["complete_s"] - row["send_s"] for row in success], 0.95),
    }
    require(
        all(
            math.isclose(value, saved[key], rel_tol=1e-10, abs_tol=1e-9)
            for key, value in summary.items()
        ),
        "independent metric mismatch",
    )
    samples = [json.loads(line) for line in (root / "gpu.jsonl").read_text().splitlines()]
    devices = sorted({device["uuid"] for sample in samples for device in sample["devices"]})
    require(len(devices) == 1, "this local sweep requires one observed physical GPU")
    return {
        "name": root.name,
        "run_id": manifest["run_id"],
        "created_at": manifest["created_at"],
        "configuration": typed.configuration.model_dump(),
        "workload_hash": typed.workload_hash,
        "client_platform": manifest["client_platform"],
        "python": manifest["python"],
        "endpoint": manifest["url"],
        "physical_device_ids": devices,
        "summary": summary,
        "warmup_records": len(rows) - len(measured),
        "dirty_paths": environment["git_status"],
        "source_provenance": provenance,
        "reviewed_untracked_runtime_exclusions": exclusions,
        "untracked_runtime_bytes_attested": False,
        "artifact_sha256": {
            path.relative_to(root).as_posix(): digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        },
    }


def require_comparable(runs: list[dict[str, Any]]) -> None:
    """Allow concurrency alone to vary; identical workload hashes do not imply equal populations."""
    require(len(runs) >= 2, "at least two concurrency cells required")
    configurations = [
        {key: value for key, value in run["configuration"].items() if key != "concurrency"}
        for run in runs
    ]
    require(
        all(config == configurations[0] for config in configurations),
        "only concurrency may differ within a sweep",
    )
    for key in ("workload_hash", "client_platform", "python", "endpoint", "physical_device_ids"):
        require(all(run[key] == runs[0][key] for run in runs), f"incomparable {key}")
    require(
        all(
            run["source_provenance"]["tracked_runtime"]
            == runs[0]["source_provenance"]["tracked_runtime"]
            for run in runs
        ),
        "tracked runtime changed between cells",
    )
    require(
        len({run["configuration"]["concurrency"] for run in runs}) == len(runs),
        "repeats require a separate uncertainty analysis",
    )


def frontier(runs: list[dict[str, Any]], latency: str) -> list[int]:
    """Find observed nondominated points, without fitting a curve or claiming statistical rank."""
    require(
        all(run["summary"]["failed_requests"] == 0 for run in runs),
        "this complete-success frontier cannot hide failed requests",
    )
    points = [(run["summary"]["requests_per_second"], run["summary"][latency]) for run in runs]
    return [
        runs[index]["configuration"]["concurrency"]
        for index, (rate, delay) in enumerate(points)
        if not any(
            other_rate >= rate
            and other_delay <= delay
            and (other_rate > rate or other_delay < delay)
            for other_rate, other_delay in points
        )
    ]


def quality_context(root: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    """Recompute the separate self-reference correctness cohort; it is not per-cell quality."""
    manifest, answers, saved = (
        read(root / path) for path in ("manifest.json", "answers.json", "quality.json")
    )
    suite = GoldenSuite.model_validate(manifest["suite"])
    require(
        manifest["status"] == "completed" and manifest["reference_manifest_sha256"] is None,
        "a completed standalone quality context is required",
    )
    require(suite.digest() == manifest["suite_hash"], "quality suite hash mismatch")
    evaluator = Path(str(quality_module.__file__)).read_bytes().replace(b"\r\n", b"\n")
    # This is the registry's LF/CRLF equivalence policy without its SQL dependency.
    evaluator_hashes = {
        hashlib.sha256(value).hexdigest()
        for value in (evaluator, evaluator.replace(b"\n", b"\r\n"))
    }
    require(manifest["evaluator_sha256"] in evaluator_hashes, "quality evaluator changed")
    require(digest(root / "answers.json") == manifest["answers_sha256"], "quality answers changed")
    for key in ("model", "model_revision", "tokenizer_revision", "engine", "engine_config"):
        require(manifest[key] == configuration[key], f"quality profile differs in {key}")
    raw = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    require(
        len(raw) == len(suite.cases)
        and {row["case_id"] for row in raw} == {case.case_id for case in suite.cases},
        "quality raw population mismatch",
    )
    require(
        {row["case_id"]: row["output"] for row in raw if "output" in row} == answers,
        "quality raw answers mismatch",
    )
    result = evaluate_quality(suite, answers, answers)
    require(all(saved[key] == value for key, value in result.items()), "quality score mismatch")
    return {
        "name": root.name,
        "run_id": manifest["run_id"],
        "suite_hash": suite.digest(),
        "evaluator_sha256": manifest["evaluator_sha256"],
        "candidate_accuracy": result["candidate_accuracy"],
        "passed": result["passed"],
        "case_count": len(suite.cases),
        "source_revision": manifest["git_sha"],
        "dirty_paths": manifest["git_status"],
        "scope": "Separate same-profile correctness; no concurrency-specific quality estimate",
        "artifact_sha256": {path.name: digest(path) for path in root.iterdir() if path.is_file()},
    }


def main() -> None:
    """Write a fresh external audit; raw source artifacts are never modified or replaced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument(
        "--allow-untracked-runtime",
        action="append",
        default=[],
        choices=sorted(REVIEWED_UNTRACKED_RUNTIME),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted(
        [audit_run(path, frozenset(args.allow_untracked_runtime)) for path in args.runs],
        key=lambda run: run["configuration"]["concurrency"],
    )
    require_comparable(runs)
    report = {
        "schema": "finserve-concurrency-frontier-v1",
        "runs": runs,
        "frontiers": {
            metric: frontier(runs, metric) for metric in ("client_ttft_median_s", "e2e_p95_s")
        },
        "quality_context": quality_context(args.quality, runs[0]["configuration"]),
        "audit_source_sha256": digest(Path(__file__)),
        "limitations": [
            "Single sequential development cell per concurrency; no uncertainty bands.",
            "Dirty trees recorded; no exact archived dirty source or runtime image digest.",
            "Explicit reviewed untracked runtime exclusions do not attest excluded file bytes.",
            "Fixed engine sequence/token limits; actual internal batch shapes unobserved.",
            "Latency uses successful requests; warmups excluded from measured denominator.",
            "Separate correctness failure prevents a release or useful-quality optimum claim.",
            "No sustained, speculation, different-engine or changed-population cells pooled.",
        ],
    }
    write_json(prepare_output(args.output) / "frontier.json", report)


if __name__ == "__main__":
    main()
