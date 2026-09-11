"""Audit saved native experiments and quality artifacts without contacting an engine."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from finserve.benchmark.runner import prepare_output, validate_comparison, validate_evidence
from finserve.evaluation.quality import GoldenSuite, QualityConfig, evaluate_quality


def read(path: Path) -> Any:
    """Load retained JSON artifacts; typed production validators check benchmark structure."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Bind the audit to exact artifact bytes, including the native runtime freezes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail the audit on disagreement instead of writing a partially trusted comparison."""
    if not condition:
        raise ValueError(message)


def close(left: float, right: float) -> bool:
    """Permit only floating-point summation and epoch subtraction rounding differences."""
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9)


def quantile(values: list[float], fraction: float) -> float:
    """Reconstruct the declared linear percentile independently of benchmark metric helpers."""
    require(bool(values), "empty quantile population")
    ordered = sorted(values)
    lower = math.floor(position := (len(ordered) - 1) * fraction)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def gpu_oracle(root: Path, manifest: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """Integrate raw sample intervals against the benchmark's independently mapped clock window."""
    saved = read(root / "gpu-summary.json")
    shift = env["clock_epoch_anchor_s"] - env["clock_monotonic_anchor_s"]
    start, end = (manifest[f"measured_{label}_s"] + shift for label in ("started", "finished"))
    require(close(start, saved["window_start_epoch_s"]), "GPU start clock mismatch")
    require(close(end, saved["window_end_epoch_s"]), "GPU end clock mismatch")
    # Epoch floats lose sub-microsecond precision; require explicit absolute bounds as well.
    require(abs(start - saved["window_start_epoch_s"]) < 1e-6, "GPU start drift")
    require(abs(end - saved["window_end_epoch_s"]) < 1e-6, "GPU end drift")
    samples = [json.loads(line) for line in (root / "gpu.jsonl").read_text().splitlines()]
    identities = {device["uuid"] for sample in samples for device in sample["devices"]}
    require(sorted(identities) == saved["device_ids"], "GPU inventory mismatch")
    require(bool(identities), "missing GPU inventory")
    covered: list[float] = []
    areas: list[float] = []
    for index, sample in enumerate(samples):
        next_time = samples[index + 1]["epoch_s"] if index + 1 < len(samples) else end
        if index + 1 < len(samples):
            require(next_time > sample["epoch_s"], "unordered GPU samples")
        devices = sample["devices"]
        if sample["error"] or {device["uuid"] for device in devices} != identities:
            continue
        require(len(devices) == len(identities), "duplicate GPU sample device")
        utilizations = [device["utilization_percent"] for device in devices]
        require(
            all(math.isfinite(value) and 0 <= value <= 100 for value in utilizations),
            "invalid GPU utilization",
        )
        duration = max(
            0,
            min(end, next_time, sample["epoch_s"] + saved["max_gap_seconds"])
            - max(start, sample["epoch_s"]),
        )
        covered.append(duration)
        areas.append(duration * math.fsum(utilizations) / len(devices))
    coverage = math.fsum(covered) / (end - start)
    require(close(coverage, saved["coverage"]), "GPU coverage mismatch")
    require(coverage >= saved["minimum_coverage"], "insufficient GPU coverage")
    utilization = math.fsum(areas) / math.fsum(covered)
    require(close(utilization, saved["average_gpu_utilization_percent"]), "GPU mean mismatch")
    status = read(root / "experiment-status.json")
    require(status["status"] == "completed", "incomplete experiment")
    require(abs(status["clock_drift_seconds"]) <= 0.1, "unreliable wall/monotonic mapping")
    return {
        **saved,
        "raw_samples": len(samples),
        "collection_failures": sum(sample["error"] is not None for sample in samples),
        "clock_drift_seconds": status["clock_drift_seconds"],
    }


def experiment(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check every raw population and independently recalculate the reported headline metrics."""
    validate_evidence(root / "run")
    manifest, env, saved = (
        read(root / name) for name in ("run/manifest.json", "environment.json", "run/summary.json")
    )
    config = manifest["configuration"]
    require(env["configuration"] == config, "environment configuration mismatch")
    require(
        env["git_sha"] == config["revision"] and env["git_status"] == [],
        "native experiment source is not a clean declared revision",
    )
    raw = [json.loads(line) for line in (root / "run/requests.jsonl").read_text().splitlines()]
    rows = sorted(
        (row for row in raw if row["phase"] == "measured"), key=lambda row: row["logical_id"]
    )
    success = [row for row in rows if row["success"]]
    seconds = manifest["measured_finished_s"] - manifest["measured_started_s"]
    tokens = sum(row["generated_tokens"] for row in success)
    summary = {
        "measured_seconds": seconds,
        "offered_requests": len(rows),
        "successful_requests": len(success),
        "failed_requests": len(rows) - len(success),
        "generated_tokens": tokens,
        "success_rate": len(success) / len(rows),
        "requests_per_second": len(success) / seconds,
        "tokens_per_second": tokens / seconds,
        "e2e_p95_s": quantile([row["complete_s"] - row["send_s"] for row in success], 0.95),
        "client_ttft_median_s": quantile(
            [
                row["first_content_s"] - row["send_s"]
                for row in success
                if row["first_content_s"] is not None
            ],
            0.5,
        ),
        "server_ttft_median_s": quantile(
            [row["server_ttft_s"] for row in success if row["server_ttft_s"] is not None], 0.5
        ),
    }
    for key, value in summary.items():
        require(close(value, saved[key]), f"independent summary mismatch: {key}")
    hashes = {
        str(path.relative_to(root)).replace("\\", "/"): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return {
        "name": root.name,
        "run_id": manifest["run_id"],
        "configuration": config,
        "workload_hash": manifest["workload_hash"],
        "summary": summary,
        "warmup_records": len(raw) - len(rows),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "gpu": gpu_oracle(root, manifest, env),
        "artifact_sha256": hashes,
    }, rows


def quality_run(root: Path, reference_root: Path, experiment_run: dict[str, Any]) -> dict[str, Any]:
    """Recompute the unchanged grader, reference linkage and answer hashes for all cases."""
    manifest, reference_manifest = (read(path / "manifest.json") for path in (root, reference_root))
    suite = GoldenSuite.model_validate(manifest["suite"])
    require(
        suite.digest() == manifest["suite_hash"] == reference_manifest["suite_hash"],
        "quality suite mismatch",
    )
    require(
        manifest["evaluator_sha256"] == reference_manifest["evaluator_sha256"],
        "quality evaluator changed",
    )
    require(
        manifest["status"] == reference_manifest["status"] == "completed", "incomplete quality run"
    )
    if root != reference_root:
        require(
            manifest["reference_manifest_sha256"] == digest(reference_root / "manifest.json"),
            "quality reference manifest changed",
        )
    for path, item in ((root, manifest), (reference_root, reference_manifest)):
        require(digest(path / "answers.json") == item["answers_sha256"], "answer hash mismatch")
    require(
        manifest["git_sha"] == experiment_run["configuration"]["revision"]
        and manifest["git_status"] == [],
        "quality source differs from benchmark",
    )
    for key in ("model", "model_revision", "tokenizer_revision", "engine", "engine_config"):
        require(manifest[key] == experiment_run["configuration"][key], f"quality config: {key}")
    reference, answers = (read(path / "answers.json") for path in (reference_root, root))
    raw = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    require(
        len(raw) == len(suite.cases) and len({row["case_id"] for row in raw}) == len(raw),
        "quality raw population mismatch",
    )
    require(
        {row["case_id"]: row["output"] for row in raw if "output" in row} == answers,
        "quality raw answers mismatch",
    )
    saved = read(root / "quality.json")
    recomputed = evaluate_quality(suite, reference, answers, QualityConfig())
    require(
        all(saved[key] == value for key, value in recomputed.items()), "quality result mismatch"
    )
    return {
        "name": root.name,
        "run_id": manifest["run_id"],
        "suite_hash": suite.digest(),
        "evaluator_sha256": manifest["evaluator_sha256"],
        "result": recomputed,
        "artifact_sha256": {
            path.name: digest(path) for path in sorted(root.iterdir()) if path.is_file()
        },
    }


def audit(
    baseline: Path, candidate: Path, baseline_quality: Path, candidate_quality: Path
) -> dict[str, Any]:
    """Separate serving parity, deterministic correctness and observed performance changes."""
    validate_comparison(baseline / "run", candidate / "run")
    left, left_rows = experiment(baseline)
    right, right_rows = experiment(candidate)
    differences = {
        key: [value, right["configuration"][key]]
        for key, value in left["configuration"].items()
        if value != right["configuration"][key]
    }
    require(set(differences) == {"engine_config"}, "unexpected experimental configuration changes")
    require(left["gpu"]["device_ids"] == right["gpu"]["device_ids"], "physical GPU changed")
    for name in ("engine-python-freeze.txt", "gateway-python-freeze.txt", "kernel.txt"):
        require(
            left["artifact_sha256"][name] == right["artifact_sha256"][name],
            f"native runtime changed: {name}",
        )
    pairs: list[dict[str, Any]] = []
    for before, after in zip(left_rows, right_rows, strict=True):
        require(
            (before["logical_id"], before["case_id"], before["family"])
            == (after["logical_id"], after["case_id"], after["family"]),
            "unpaired workload",
        )
        pairs.append(
            {
                "logical_id": before["logical_id"],
                "case_id": before["case_id"],
                "family": before["family"],
                "exact_match": before["output"] == after["output"],
                "token_count_match": before["generated_tokens"] == after["generated_tokens"],
            }
        )
    quality = [
        quality_run(baseline_quality, baseline_quality, left),
        quality_run(candidate_quality, baseline_quality, right),
    ]
    return {
        "scope": "One ordered native local GPU comparison; no randomized repeats or cloud cost.",
        "runs": [left, right],
        "configuration_differences": differences,
        "measured_requests_total": len(left_rows) + len(right_rows),
        "serving_parity": {
            "pairs": len(pairs),
            "exact_matches": sum(row["exact_match"] for row in pairs),
            "parity": sum(row["exact_match"] for row in pairs) / len(pairs),
            "token_count_matches": sum(row["token_count_match"] for row in pairs),
            "pairs_detail": pairs,
        },
        "relative_change_percent": {
            key: (right["summary"][key] / left["summary"][key] - 1) * 100
            for key in (
                "requests_per_second",
                "tokens_per_second",
                "e2e_p95_s",
                "server_ttft_median_s",
                "client_ttft_median_s",
            )
        },
        "quality": quality,
        "candidate_quality_gate_passed": quality[1]["result"]["passed"],
        "limitations": [
            "Quality uses a separate frozen 32-case suite, not the load-test prompts.",
            "Exact output agreement does not establish task correctness.",
            "Successful HTTP samples do not establish a long-term availability SLO.",
            "Native image/config digests remain undeclared; no container promotion claim.",
            "The runs share one workstation GPU; order, thermal and host effects remain.",
            "GPU utilization is observed device activity, not useful-work efficiency.",
            "Local hardware has no measured billed cloud cost.",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    """Render the audited numbers with the failed quality gate next to the speed observations."""
    left, right = report["runs"]
    lines = [
        "# Audited native sustained comparison",
        "",
        report["scope"],
        "",
        "Candidate quality gate: **"
        + ("passed" if report["candidate_quality_gate_passed"] else "FAILED; optimization rejected")
        + "**.",
        "",
        "| Metric | Eager baseline | Compiled candidate |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("successful_requests", "Successful measured requests"),
        ("generated_tokens", "Generated tokens"),
        ("measured_seconds", "Measured seconds"),
        ("requests_per_second", "Requests/s"),
        ("tokens_per_second", "Tokens/s"),
        ("e2e_p95_s", "E2E p95 seconds"),
        ("server_ttft_median_s", "Server TTFT median seconds"),
        ("client_ttft_median_s", "Client TTFT median seconds"),
    ):
        lines.append(f"| {label} | {left['summary'][key]:.6f} | {right['summary'][key]:.6f} |")
    for key, label in (
        ("average_gpu_utilization_percent", "GPU utilization %"),
        ("coverage", "GPU coverage fraction"),
    ):
        lines.append(f"| {label} | {left['gpu'][key]:.6f} | {right['gpu'][key]:.6f} |")
    parity = report["serving_parity"]
    lines += [
        "",
        f"Exact paired serving output agreement: {parity['exact_matches']}/{parity['pairs']}"
        f" ({parity['parity']:.4%}). This is separate from correctness.",
        "",
    ]
    for result in report["quality"]:
        score = result["result"]
        lines.append(
            f"- {result['name']}: {score['candidate_accuracy']:.2%} correctness, "
            f"{score['parity']:.2%} grader agreement, gate passed={score['passed']}."
        )
    lines += [
        "",
        "Both raw populations, summaries, GPU windows and quality reports were recomputed. "
        "comparison.json records run IDs, source/model configuration and every input file hash.",
        "",
    ]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    """Require a new external report directory so prior audit evidence is never overwritten."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "baseline-quality", "candidate-quality", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.baseline, args.candidate, args.baseline_quality, args.candidate_quality)
    destination = prepare_output(args.output)
    (destination / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (destination / "comparison.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
