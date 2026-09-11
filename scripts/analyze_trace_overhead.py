"""Recompute the fixed six-cohort HTTP tracing experiment without launching any runtime."""

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from finserve.benchmark.runner import validate_evidence
from finserve.evaluation.quality import unique_object

ORDER = ("off", "sampled", "full", "full", "sampled", "off")
MAX_FILE_BYTES = 32 * 1024**2


def read(path: Path) -> bytes:
    """Bound this local evidence audit before parsing or passing files to existing validators."""
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("audit input exceeds byte limit")
    return raw


def document(path: Path) -> dict[str, Any]:
    """Reject ambiguous duplicate keys and non-object metadata in retained receipts."""
    value = json.loads(read(path), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("audit metadata must be an object")
    return cast(dict[str, Any], value)


def check(condition: bool, message: str) -> None:
    """Explicit checks remain active when Python assertion optimization is enabled."""
    if not condition:
        raise ValueError(message)


def percentile(values: list[float], quantile: float) -> float:
    """Independent linear interpolation mirrors the declared method without importing metrics."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def validate_traces(directory: Path, mode: str, receipt: dict[str, Any]) -> int:
    """Verify population, root identity, timestamps and the exact synthetic metadata allowlist."""
    path = directory / "traces.jsonl"
    raw = read(path) if path.exists() else b""
    spans = [json.loads(line, object_pairs_hook=unique_object) for line in raw.splitlines()]
    check(len(spans) == receipt["exported_spans_including_warmup"], "trace count mismatch")
    digest = hashlib.sha256(raw).hexdigest() if path.exists() else None
    check(digest == receipt["trace_sha256"], "trace hash mismatch")
    check(mode != "off" or not path.exists(), "disabled tracing unexpectedly has an artifact")
    check(mode != "full" or len(spans) == 1056, "full tracing population mismatch")
    check(len(spans) <= 1056, "too many sampled spans")
    identities: set[tuple[str, str]] = set()
    for span in spans:
        check(
            set(span)
            == {
                "name",
                "trace_id",
                "span_id",
                "parent_span_id",
                "start_time_ns",
                "end_time_ns",
                "status",
                "attributes",
            },
            "unexpected exported trace field",
        )
        check(span["name"] == "finserve.inference" and span["status"] == "OK", "trace outcome")
        check(span["parent_span_id"] is None, "fixture trace unexpectedly has a parent")
        identity = (span["trace_id"], span["span_id"])
        check(bool(re.fullmatch(r"[0-9a-f]{32}", identity[0])), "invalid trace identity")
        check(bool(re.fullmatch(r"[0-9a-f]{16}", identity[1])), "invalid span identity")
        check(identity not in identities, "duplicate exported span")
        identities.add(identity)
        check(
            receipt["started_epoch_s"]
            <= span["start_time_ns"] / 1e9
            <= span["end_time_ns"] / 1e9
            <= receipt["finished_epoch_s"],
            "span outside owned process lifetime",
        )
        check(
            span["attributes"]
            == {
                "gen_ai.request.model": "reference",
                "finserve.max_tokens": 32,
                "finserve.prompt_characters": 32,
                "finserve.outcome": "success",
                "gen_ai.usage.output_tokens": 32,
            },
            "unexpected trace metadata",
        )
    return len(spans)


def cohort(directory: Path, mode: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Recompute the complete raw population, exact output and timing before accepting summaries."""
    for filename in ("requests.jsonl", "manifest.json", "summary.json"):
        read(directory / "run" / filename)
    manifest = validate_evidence(directory / "run")
    configuration = manifest.configuration
    check(
        (configuration.requests, configuration.warmup, configuration.concurrency) == (1024, 32, 8),
        "changed load envelope",
    )
    check(configuration.mode == "closed" and configuration.timeout_s == 10, "changed arrival")
    check(
        configuration.model == "reference" and configuration.engine == "fixture-character-v1",
        "changed engine identity",
    )
    check(configuration.engine_config == "json-tracing-" + mode, "mode identity mismatch")
    check(configuration.revision == plan["repository"]["git_revision"], "source revision mismatch")
    check(manifest.workload_hash == plan["workload_hash"], "workload hash mismatch")
    check(
        manifest.workload.model_dump(mode="json") == plan["workload"], "embedded workload mismatch"
    )
    rows = [
        json.loads(line, object_pairs_hook=unique_object)
        for line in read(directory / "run/requests.jsonl").splitlines()
    ]
    check(
        all(
            row["success"]
            and row["status_code"] == 200
            and row["generated_tokens"] == 32
            and row["output"] == "0123456789abcdef" * 2
            for row in rows
        ),
        "fixture success/output/token mismatch",
    )
    measured = [row for row in rows if row["phase"] == "measured"]
    elapsed = manifest.measured_finished_s - manifest.measured_started_s
    expected = {
        "requests_per_second": len(measured) / elapsed,
        "tokens_per_second": sum(row["generated_tokens"] for row in measured) / elapsed,
        "e2e_p95_s": percentile([row["complete_s"] - row["send_s"] for row in measured], 0.95),
        "client_ttft_median_s": percentile(
            [row["first_content_s"] - row["send_s"] for row in measured], 0.5
        ),
    }
    summary = document(directory / "run/summary.json")
    check(
        all(
            math.isclose(value, summary[key], rel_tol=1e-12, abs_tol=1e-12)
            for key, value in expected.items()
        ),
        "independent timing recomputation mismatch",
    )
    receipt = document(directory / "process.json")
    check(receipt["mode"] == mode and receipt["status"] == "complete", "cohort not complete")
    check(receipt["summary"] == summary, "process summary mismatch")
    shutdown = receipt["shutdown"]
    check(shutdown["drained"] is True and shutdown["forced"] is False, "unresolved cleanup")
    check(
        shutdown["pid"] == receipt["ready"]["pid"] and shutdown["returncode"] == -15,
        "owned process identity/exit mismatch",
    )
    check(not receipt["ready"]["unowned_session_members"], "unowned process in session")
    spans = validate_traces(directory, mode, receipt)
    return {
        "cohort": directory.name,
        "mode": mode,
        "measured_requests": len(measured),
        "generated_tokens": sum(row["generated_tokens"] for row in measured),
        "warmup_requests": len(rows) - len(measured),
        "exported_spans": spans,
        "measured_seconds": elapsed,
        "started_epoch_s": receipt["started_epoch_s"],
        "finished_epoch_s": receipt["finished_epoch_s"],
        **expected,
    }


def analyze(root: Path) -> dict[str, Any]:
    """Validate archived source and six independent processes without executing archived code."""
    plan, result = document(root / "plan.json"), document(root / "result.json")
    check(tuple(plan["order"]) == ORDER and result["status"] == "complete", "experiment status")
    check(len(result["cohorts"]) == 6, "experiment cohort count")
    for name, digest in plan["source_sha256"].items():
        path = (root / "source" / name).resolve()
        check((root / "source").resolve() in path.parents, "archive path escape")
        check(hashlib.sha256(read(path)).hexdigest() == digest, "archived source hash mismatch")
    cohorts: list[dict[str, Any]] = []
    for index, mode in enumerate(ORDER, 1):
        directory = root / f"cohort-{index}-{mode}"
        check(
            document(directory / "process.json") == result["cohorts"][index - 1],
            "top-level cohort receipt mismatch",
        )
        cohorts.append(cohort(directory, mode, plan))
    check(
        all(
            left["finished_epoch_s"] <= right["started_epoch_s"]
            for left, right in zip(cohorts, cohorts[1:], strict=False)
        ),
        "overlapping cohorts",
    )
    artifacts = {
        str(path.relative_to(root)): hashlib.sha256(read(path)).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and "source" not in path.parts
    }
    return {
        "status": "verified",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "scope": "fixed local HTTP fixture; JSON export; no GPU/OTLP overhead claim",
        "measured_requests": sum(row["measured_requests"] for row in cohorts),
        "generated_tokens": sum(row["generated_tokens"] for row in cohorts),
        "archived_source_files": len(plan["source_sha256"]),
        "cohorts": cohorts,
        "input_sha256": artifacts,
        "interpretation": "Single balanced sequence with substantial within-mode variation; "
        "no isolated tracing penalty or reduction established.",
    }


def main() -> None:
    """Require fresh external report output so audits cannot silently overwrite evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    check(output != repository and repository not in output.parents, "report must be external")
    report = analyze(args.input.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        target.write(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps({"status": report["status"], "measured_requests": report["measured_requests"]})
    )


if __name__ == "__main__":
    main()
