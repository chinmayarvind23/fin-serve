"""Recompute frozen four-cohort routing sessions into fresh external audit artifacts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from finserve.benchmark.gpu import TelemetrySample
from finserve.benchmark.metrics import RequestRecord
from finserve.benchmark.routing_cohort import cohort_summary, gpu_summary
from finserve.benchmark.routing_workload import RoutingWorkload, frozen_workload
from finserve.benchmark.runner import prepare_output, write_json

SOURCES = frozenset(
    "benchmark/routing_cohort.py benchmark/routing_workload.py benchmark/routing_session.py "
    "benchmark/gpu.py benchmark/metrics.py benchmark/runner.py engines/ray_http.py "
    "engines/ray_backends.py engines/ray_serve.py engines/backend_observations.py "
    "engines/openai_adapter.py engines/chat_protocol.py contracts/inference.py "
    "contracts/routing.py scheduler/policy.py scheduler/router.py "
    "telemetry/propagation.py telemetry/tracing.py".split()
)
DATA_FILES = {
    "final-status.json",
    "initial-status.json",
    "gpu-samples.json",
    "status.jsonl",
    "requests.jsonl",
    "summary.json",
    "workload.json",
}


def read(path: Path) -> Any:
    """Read only retained local artifacts; typed workload and request models check boundaries."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Record exact input bytes; this audit never rewrites a measured run or its manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail on missing or inconsistent evidence instead of masking a failed population."""
    if not condition:
        raise ValueError(message)


def validate_artifacts(directory: Path, manifest: dict[str, Any]) -> None:
    """Bind every consumed file and archived runtime source to the exact retained set."""
    expected = DATA_FILES | {"source__" + path.replace("/", "__") for path in SOURCES}
    actual = {path.name for path in directory.iterdir() if path.name != "manifest.json"}
    require(set(manifest["artifacts"]) == actual == expected, "artifact set mismatch")
    require(set(manifest["sources"]) == set(SOURCES), "source mapping set mismatch")
    for name, sha in manifest["artifacts"].items():
        require(digest(directory / name) == sha, f"artifact changed: {name}")
    for source, sha in manifest["sources"].items():
        name = "source__" + source.replace("/", "__")
        require(manifest["artifacts"][name] == sha, f"source digest mismatch: {source}")


def validate_decisions(
    raw: list[dict[str, Any]], manifest: dict[str, Any], final: dict[str, Any]
) -> None:
    """Reconcile successful streams with actual router decisions, not just policy labels."""
    request_ids: set[str] = set()
    successful: set[str] = set()
    for row in raw:
        record = row["record"]
        identity = f"{manifest['run_id']}-{record['phase']}-{record['logical_id']}"
        require(row["request_id"] == identity, "request identity mismatch")
        require(identity not in request_ids, "duplicate request identity")
        request_ids.add(identity)
        if record["success"]:
            successful.add(identity)
    replicas = {row["replica_id"] for row in manifest["topology"]["backends"]}
    observed: set[str] = set()
    for row in final["recent_decisions"]:
        identity = row["request_id"]
        require(identity in request_ids and identity not in observed, "foreign/duplicate decision")
        observed.add(identity)
        require(row["decision"]["policy"] == manifest["policy"], "decision policy mismatch")
        require(row["decision"]["replica_id"] in replicas, "decision replica mismatch")
    require(successful <= observed, "successful request has no recorded decision")


def validate_group(manifests: list[dict[str, Any]]) -> None:
    """Reject mixed runtime/configuration populations and overlapping cohort intervals."""
    controls = ("topology", "sources", "tracing", "url", "workload_sha256")
    require(len({row["run_id"] for row in manifests}) == len(manifests), "reused cohort identity")
    for row in manifests:
        require(all(row[key] == manifests[0][key] for key in controls), "cohort controls differ")
        require(
            row["started_epoch_s"]
            <= row["measured_started_epoch_s"]
            < row["measured_ended_epoch_s"]
            <= row["finished_epoch_s"],
            "invalid cohort interval",
        )
    for left, right in zip(manifests, manifests[1:], strict=False):
        require(left["finished_epoch_s"] <= right["started_epoch_s"], "cohort intervals overlap")


def audit_cohort(directory: Path, policy: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate artifact hashes and recompute request/GPU summaries using repository helpers."""
    manifest, saved = (read(directory / path) for path in ("manifest.json", "summary.json"))
    require(
        manifest["status"] == "complete" and manifest["policy"] == policy,
        "missing completed attempted cohort or wrong policy order",
    )
    workload = RoutingWorkload.model_validate(read(directory / "workload.json"))
    require(
        workload.digest() == manifest["workload_sha256"] == frozen_workload().digest(),
        "frozen workload changed",
    )
    validate_artifacts(directory, manifest)
    raw = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    records = [RequestRecord.model_validate(row["record"]) for row in raw]
    require(len(raw) == len({row["request_id"] for row in raw}), "duplicate request identity")
    for phase, expected in (
        ("measured", len(workload.cases)),
        ("warmup", workload.warmup_per_cohort),
    ):
        selected = [record for record in records if record.phase == phase]
        require(
            len(selected) == expected
            and {row.logical_id for row in selected} == set(range(expected)),
            "missing or duplicated logical request",
        )
        for row in selected:
            case = workload.cases[row.logical_id]
            require(
                row.case_id == case.case_id
                and row.family == f"{case.input_class}-{case.max_tokens}",
                "record differs from frozen case",
            )
            require(
                row.generated_tokens is None or row.generated_tokens <= case.max_tokens,
                "generated count exceeds budget",
            )
    started, ended = manifest["measured_started_s"], manifest["measured_ended_s"]
    for row in records:
        if row.phase == "measured":
            require(
                started <= row.scheduled_s <= row.complete_s <= ended,
                "measured request lies outside interval",
            )
    require(
        saved["requests"] == cohort_summary(records, workload, ended - started),
        "request summary mismatch",
    )
    samples = [TelemetrySample.model_validate(row) for row in read(directory / "gpu-samples.json")]
    require(
        {device.uuid for sample in samples for device in sample.devices}
        == {manifest["topology"]["physical_gpu_uuid"]},
        "observed physical GPU differs from topology",
    )
    require(
        saved["physical_gpu"]
        == gpu_summary(
            samples,
            manifest["measured_started_epoch_s"],
            manifest["measured_ended_epoch_s"],
            started,
            ended,
        ),
        "physical GPU summary mismatch",
    )
    measured = [row for row in records if row.phase == "measured"]
    final = read(directory / "final-status.json")
    validate_decisions(raw, manifest, final)
    report = {
        "name": directory.name,
        "run_id": manifest["run_id"],
        "policy": policy,
        "repository": manifest["repository"],
        "workload_sha256": workload.digest(),
        "summary": saved,
        "errors": dict(Counter(row.error for row in measured if not row.success)),
        "failed_with_visible_output": sum(bool(row.output) for row in measured if not row.success),
        "decisions_by_replica_including_warmup": dict(
            Counter(row["decision"]["replica_id"] for row in final["recent_decisions"])
        ),
        "final_admission_statistics": {
            key: final.get(key)
            for key in ("pending_admissions", "admission_wait_count", "admission_wait_seconds")
        },
    }
    return report, {row.case_id: row.output for row in measured if row.success}


def audit_session(root: Path) -> dict[str, Any]:
    """Retain the overall failed session even when all four attempted load cohorts completed."""
    policies = frozen_workload().policies
    expected_names = [f"cohort-{index + 1}-{policy}" for index, policy in enumerate(policies)]
    require(
        sorted(path.name for path in root.glob("cohort-*")) == sorted(expected_names),
        "expected exactly the frozen four cohorts",
    )
    values = [
        audit_cohort(root / name, policy)
        for name, policy in zip(expected_names, policies, strict=True)
    ]
    validate_group([read(root / name / "manifest.json") for name in expected_names])
    pairs: list[dict[str, Any]] = []
    for first, second in ((0, 1), (3, 2)):
        left, right = values[first][1], values[second][1]
        common = left.keys() & right.keys()
        pairs.append(
            {
                "left": expected_names[first],
                "right": expected_names[second],
                "offered_pairs": 64,
                "both_successful": len(common),
                "equal_output_pairs": sum(left[key] == right[key] for key in common),
            }
        )
    return {
        "scope": "Frozen local load and exact-output agreement; no task correctness estimate",
        "session": read(root / "manifest.json"),
        "cohorts": [value[0] for value in values],
        "pairs": pairs,
        "audit_source_sha256": digest(Path(__file__)),
        "artifact_sha256": {
            path.relative_to(root).as_posix(): digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        },
    }


def main() -> None:
    """Use explicit input/output paths and refuse in-repository or existing output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_session(args.session)
    write_json(prepare_output(args.output) / "routing-audit.json", report)


if __name__ == "__main__":
    main()
