"""Execute six named semantic faults in isolated copies, preserving all evidence.

This is a curated mutation check, not exhaustive mutmut coverage. Each mutant must
compile and trigger an assertion failure in the unchanged baseline tests to count
as killed. Collection errors, timeouts and other runner failures are inconclusive.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

TESTS = (
    "tests/unit/test_admission.py",
    "tests/unit/test_metrics.py",
    "tests/unit/test_quality.py",
)


@dataclass(frozen=True)
class Mutation:
    """Exact one-occurrence substitutions fail closed when implementation has changed."""

    name: str
    file: str
    before: str
    after: str


MUTATIONS = (
    Mutation(
        "admit_at_capacity",
        "src/finserve/gateway/admission.py",
        "if self.active >= self.limit:",
        "if self.active > self.limit:",
    ),
    Mutation(
        "ignore_lease_underflow",
        "src/finserve/gateway/admission.py",
        'raise RuntimeError("admission lease released twice")',
        "return",
    ),
    Mutation(
        "replace_percentile_interpolation",
        "src/finserve/benchmark/metrics.py",
        "return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)",
        "return ordered[lower]",
    ),
    Mutation(
        "exclude_failed_requests_from_denominator",
        "src/finserve/benchmark/metrics.py",
        '"success_rate": len(succeeded) / len(measured) if measured else None,',
        '"success_rate": 1.0 if succeeded else None,',
    ),
    Mutation(
        "count_requests_as_tokens",
        "src/finserve/benchmark/metrics.py",
        "tokens = sum(row.generated_tokens or 0 for row in succeeded) if known_tokens else None",
        "tokens = len(succeeded) if known_tokens else None",
    ),
    Mutation(
        "ignore_hard_quality_failures",
        "src/finserve/evaluation/quality.py",
        '"passed": not hard_failures\n        and parity_score >= config.minimum_parity',
        '"passed": parity_score >= config.minimum_parity',
    ),
)

BOOTSTRAP = '''"""Verify import isolation before executing the unchanged test files."""
import importlib
import pathlib
import sys
import pytest

source = pathlib.Path("src").resolve()
for name in ("finserve.gateway.admission", "finserve.benchmark.metrics",
             "finserve.evaluation.quality"):
    module = importlib.import_module(name)
    origin = pathlib.Path(module.__file__).resolve()
    assert origin.is_relative_to(source), f"Import escaped isolated copy: {origin}"
    print(f"isolated_import {name} {origin}", flush=True)
raise SystemExit(pytest.main(sys.argv[1:]))
'''


def write_json(path: Path, value: object) -> None:
    """Persist strict JSON after each result so interruptions retain completed observations."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(source: Path, destination: Path) -> None:
    """Copy source and selected tests while excluding bytecode and unrelated test dependencies."""
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        source / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for relative in (*TESTS, "pyproject.toml"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    (destination / "verify_imports.py").write_text(BOOTSTRAP, encoding="utf-8")


def source_identity(source: Path) -> dict[str, object]:
    """Store content hashes because a clean commit alone cannot describe concurrent local edits."""
    paths = sorted({mutation.file for mutation in MUTATIONS} | set(TESTS) | {"pyproject.toml"})
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in paths}
    return {"sha256": hashes}


def junit_result(path: Path) -> dict[str, object]:
    """Only real test assertion failures count; collection errors remain inconclusive."""
    if not path.exists():
        return {"failures": [], "errors": [], "tests": 0}
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failures: list[str] = []
    errors: list[str] = []
    for case in cases:
        identity = f"{case.get('classname')}::{case.get('name')}"
        if case.find("failure") is not None:
            failures.append(identity)
        if case.find("error") is not None:
            errors.append(identity)
    return {"failures": failures, "errors": errors, "tests": len(cases)}


def run_tests(case: Path, timeout: float) -> dict[str, object]:
    """A separate interpreter and source path prevent module caches from hiding mutations."""
    command = [sys.executable, "verify_imports.py", *TESTS, "-q", "--junitxml=pytest.xml"]
    environment = {**os.environ, "PYTHONPATH": str(case / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=case,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        (case / "pytest.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        evidence = junit_result(case / "pytest.xml")
        code: int | None = result.returncode
        status = "passed" if code == 0 else "inconclusive"
        if code == 1 and evidence["failures"] and not evidence["errors"]:
            status = "assertion_failures"
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"") + (exc.stderr or b"")
        (case / "pytest.log").write_bytes(output)
        code, status, evidence = None, "timeout", {}
    return {
        "status": status,
        "exit_code": code,
        "elapsed_seconds": time.monotonic() - started,
        "command": command,
        **evidence,
    }


def execute_mutation(
    baseline: Path, root: Path, mutation: Mutation, timeout: float
) -> dict[str, object]:
    """Each fault starts from the same baseline; source revisions cannot accumulate across cases."""
    case = root / mutation.name
    snapshot(baseline, case)
    path = case / mutation.file
    original = path.read_text(encoding="utf-8")
    if original.count(mutation.before) != 1:
        return {"name": mutation.name, "status": "inapplicable", "mutation": asdict(mutation)}
    changed = original.replace(mutation.before, mutation.after, 1)
    ast.parse(changed, filename=mutation.file)
    path.write_text(changed, encoding="utf-8")
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        changed.splitlines(keepends=True),
        fromfile=f"baseline/{mutation.file}",
        tofile=f"{mutation.name}/{mutation.file}",
    )
    (case / "mutation.diff").write_text("".join(diff), encoding="utf-8")
    result = run_tests(case, timeout)
    result["status"] = {"passed": "survived", "assertion_failures": "killed"}.get(
        str(result["status"]), result["status"]
    )
    return {"name": mutation.name, "mutation": asdict(mutation), **result}


def run_suite(repository: Path, output: Path, timeout: float) -> dict[str, object]:
    """Require a fresh external evidence directory and a passing baseline before any mutation."""
    if output == repository or repository in output.parents:
        raise ValueError("mutation evidence must be outside the source repository")
    output.mkdir(parents=True, exist_ok=False)
    baseline = output / "baseline"
    snapshot(repository, baseline)
    manifest: dict[str, object] = {
        "kind": "curated semantic mutants; not exhaustive mutation coverage",
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("pytest", "pydantic", "httpx")
        },
        "source": source_identity(baseline),
        "mutations": [asdict(item) for item in MUTATIONS],
    }
    write_json(output / "manifest.json", manifest)
    clean = run_tests(baseline, timeout)
    results: list[dict[str, object]] = []
    report: dict[str, object] = {"baseline": clean, "mutants": results, "complete": False}
    write_json(output / "summary.json", report)
    if clean["status"] != "passed":
        return report
    for mutation in MUTATIONS:
        result = execute_mutation(baseline, output, mutation, timeout)
        results.append(result)
        print(f"{mutation.name}: {result['status']}", flush=True)
        write_json(output / "summary.json", report)
    report["complete"] = True
    report["killed"] = sum(result["status"] == "killed" for result in results)
    report["survived"] = sum(result["status"] == "survived" for result in results)
    report["passed"] = all(result["status"] == "killed" for result in results)
    write_json(output / "summary.json", report)
    return report


def main() -> None:
    """Expose a bounded Linux-friendly check without installing tools into the project runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.timeout <= 300:
        parser.error("timeout must be between 0 and 300 seconds")
    repository = Path(__file__).resolve().parents[1]
    report = run_suite(repository, args.output.resolve(), args.timeout)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report.get("passed") is True else 1)


if __name__ == "__main__":
    main()
