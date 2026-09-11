"""Run a frozen deterministic quality suite over the actual serving HTTP boundary."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import httpx
from pydantic import TypeAdapter

from finserve.benchmark.runner import prepare_output, write_json
from finserve.evaluation import quality
from finserve.evaluation.quality import GoldenSuite, evaluate_quality


async def answer(client: httpx.AsyncClient, url: str, model: str, prompt: str) -> str:
    """Bound total case time and decoded bytes, including a peer trickling response chunks."""
    async with asyncio.timeout(30):
        async with client.stream(
            "POST",
            url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.0,
                "stream": False,
            },
        ) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > 1_048_576:
                    raise ValueError("quality response exceeds byte limit")
                body.extend(chunk)
    value = json.loads(body)["choices"][0]["text"]
    if not isinstance(value, str):
        raise ValueError("invalid completion shape")
    return value


def reference_answers(path: Path, suite: GoldenSuite) -> dict[str, str]:
    """Bind baseline answers to the exact frozen suite and content hash before comparison."""
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["suite_hash"] != suite.digest():
        raise ValueError("reference suite differs")
    frozen = GoldenSuite.model_validate(manifest["suite"])
    if frozen.digest() != suite.digest():
        raise ValueError("reference suite manifest was modified")
    payload = (path / "answers.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["answers_sha256"]:
        raise ValueError("reference answer artifact was modified")
    if manifest["status"] != "completed":
        raise ValueError("reference quality run did not complete")
    return TypeAdapter(dict[str, str]).validate_json(payload, strict=True)


async def collect_answers(url: str, model: str, suite: GoldenSuite, output: Path) -> dict[str, str]:
    """Retain failed logical cases; missing answers remain hard failures in the evaluator."""
    answers: dict[str, str] = {}
    headers = (
        {"Authorization": f"Bearer {os.environ['FINSERVE_API_KEY']}"}
        if os.getenv("FINSERVE_API_KEY")
        else {}
    )
    async with httpx.AsyncClient(timeout=30, headers=headers, trust_env=False) as client:
        with (output / "requests.jsonl").open("x", encoding="utf-8") as raw:
            for case in suite.cases:
                started = time.perf_counter()
                record: dict[str, object] = {"case_id": case.case_id}
                try:
                    value = await answer(client, url, model, case.prompt)
                    answers[case.case_id] = value
                    record["output"] = value
                except (
                    httpx.HTTPError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    IndexError,
                    TypeError,
                ) as exc:
                    record["error"] = type(exc).__name__
                    if isinstance(exc, httpx.HTTPStatusError):
                        record["status_code"] = exc.response.status_code
                record["elapsed_seconds"] = time.perf_counter() - started
                raw.write(json.dumps(record, allow_nan=False) + "\n")
                raw.flush()
    return answers


def main() -> None:
    """Use the reference answer artifact only for parity, never as a replacement for truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    suite = GoldenSuite.model_validate_json(args.suite.read_text())
    reference = reference_answers(args.reference, suite) if args.reference else None
    output = prepare_output(args.output)
    repository = Path(__file__).resolve().parents[1]
    manifest = {
        "run_id": str(uuid.uuid4()),
        "started_at_epoch_s": time.time(),
        "status": "running",
        "suite": suite.model_dump(),
        "suite_hash": suite.digest(),
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "engine": args.engine,
        "engine_config": args.engine_config,
        "generation": {"temperature": 0.0, "max_tokens": 128, "stream": False},
        "evaluator_sha256": hashlib.sha256(Path(quality.__file__).read_bytes()).hexdigest(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True
        ).splitlines(),
        "reference_manifest_sha256": hashlib.sha256(
            (args.reference / "manifest.json").read_bytes()
        ).hexdigest()
        if args.reference
        else None,
    }
    write_json(output / "manifest.json", manifest)
    try:
        answers = asyncio.run(collect_answers(args.url, args.model, suite, output))
    except BaseException as exc:
        manifest.update(status="interrupted", error=type(exc).__name__)
        write_json(output / "manifest.json", manifest)
        raise
    write_json(output / "answers.json", answers)
    report = evaluate_quality(suite, reference if reference is not None else answers, answers)
    report["reference_scope"] = "external baseline" if args.reference else "self; accuracy only"
    write_json(output / "quality.json", report)
    manifest.update(
        status="completed",
        finished_at_epoch_s=time.time(),
        answers_sha256=hashlib.sha256((output / "answers.json").read_bytes()).hexdigest(),
    )
    write_json(output / "manifest.json", manifest)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
