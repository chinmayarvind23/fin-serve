"""Exercise quality evidence provenance and bounded HTTP failure paths independently."""

import asyncio
import hashlib
import importlib.util
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from finserve.evaluation.quality import default_suite


@pytest.fixture
def runner() -> ModuleType:
    """Load the standalone CLI without creating a new public application package."""
    path = Path(__file__).resolve().parents[2] / "scripts/run_quality.py"
    spec = importlib.util.spec_from_file_location("quality_runner_review", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fault", ["none", "hash", "suite", "answers", "status"])
def test_reference_artifacts_are_bound(runner: ModuleType, tmp_path: Path, fault: str) -> None:
    """Same case IDs do not excuse changed prompts, changed bytes or incomplete runs."""
    suite = default_suite()
    answers = {case.case_id: case.expected for case in suite.cases}
    payload = json.dumps(answers).encode()
    manifest = {
        "suite": suite.model_dump(),
        "suite_hash": suite.digest(),
        "answers_sha256": hashlib.sha256(payload).hexdigest(),
        "status": "completed",
    }
    if fault == "hash":
        manifest["suite_hash"] = "0" * 64
    if fault == "suite":
        modified = suite.model_copy(update={"suite_id": "changed-prompts"})
        manifest["suite"] = modified.model_dump()
    if fault == "answers":
        payload += b" "
    if fault == "status":
        manifest["status"] = "interrupted"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "answers.json").write_bytes(payload)
    if fault == "none":
        assert runner.reference_answers(tmp_path, suite) == answers
    else:
        with pytest.raises(ValueError):
            runner.reference_answers(tmp_path, suite)


class AdversarialBody(httpx.AsyncByteStream):
    """Model a slow peer or oversized body while exposing transport ownership."""

    def __init__(self, oversized: bool) -> None:
        """Track whether the response context releases the underlying stream."""
        self.oversized = oversized
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """The slow stream keeps making progress; an inactivity timeout alone cannot stop it."""
        if self.oversized:
            yield b"x" * 1_048_577
        else:
            while True:
                await asyncio.sleep(0.001)
                yield b" "

    async def aclose(self) -> None:
        """Observe close on both validation error and cancellation-induced deadline."""
        self.closed = True


@pytest.mark.parametrize("oversized", [True, False])
async def test_response_limit_and_total_deadline_close_transport(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, oversized: bool
) -> None:
    """An overall deadline must stop a continuously trickling response and close it."""
    stream = AdversarialBody(oversized)
    real_timeout = asyncio.timeout

    def short_timeout(delay: float) -> asyncio.Timeout:
        """Accelerate the deadline without modifying production duration semantics."""
        assert delay == 30
        return real_timeout(0.025)

    monkeypatch.setattr(runner.asyncio, "timeout", short_timeout)

    def respond(request: httpx.Request) -> httpx.Response:
        """Verify generation settings are explicit in the actual HTTP request."""
        payload = json.loads(request.content)
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 128
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError if oversized else TimeoutError):
            await runner.answer(client, "http://test/completions", "test", "prompt")
    assert stream.closed


def test_interrupted_cli_persists_terminal_status(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled collection cannot leave its manifest looking complete or reusable."""
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(default_suite().model_dump_json())
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_quality.py",
            "--suite",
            str(suite_path),
            "--output",
            str(output),
            "--url",
            "http://test/completions",
            "--model",
            "test",
            "--model-revision",
            "commit-model",
            "--tokenizer-revision",
            "commit-tokenizer",
            "--engine",
            "test-engine",
            "--engine-config",
            "fixed",
        ],
    )

    async def cancel(*args: object) -> dict[str, str]:
        """Inject cancellation after initial provenance has been persisted."""
        raise asyncio.CancelledError

    monkeypatch.setattr(runner, "collect_answers", cancel)
    with pytest.raises(asyncio.CancelledError):
        runner.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["error"] == "CancelledError"
    assert manifest["generation"]["temperature"] == 0.0
    assert manifest["evaluator_sha256"]
    assert not (output / "quality.json").exists()
