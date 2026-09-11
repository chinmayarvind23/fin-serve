"""CLI benchmark clients authenticate over real HTTP without putting credentials in evidence."""

import asyncio
import json
import sys
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from finserve.benchmark import experiment as experiment_module
from finserve.benchmark import runner
from finserve.benchmark.gpu import TelemetrySample
from finserve.benchmark.runner import RunConfig, benchmark_client
from finserve.benchmark.workload import default_workload

SECRET = "fixture-benchmark-bearer"


@pytest.fixture
def authenticated_http() -> Generator[str]:
    """Use an ephemeral local socket with deterministic auth and no external inference service."""

    class Handler(BaseHTTPRequestHandler):
        """Require the fixture credential before returning a valid completion stream."""

        def do_POST(self) -> None:
            """A rejected credential remains an HTTP failure in the raw benchmark population."""
            size = int(self.headers["Content-Length"])
            assert 0 < size < 131072
            self.rfile.read(size)
            accepted = self.headers.get("Authorization") == "Bearer " + SECRET
            self.send_response(200 if accepted else 401)
            self.send_header(
                "Content-Type", "text/event-stream" if accepted else "application/json"
            )
            self.end_headers()
            self.wfile.write(
                b'data: {"choices":[{"text":"ok","finish_reason":"stop"}],'
                b'"usage":{"completion_tokens":1}}\n\n'
                b"data: [DONE]\n\n"
                if accepted
                else b'{"error":"unauthorized"}'
            )

        def log_message(self, format: str, *args: object) -> None:
            """Suppress fixture logs so request details never enter test output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/completions"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("entrypoint", ["runner", "experiment"])
@pytest.mark.parametrize(
    "credential", [SECRET, "wrong-fixture-key", None], ids=["valid", "rejected", "missing"]
)
async def test_benchmark_entrypoints_authenticate_without_persisting_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    authenticated_http: str,
    entrypoint: str,
    credential: str | None,
) -> None:
    """Both entrypoints use common auth; rejected auth stays measurable and proxies are ignored."""
    if credential is None:
        monkeypatch.delenv("FINSERVE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("FINSERVE_API_KEY", credential)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    output = tmp_path / entrypoint
    config = RunConfig(requests=2, warmup=1, concurrency=1)
    if entrypoint == "runner":
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "runner",
                "--url",
                authenticated_http,
                "--output",
                str(output),
                "--requests",
                "2",
                "--warmup",
                "1",
                "--concurrency",
                "1",
            ],
        )
        await asyncio.to_thread(runner.main)
        directory = output
    else:

        def no_gpu() -> TelemetrySample:
            """Explicit unavailable fixture telemetry avoids all GPU or nvidia-smi work."""
            return TelemetrySample(epoch_s=1, collection_seconds=0, devices=[], error="fixture")

        monkeypatch.setattr(experiment_module, "collect", no_gpu)
        await experiment_module.experiment(authenticated_http, output, default_workload(), config)
        directory = output / "run"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["offered_requests"] == 2
    assert summary["successful_requests"] == (2 if credential == SECRET else 0)
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {row["status_code"] for row in rows} == ({200} if credential == SECRET else {401})
    persisted = "\n".join(path.read_text() for path in output.rglob("*.json*"))
    assert SECRET not in persisted and "wrong-fixture-key" not in persisted
    assert "Authorization" not in persisted and "FINSERVE_API_KEY" not in persisted
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err


@pytest.mark.parametrize("credential", [None, ""], ids=["unset", "empty"])
async def test_no_key_preserves_unauthenticated_client(
    monkeypatch: pytest.MonkeyPatch, credential: str | None
) -> None:
    """Historical no-key runs keep identical request fields and omit Authorization entirely."""
    if credential is None:
        monkeypatch.delenv("FINSERVE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("FINSERVE_API_KEY", credential)
    async with benchmark_client(RunConfig()) as client:
        assert "authorization" not in client.headers
        assert not client.trust_env and not client.follow_redirects


@pytest.mark.parametrize(
    "credential",
    ["private\nvalue", "private\x7fvalue", "é", "x" * 4097],
    ids=["newline", "control", "non-ascii", "oversize"],
)
def test_invalid_header_credentials_fail_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch, credential: str
) -> None:
    """Reject invalid credentials with a static message before constructing HTTP work."""
    monkeypatch.setenv("FINSERVE_API_KEY", credential)
    with pytest.raises(ValueError, match="^Invalid benchmark API credential$"):
        benchmark_client(RunConfig())
