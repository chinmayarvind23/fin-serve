"""Verify one synthetic sanitized span in a local Langfuse database through its public API."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from opentelemetry.trace import Status, StatusCode

from finserve.telemetry.tracing import from_env

_SECRET_NAMES = (
    "LF_POSTGRES_PASSWORD",
    "LF_CLICKHOUSE_PASSWORD",
    "LF_REDIS_PASSWORD",
    "LF_MINIO_PASSWORD",
    "LF_SALT",
    "LF_ENCRYPTION_KEY",
    "LF_NEXTAUTH_SECRET",
    "LF_OWNER_PASSWORD",
    "LF_PUBLIC_KEY",
    "LF_SECRET_KEY",
)


def external(path: Path) -> Path:
    """Keep credentials and actual evidence outside the source tree, including resolved symlinks."""
    resolved = path.resolve()
    repository = Path(__file__).resolve().parents[3]
    if resolved == repository or repository in resolved.parents:
        raise ValueError("Langfuse secrets and evidence must be outside the source repository")
    return resolved


def prepare_secrets(path: Path) -> None:
    """Create independent bootstrap secrets exclusively; POSIX mode600 complements host ACLs."""
    output = external(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = {name: secrets.token_hex(32) for name in _SECRET_NAMES}
    values["LF_PUBLIC_KEY"] = "pk-lf-" + uuid4().hex
    values["LF_SECRET_KEY"] = "sk-lf-" + secrets.token_hex(32)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as target:
        target.write("".join(f"{key}={value}\n" for key, value in values.items()))


def read_secrets(path: Path) -> dict[str, str]:
    """Read only this tool's bounded format, rejecting duplicate/missing/extra configuration."""
    with external(path).open("rb") as source:
        raw = source.read(8193)
    if len(raw) > 8192:
        raise ValueError("Invalid local Langfuse secret file")
    values: dict[str, str] = {}
    try:
        for line in raw.decode("ascii").splitlines():
            key, value = line.split("=", 1)
            if key in values or key not in _SECRET_NAMES or not 32 <= len(value) <= 128:
                raise ValueError
            if any(not (character.isalnum() or character == "-") for character in value):
                raise ValueError
            values[key] = value
        if set(values) != set(_SECRET_NAMES):
            raise ValueError
    except (UnicodeError, ValueError):
        raise ValueError("Invalid local Langfuse secret file") from None
    return values


def local_url(value: str) -> str:
    """This verification tool cannot send generated local credentials to a remote service."""
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and "?" not in value
            and "#" not in value
            and "\\" not in value
            and all(33 <= ord(character) <= 126 for character in value)
            and parsed.port is not None
            and 1 <= parsed.port <= 65535
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Local Langfuse verification requires a loopback HTTP base URL")
    return value.rstrip("/")


@contextmanager
def exporter_environment(url: str, authorization: str) -> Generator[None]:
    """Scope process-local exporter configuration to this standalone probe; never persist auth."""
    configuration = {
        "FINSERVE_TRACE_PATH": "",
        "FINSERVE_OTLP_ENDPOINT": url + "/api/public/otel/v1/traces",
        "FINSERVE_OTLP_AUTHORIZATION": authorization,
        "FINSERVE_OTLP_PROTOCOL": "langfuse-v4",
        "FINSERVE_TRACE_SAMPLE_RATIO": "1",
        "FINSERVE_OTLP_TIMEOUT_SECONDS": "5",
    }
    previous = {key: os.environ.get(key) for key in configuration}
    os.environ.update(configuration)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def read_observations(
    client: httpx.AsyncClient, url: str, parameters: dict[str, str], remaining_seconds: float
) -> dict[str, Any]:
    """Bound every API attempt in bytes and elapsed time, including trickling responses."""
    async with asyncio.timeout(remaining_seconds):
        async with client.stream(
            "GET", url + "/api/public/v2/observations", params=parameters
        ) as response:
            if response.status_code in {401, 403}:
                raise ValueError("Langfuse API authentication rejected")
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > 262144:
                    raise ValueError("Langfuse query response exceeds bound")
                body.extend(chunk)
    document: Any = json.loads(body)
    if not isinstance(document, dict):
        raise ValueError("Invalid Langfuse observations response")
    value = cast(dict[str, Any], document)
    if not isinstance(value.get("data"), list):
        raise ValueError("Invalid Langfuse observations response")
    return value


def verified_observation(
    document: dict[str, Any], trace_id: str, span_id: str, forbidden: list[str]
) -> dict[str, Any] | None:
    """Require stored identity/name and absent private payload, not just a collector HTTP200."""
    serialized = json.dumps(document, sort_keys=True)
    if any(value in serialized for value in forbidden):
        raise ValueError("Langfuse stored forbidden trace content")
    rows = [
        row
        for row in document["data"]
        if row.get("id") == span_id and row.get("traceId") == trace_id
    ]
    if not rows:
        return None
    if len(rows) != 1 or rows[0].get("name") != "finserve.inference":
        raise ValueError("Langfuse stored observation identity mismatch")
    row = rows[0]
    if row.get("input") is not None or row.get("output") is not None:
        raise ValueError("Langfuse observation unexpectedly contains input or output")
    return row


async def verify(
    path: Path, secret_file: Path, base_url: str, wait_seconds: float = 120
) -> dict[str, Any]:
    """Write a terminal manifest for success, failed ingestion, query failure or interruption."""
    if not 1 <= wait_seconds <= 300:
        raise ValueError("Langfuse verification wait must be within 1–300 seconds")
    url, keys = local_url(base_url), read_secrets(secret_file)
    output = external(path)
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "status": "running",
        "run_id": uuid4().hex,
        "started_at": datetime.now(UTC).isoformat(),
        "scope": (
            "local Langfuse OTLP/database-query integration; synthetic span; no model inference"
        ),
        "base_url": url,
        "server_version": "4.33.0",
    }
    manifest_path = output / "verification.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    authorization = (
        "Basic "
        + base64.b64encode(f"{keys['LF_PUBLIC_KEY']}:{keys['LF_SECRET_KEY']}".encode()).decode()
    )
    marker = "private-probe-" + secrets.token_hex(16)
    try:
        with exporter_environment(url, authorization):
            runtime = from_env()
            assert runtime is not None
            try:
                span = runtime.tracer.start_span(
                    "finserve.inference",
                    attributes={
                        "gen_ai.request.model": "finserve-langfuse-fixture",
                        "gen_ai.usage.output_tokens": 7,
                        "finserve.outcome": "success",
                        "prompt": marker,
                    },
                )
                span.add_event(marker, {"secret": marker})
                span.set_status(Status(StatusCode.ERROR, marker))
                context = span.get_span_context()
                trace_id, span_id = f"{context.trace_id:032x}", f"{context.span_id:016x}"
                span.end()
            finally:
                cleanup = asyncio.create_task(asyncio.to_thread(runtime.close))
                cancelled = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled = True
                cleanup.result()
                if cancelled:
                    raise asyncio.CancelledError
            manifest.update(
                trace_id=trace_id, span_id=span_id, export_failures=runtime.exporter.failures
            )
            if runtime.exporter.failures:
                raise ValueError("Langfuse OTLP export failed")
        now = datetime.now(UTC)
        parameters = {
            "traceId": trace_id,
            "limit": "10",
            "fields": "core,basic,io,metadata,model,usage",
            "fromStartTime": (now - timedelta(minutes=5)).isoformat(),
            "toStartTime": (now + timedelta(minutes=5)).isoformat(),
        }
        deadline = time.monotonic() + wait_seconds
        async with httpx.AsyncClient(
            headers={"Authorization": authorization}, trust_env=False, follow_redirects=False
        ) as client:
            while time.monotonic() < deadline:
                document = await read_observations(
                    client, url, parameters, min(5, deadline - time.monotonic())
                )
                observed = verified_observation(
                    document, trace_id, span_id, [marker, *keys.values(), authorization]
                )
                if observed is not None:
                    (output / "observation.json").write_text(
                        json.dumps(observed, indent=2), encoding="utf-8"
                    )
                    manifest.update(
                        status="verified",
                        observed_name=observed["name"],
                        privacy_checks_passed=True,
                    )
                    break
                await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))
            else:
                raise TimeoutError("Langfuse observation was not queryable within the budget")
    except BaseException as exc:
        manifest.update(
            status="failed" if isinstance(exc, Exception) else "interrupted",
            error=type(exc).__name__,
        )
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["artifacts"] = {
            item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in output.iterdir()
            if item.is_file() and item != manifest_path
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Keep bootstrap and verification explicit; neither operation starts or deletes containers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify"))
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:3037")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare_secrets(args.secrets_file)
        print("Private bootstrap environment created")
        return
    if args.output is None:
        parser.error("verify requires --output")
    try:
        result = asyncio.run(verify(args.output, args.secrets_file, args.base_url))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({"status": result["status"], "trace_id": result["trace_id"]}))


if __name__ == "__main__":
    main()
