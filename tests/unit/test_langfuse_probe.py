"""Local collector configuration and queried-evidence checks without starting Docker services."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from finserve.telemetry.langfuse_probe import (
    exporter_environment,
    local_url,
    prepare_secrets,
    read_observations,
    read_secrets,
    verified_observation,
)


def test_private_bootstrap_is_exclusive_and_bounded(tmp_path: Path) -> None:
    """Generate independent credentials without replacing an already initialized deployment."""
    path = tmp_path / "private.env"
    prepare_secrets(path)
    values = read_secrets(path)
    assert len(values) == len(set(values.values())) == 10
    assert len(values["LF_ENCRYPTION_KEY"]) == 64
    with pytest.raises(FileExistsError):
        prepare_secrets(path)
    path.write_text(path.read_text() + "LF_PUBLIC_KEY=private-duplicate-" + "a" * 32 + "\n")
    with pytest.raises(ValueError, match="Invalid"):
        read_secrets(path)


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example:3037",
        "https://remote.example:3037",
        "http://secret@localhost:3037",
        "http://localhost:3037/?secret=x",
        "http://localhost:3037/#secret",
        "http://localhost:999999",
        "http://localhost:3037/private",
        "http://localhost:3037\n",
    ],
)
def test_local_credentials_cannot_target_remote_or_decorated_url(url: str) -> None:
    """The standalone verifier is intentionally narrower than the production OTLP exporter."""
    with pytest.raises(ValueError, match="loopback") as error:
        local_url(url)
    assert "secret" not in str(error.value)
    assert local_url("http://127.0.0.1:3037/") == "http://127.0.0.1:3037"


def test_probe_environment_restores_previous_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone probe cannot leave its credentials configured after failure or success."""
    import os

    monkeypatch.setenv("FINSERVE_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.delenv("FINSERVE_OTLP_AUTHORIZATION", raising=False)
    with exporter_environment("http://localhost:3037", "Basic synthetic-private"):
        assert os.environ["FINSERVE_OTLP_PROTOCOL"] == "langfuse-v4"
        assert os.environ["FINSERVE_OTLP_AUTHORIZATION"] == "Basic synthetic-private"
    assert "FINSERVE_OTLP_AUTHORIZATION" not in os.environ
    assert os.environ["FINSERVE_OTLP_ENDPOINT"] == "http://localhost:4318/v1/traces"


def test_stored_observation_identity_and_privacy() -> None:
    """HTTP success or a different trace is insufficient to establish database ingestion."""
    row: dict[str, Any] = {
        "id": "span",
        "traceId": "trace",
        "name": "finserve.inference",
        "input": None,
        "output": None,
    }
    document = {"data": [row]}
    assert verified_observation(document, "trace", "span", ["private-marker"]) == row
    assert verified_observation(document, "other", "span", ["private-marker"]) is None
    row["metadata"] = {"prompt": "private-marker"}
    with pytest.raises(ValueError, match="forbidden"):
        verified_observation(document, "trace", "span", ["private-marker"])
    del row["metadata"]
    row["input"] = "unexpected"
    with pytest.raises(ValueError, match="input or output"):
        verified_observation(document, "trace", "span", [])


@pytest.mark.parametrize("mode", ["auth", "oversize", "shape"])
async def test_observation_query_rejects_failures(mode: str) -> None:
    """Public API errors, large replies and wrong shapes do not become stored-span evidence."""
    response = {
        "auth": httpx.Response(401),
        "oversize": httpx.Response(200, content=b"x" * 262145),
        "shape": httpx.Response(200, content=json.dumps({"data": {}})),
    }[mode]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(ValueError):
            await read_observations(client, "http://localhost:3037", {}, 1)
