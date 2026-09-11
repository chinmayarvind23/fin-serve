"""Producer quality evidence records every request and rejects changed artifacts on reload."""

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from finserve.benchmark.runner import RunConfig
from finserve.contracts.deployment import Revision
from finserve.contracts.producer import QualityCollectionSpec
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.evaluation.quality import default_suite
from finserve.registry.quality_collection import (
    QualityCollectionResult,
    bounded_file,
    collect_quality,
    load_quality_collection,
    write_json,
)


def specification() -> QualityCollectionSpec:
    """Synthetic identities exercise binding without claiming a built model or image."""
    profile = ServingProfileV1(
        engine="fixture",
        engine_version="1.0.0",
        model_revision="a" * 40,
        tokenizer_revision="a" * 40,
        model_manifest_sha256="b" * 64,
        tokenizer_manifest_sha256="b" * 64,
        base_url="http://fixture/v1",
        served_model="fixture",
    )
    revision = Revision(
        revision_id="fixture",
        model_revision=profile.model_revision,
        tokenizer_revision=profile.tokenizer_revision,
        source_revision="c" * 40,
        image_digest="sha256:" + "d" * 64,
        config_digest=profile.digest(),
        engine=profile.engine,
        engine_config=profile.engine_parameters_json,
    )
    config = RunConfig(
        model="fixture",
        revision=revision.source_revision,
        model_revision=revision.model_revision,
        tokenizer_revision=revision.tokenizer_revision,
        engine=revision.engine,
        engine_config=revision.engine_config,
        image_digest=revision.image_digest,
        config_digest=revision.config_digest,
        request_api="chat",
        system_prompt="Only answer.",
        chat_template_sha256="e" * 64,
    )
    return QualityCollectionSpec(
        collection_id="quality-01",
        profile=profile,
        revision=revision,
        suite=default_suite(),
        configuration=config,
    )


async def test_collection_preserves_failure_and_raw_mapping(tmp_path: Path) -> None:
    """A503 remains one of the planned cases and cannot become an empty successful answer."""
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return two successful fixture responses around a real HTTP error status."""
        nonlocal count
        count += 1
        assert request.url.path == "/v1/chat/completions"
        if count == 2:
            return httpx.Response(503)
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"5"}}]}\n\n'
                'data: {"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await collect_quality(client, specification(), tmp_path / "collection")
        with pytest.raises(FileExistsError):
            await collect_quality(client, specification(), tmp_path / "collection")
    assert result.planned == result.recorded == 3 and result.successful == 2
    assert set(result.outputs) == {"margin", "general"}
    assert load_quality_collection(tmp_path / "collection") == (specification(), result)


@pytest.mark.parametrize(
    "target",
    ["result", "request", "extra", "truncated", "identity", "offered", "unobserved", "duplicate"],
)
async def test_collection_rejects_tampered_artifacts(tmp_path: Path, target: str) -> None:
    """Digest, request identity, record count and output reconstruction are independent checks."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as client:
        await collect_quality(client, specification(), tmp_path / "collection")
    path = tmp_path / "collection"
    if target in {"result", "identity"}:
        result = json.loads((path / "result.json").read_text())
        result["requests_sha256" if target == "result" else "request_mapping_sha256"] = "0" * 64
        (path / "result.json").write_text(json.dumps(result))
    else:
        raw = (path / "requests.jsonl").read_bytes()
        if target in {"request", "offered", "unobserved", "duplicate"}:
            rows = [json.loads(line) for line in raw.splitlines()]
            if target == "request":
                rows[0]["request"]["messages"] = [{"role": "user", "content": "changed"}]
            elif target == "offered":
                rows[0]["response"]["offered"] = False
            elif target == "unobserved":
                rows[0]["response"]["output"] = "invented"
            raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
            if target == "duplicate":
                raw = raw.replace(b'{"request":', b'{"request":{},"request":', 1)
        elif target == "extra":
            raw += raw.splitlines(keepends=True)[0]
        else:
            raw = raw[:-2]
        (path / "requests.jsonl").write_bytes(raw)
        result = json.loads((path / "result.json").read_text())
        result["requests_sha256"] = hashlib.sha256(raw).hexdigest()
        (path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        load_quality_collection(path)


async def test_cancelled_collection_retains_partial_record(tmp_path: Path) -> None:
    """Cancellation leaves a durable failed row and an interrupted result, never approval."""
    arrived = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        """Hold after one content chunk until the owning request is cancelled."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """The collector must retain content observed before cancellation."""
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            arrived.set()
            await asyncio.Event().wait()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))
    ) as client:
        task = asyncio.create_task(
            collect_quality(client, specification(), tmp_path / "collection")
        )
        await asyncio.wait_for(arrived.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    raw = json.loads((tmp_path / "collection/requests.jsonl").read_text())
    assert raw["response"]["output"] == "partial"
    assert raw["response"]["error"] == "CancelledError"
    assert json.loads((tmp_path / "collection/result.json").read_text())["recorded"] == 1
    with pytest.raises(ValueError, match="completion"):
        load_quality_collection(tmp_path / "collection")


@pytest.mark.parametrize("field", ["model", "image_digest"])
def test_frozen_collection_rejects_runtime_mismatch(field: str) -> None:
    """Profile/configuration drift fails before filesystem output or HTTP work."""
    value = specification().model_dump()
    value["configuration"][field] = "changed"
    with pytest.raises(ValueError, match="differs"):
        QualityCollectionSpec.model_validate(value)


@pytest.mark.parametrize(
    "change",
    [
        {"successful": 1},
        {"outputs": {"extra": "value"}},
        {"status": "completed"},
    ],
)
def test_result_cannot_invent_completion(change: dict[str, object]) -> None:
    """Declared completion/counts must agree even before loading their raw evidence."""
    value: dict[str, object] = dict(
        specification_sha256="a" * 64,
        suite_hash="a" * 64,
        request_mapping_sha256="a" * 64,
        requests_sha256="a" * 64,
        status="interrupted",
        planned=3,
        recorded=0,
        successful=0,
        outputs={},
    )
    value.update(change)
    with pytest.raises(ValueError):
        QualityCollectionResult.model_validate(value)


async def test_collection_byte_budget_preserves_crossing_record(tmp_path: Path) -> None:
    """Oversized evidence aborts only after its last observed response is recorded for audit."""
    spec = specification().model_copy(update={"maximum_raw_bytes": 1024})
    content = "data: " + json.dumps({"choices": [{"delta": {"content": "x" * 2048}}]})
    content += '\n\ndata: {"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    ) as client:
        with pytest.raises(ValueError, match="byte budget"):
            await collect_quality(client, spec, tmp_path / "collection")
    raw = (tmp_path / "collection/requests.jsonl").read_text().splitlines()
    assert len(raw) == 1 and json.loads(raw[0])["response"]["output"] == "x" * 2048
    assert json.loads((tmp_path / "collection/status.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="byte budget"):
        load_quality_collection(tmp_path / "collection")
    with pytest.raises(ValueError, match="byte budget"):
        bounded_file(tmp_path / "collection/requests.jsonl", 10)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as client:
        with pytest.raises(ValueError, match="byte budget"):
            await collect_quality(client, spec, tmp_path / "small-result")
    with pytest.raises(ValueError, match="raw evidence"):
        load_quality_collection(tmp_path / "small-result")


def test_collection_suite_budget() -> None:
    """A huge golden input is rejected before any output directory or network request exists."""
    value = specification().model_dump()
    value["suite"]["cases"][0]["prompt"] = "a" * 1024**2
    with pytest.raises(ValueError, match="budget"):
        QualityCollectionSpec.model_validate(value)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 3601])
def test_collection_timeout_is_finite_and_bounded(timeout: float) -> None:
    """The overall producer deadline cannot be disabled with invalid numeric input."""
    value = specification().model_dump()
    value["timeout_seconds"] = timeout
    with pytest.raises(ValueError):
        QualityCollectionSpec.model_validate(value)


def test_default_collection_canonical_roundtrip() -> None:
    """Nested unvalidated numeric defaults cannot change an immutable digest after JSON reload."""
    original = specification()
    loaded = QualityCollectionSpec.model_validate_json(original.model_dump_json())
    assert original.canonical() == loaded.canonical()
    assert original.digest() == loaded.digest()


async def test_collection_overall_deadline_preserves_partial_row(tmp_path: Path) -> None:
    """A whole-stage timeout persists its active response and records failure, not cancellation."""

    class Stream(httpx.AsyncByteStream):
        """Keep a request open beyond the frozen collection deadline."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """Emit real content before the overall deadline interrupts this request."""
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            await asyncio.Event().wait()

    spec = specification().model_copy(update={"timeout_seconds": 0.05})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))
    ) as client:
        with pytest.raises(TimeoutError):
            await collect_quality(client, spec, tmp_path / "deadline")
    result = json.loads((tmp_path / "deadline/result.json").read_text())
    row = json.loads((tmp_path / "deadline/requests.jsonl").read_text())
    assert result["status"] == "failed" and result["recorded"] == 1
    assert row["response"]["output"] == "partial"
    assert not row["response"]["success"]


async def test_deadline_during_terminal_disk_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native I/O can outlive a deadline, but cannot retain completed evidence afterward."""
    entered, release = threading.Event(), threading.Event()

    def blocked(path: Path, value: object) -> None:
        """Block only the first terminal result write while the event loop observes its deadline."""
        if path.name == "result.json" and not entered.is_set():
            entered.set()
            assert release.wait(3)
        write_json(path, value)

    monkeypatch.setattr("finserve.registry.quality_collection.write_json", blocked)
    spec = specification().model_copy(update={"timeout_seconds": 0.2})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as client:
        task = asyncio.create_task(collect_quality(client, spec, tmp_path / "slow-disk"))
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.sleep(0.25)
        assert not task.done()
        release.set()
        with pytest.raises(TimeoutError):
            await task
    result = json.loads((tmp_path / "slow-disk/result.json").read_text())
    assert result["status"] == "failed" and result["recorded"] == 3
    with pytest.raises(ValueError, match="completion"):
        load_quality_collection(tmp_path / "slow-disk")
