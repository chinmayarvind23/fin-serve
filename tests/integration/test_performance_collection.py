"""Collector loops retain raw work and drain native operations under cancellation."""

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from finserve.benchmark.experiment import prepare_experiment
from finserve.benchmark.gpu import TelemetrySample
from finserve.benchmark.metrics import RequestRecord
from finserve.benchmark.runner import RunConfig, run_phase, validate_evidence
from finserve.benchmark.workload import WorkItem, Workload
from finserve.contracts.deployment import Revision
from finserve.contracts.performance import PerformanceCollectionSpec
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.http_ownership import HTTPClosureError
from finserve.registry.performance_collection import collect_performance


def specification() -> PerformanceCollectionSpec:
    """Explicit fixture provenance exercises canonical binding without claiming a real image."""
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
        requests=4,
        warmup=1,
        concurrency=2,
        hardware="fixture-cpu",
        model="fixture",
        revision=revision.source_revision,
        model_revision=revision.model_revision,
        tokenizer_revision=revision.tokenizer_revision,
        engine=revision.engine,
        engine_config=revision.engine_config,
        image_digest=revision.image_digest,
        config_digest=revision.config_digest,
    )
    return PerformanceCollectionSpec(
        collection_id="fixture",
        collector_revision="c" * 40,
        profile=profile,
        revision=revision,
        configuration=config,
        workload=Workload(
            suite_id="fixture",
            version=1,
            items=(WorkItem(case_id="one", prompt="hello", max_tokens=1),),
        ),
        timeout_seconds=5,
    )


def clean_fixture_git(arguments: list[str], **kwargs: object) -> str:
    """Tests declare synthetic clean collector provenance without relabeling actual experiments."""
    return "" if arguments[1] == "status" else "c" * 40 + "\n"


async def test_worker_retains_real_protocol_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use actual benchmark/SSE logic on the worker loop; missing GPU is recorded as missing."""
    requests: list[str] = []

    def client(_: RunConfig) -> httpx.AsyncClient:
        """The injected transport returns original synthetic content to the real HTTP parser."""

        def response(request: httpx.Request) -> httpx.Response:
            """Record each request and supply one authoritative generated token."""
            requests.append(str(request.url))
            return httpx.Response(
                200,
                content=(
                    'data: {"choices":[{"text":"Hi","finish_reason":"length"}],'
                    '"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
                ),
            )

        return httpx.AsyncClient(transport=httpx.MockTransport(response))

    monkeypatch.setattr("finserve.benchmark.experiment.benchmark_client", client)
    monkeypatch.setattr("finserve.benchmark.experiment.subprocess.check_output", clean_fixture_git)
    monkeypatch.setattr(
        "finserve.benchmark.experiment.collect",
        lambda: TelemetrySample(
            epoch_s=time.time(), collection_seconds=0, devices=[], error="FixtureNoGPU"
        ),
    )
    spec = specification().model_copy(update={"timeout_seconds": 15})
    output = tmp_path / "experiment"
    await collect_performance(spec, output)
    assert len(requests) == 5
    assert validate_evidence(output / "run").configuration == spec.configuration
    assert json.loads((output / "experiment-status.json").read_text())["status"] == "completed"
    assert (output / "gpu.jsonl").read_text()
    with pytest.raises(ValueError, match="already exists"):
        await collect_performance(spec, output)


@pytest.mark.parametrize("close_failure", [False, True])
async def test_cancellation_drains_worker_under_repeated_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close_failure: bool
) -> None:
    """The orchestration loop stays responsive while an owned native write delays worker exit."""
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()

    async def blocked(
        url: str,
        output: Path,
        workload: Workload,
        config: RunConfig,
        *,
        collector_revision: str | None = None,
    ) -> dict[str, object]:
        """Retain an active raw row and hold cleanup until the test releases ownership."""
        (output / "run").mkdir(parents=True)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
            assert release.wait(5)
            (output / "run" / "requests.jsonl").write_text(
                json.dumps({"error": "HTTPClosureError" if close_failure else "CancelledError"})
                + "\n"
            )
        raise AssertionError("unreachable")

    monkeypatch.setattr("finserve.registry.performance_collection.experiment", blocked)
    task = asyncio.create_task(collect_performance(specification(), tmp_path / "experiment"))
    assert await asyncio.to_thread(entered.wait, 3)
    task.cancel()
    assert await asyncio.to_thread(cancelled.wait, 3)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    release.set()
    with pytest.raises(HTTPClosureError if close_failure else asyncio.CancelledError):
        await task
    assert (tmp_path / "experiment" / "run" / "requests.jsonl").exists()


@pytest.mark.parametrize(
    "fault", ["deadline", "bytes", "late_native", "failed_close", "partial_close", "failure"]
)
async def test_worker_rejects_late_or_unresolved_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """A completed inner task cannot bypass observed deadline, byte budget or cleanup failure."""
    spec = specification().model_copy(update={"timeout_seconds": 0.1, "maximum_raw_bytes": 1024})

    def native_write() -> None:
        """Model a blocking native write that prevents the isolated loop's deadline callback."""
        time.sleep(0.15)

    async def faulty(
        url: str,
        output: Path,
        workload: Workload,
        config: RunConfig,
        *,
        collector_revision: str | None = None,
    ) -> dict[str, object]:
        """Inject one bounded ownership failure instead of performing a real GPU benchmark."""
        (output / "run").mkdir(parents=True)
        if fault == "deadline":
            await asyncio.Event().wait()
        elif fault == "bytes":
            (output / "gpu.jsonl").write_text("x" * 2048)
        elif fault == "late_native":
            native_write()
        elif fault == "failed_close":
            (output / "run" / "requests.jsonl").write_text('{"error":"HTTPClosureError"}\n')
        elif fault == "partial_close":
            (output / "run" / "requests.jsonl").write_text('{"error":')
            raise HTTPClosureError("fixture close and partial write failure")
        else:
            raise RuntimeError("fixture failure")
        return {}

    monkeypatch.setattr("finserve.registry.performance_collection.experiment", faulty)
    expected = (
        TimeoutError
        if fault in {"deadline", "late_native"}
        else ValueError
        if fault == "bytes"
        else HTTPClosureError
        if fault in {"failed_close", "partial_close"}
        else RuntimeError
    )
    with pytest.raises(expected):
        await collect_performance(spec, tmp_path / "experiment")


def test_performance_spec_is_canonical_and_bounded() -> None:
    """Nested float defaults and request mapping remain stable across serialized task handoffs."""
    spec = specification()
    assert (
        spec.digest()
        == PerformanceCollectionSpec.model_validate_json(spec.model_dump_json()).digest()
    )
    for key, value in (
        ("model", "wrong"),
        ("revision", "wrong"),
        ("concurrency", 129),
        ("requests", 65536),
    ):
        with pytest.raises(ValueError):
            PerformanceCollectionSpec.model_validate(
                {
                    **spec.model_dump(),
                    "configuration": {**spec.configuration.model_dump(), key: value},
                }
            )
    chat = PerformanceCollectionSpec.model_validate(
        {
            **spec.model_dump(),
            "configuration": {
                **spec.configuration.model_dump(),
                "request_api": "chat",
                "chat_template_sha256": "f" * 64,
            },
        }
    )
    assert chat.endpoint().endswith("/chat/completions")


@pytest.mark.parametrize("warmup", [0, 2])
@pytest.mark.parametrize("failed_evidence", [False, True])
async def test_failed_http_close_stops_new_offers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warmup: int, failed_evidence: bool
) -> None:
    """A real stream-close failure must stop both warmup and measured dispatch immediately."""
    calls = 0

    class FailedClose(httpx.AsyncByteStream):
        """Finish a valid response but explicitly fail to release its local transport."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """Content is visible before cleanup uncertainty is discovered."""
            yield (
                b'data: {"choices":[{"text":"Hi","finish_reason":"length"}],'
                b'"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
            )

        async def aclose(self) -> None:
            """No successful transport release is available for this fixture."""
            raise RuntimeError("fixture close failure")

    def response(request: httpx.Request) -> httpx.Response:
        """Count actual offered HTTP requests, including any unsafe successor."""
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=FailedClose())

    def client(_: RunConfig) -> httpx.AsyncClient:
        """Run the normal benchmark parser and ownership wrapper on the collector loop."""
        if failed_evidence:

            class FailedClientExit(httpx.AsyncClient):
                """An additional pool-close error must not erase stream ownership uncertainty."""

                async def __aexit__(self, *args: Any) -> None:
                    """Release the fixture pool before injecting its independent unwind failure."""
                    await super().__aexit__(*args)
                    raise OSError("fixture client close failed")

            return FailedClientExit(transport=httpx.MockTransport(response))
        return httpx.AsyncClient(transport=httpx.MockTransport(response))

    monkeypatch.setattr("finserve.benchmark.experiment.benchmark_client", client)
    monkeypatch.setattr("finserve.benchmark.experiment.subprocess.check_output", clean_fixture_git)
    if failed_evidence:
        from finserve.benchmark.runner import write_json

        original_open = Path.open

        class FailedRawExit:
            """Model a buffered file whose final flush fails while unwinding a failed request."""

            def __init__(self, stream: Any) -> None:
                """Retain the actual fixture file for deterministic local cleanup."""
                self.stream = stream

            def __enter__(self) -> Any:
                """Use the real stream throughout benchmark execution."""
                return self.stream.__enter__()

            def __exit__(self, *args: Any) -> None:
                """Close the fixture handle, then report the simulated buffered-write error."""
                self.stream.__exit__(*args)
                raise OSError("fixture raw close failed")

        def failed_raw_exit(path: Path, *args: Any, **kwargs: Any) -> Any:
            """Alter only creation of the raw recorder; replay reads still use real files."""
            stream = cast(Any, original_open(path, *args, **kwargs))
            if path.name == "requests.jsonl" and args and args[0] == "x":
                return FailedRawExit(stream)
            return stream

        monkeypatch.setattr(Path, "open", failed_raw_exit)

        async def failed_persistence_phase(
            client: httpx.AsyncClient,
            url: str,
            workload: Workload,
            config: RunConfig,
            count: int,
            phase: str,
            record: Callable[[RequestRecord], None],
        ) -> float:
            """Fail raw persistence while keeping real dispatch, SSE and cleanup behavior."""

            def failed_record(row: RequestRecord) -> None:
                """No closure marker reaches disk when this storage operation fails."""
                raise OSError("fixture disk full")

            return await run_phase(client, url, workload, config, count, phase, failed_record)

        def failed_status(path: Path, value: object) -> None:
            """Initial files succeed; interrupted manifest and experiment status writes fail."""
            if (
                isinstance(value, dict)
                and cast(dict[str, object], value).get("status") == "interrupted"
            ):
                raise OSError("fixture disk full")
            write_json(path, cast(object, value))

        monkeypatch.setattr("finserve.benchmark.runner.run_phase", failed_persistence_phase)
        monkeypatch.setattr("finserve.benchmark.runner.write_json", failed_status)
        monkeypatch.setattr("finserve.benchmark.experiment.write_json", failed_status)
    monkeypatch.setattr(
        "finserve.benchmark.experiment.collect",
        lambda: TelemetrySample(
            epoch_s=time.time(), collection_seconds=0, devices=[], error="FixtureNoGPU"
        ),
    )
    base = specification()
    spec = base.model_copy(
        update={
            "configuration": base.configuration.model_copy(
                update={"concurrency": 1, "warmup": warmup}
            )
        }
    )
    output = tmp_path / "experiment"
    with pytest.raises(HTTPClosureError):
        await collect_performance(spec, output)
    rows = [
        json.loads(line) for line in (output / "run" / "requests.jsonl").read_text().splitlines()
    ]
    assert calls == 1
    if failed_evidence:
        assert rows == []
        assert not (output / "experiment-status.json").exists()
        return
    assert len(rows) == (warmup or spec.configuration.requests)
    assert rows[0]["error"] == "HTTPClosureError"
    assert all(row["error"] == "run_interrupted_before_send" for row in rows[1:])
    assert json.loads((output / "run" / "manifest.json").read_text())["status"] == "interrupted"
    assert not (output / "run" / "summary.json").exists()


async def test_failed_close_survives_raw_persistence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither the first raw write nor interruption accounting can authorize a retry."""
    calls = 0

    async def failed_request(*args: object, **kwargs: object) -> RequestRecord:
        """Supply a known unresolved close before the independently failing evidence writer."""
        nonlocal calls
        calls += 1
        return RequestRecord(
            logical_id=0,
            case_id="one",
            family="general",
            phase="measured",
            scheduled_s=1,
            send_s=1,
            complete_s=2,
            success=False,
            error="HTTPClosureError",
        )

    def failed_write(row: RequestRecord) -> None:
        """A full disk must not downgrade transport uncertainty to a retryable I/O failure."""
        raise OSError("fixture disk full")

    monkeypatch.setattr("finserve.benchmark.runner.request_one", failed_request)
    spec = specification()
    config = spec.configuration.model_copy(update={"concurrency": 1})
    async with httpx.AsyncClient() as client:
        with pytest.raises(HTTPClosureError):
            await run_phase(
                client, spec.endpoint(), spec.workload, config, 4, "measured", failed_write
            )
    assert calls == 1


@pytest.mark.parametrize("dirty", [False, True])
def test_source_preflight_rejects_drift_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirty: bool
) -> None:
    """A dirty checkout or different collector commit cannot be used for canonical evidence."""

    def git(arguments: list[str], **kwargs: object) -> str:
        """Return explicit incompatible source observations before any HTTP/GPU work exists."""
        if arguments[1] == "status":
            return " M modified.py\n" if dirty else ""
        return ("c" if dirty else "d") * 40 + "\n"

    monkeypatch.setattr("finserve.benchmark.experiment.subprocess.check_output", git)
    spec = specification()
    with pytest.raises(ValueError, match="frozen clean commit"):
        prepare_experiment(
            tmp_path / "output", spec.workload, spec.configuration, spec.collector_revision
        )
    assert not (tmp_path / "output").exists()
