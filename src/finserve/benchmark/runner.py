"""Bounded HTTP SSE load generator; raw evidence is written outside the source repository."""

import argparse
import asyncio
import json
import platform
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from finserve.benchmark.metrics import RequestRecord, summarize
from finserve.benchmark.workload import WorkItem, Workload, default_workload


class RunConfig(BaseModel):
    """Freeze arrival policy and deadlines before collecting any measured response."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    requests: int = Field(default=100, ge=1, le=1_000_000)
    concurrency: int = Field(default=4, ge=1, le=1024)
    warmup: int = Field(default=4, ge=0, le=10000)
    mode: Literal["closed", "open"] = "closed"
    rate: float = Field(default=10.0, gt=0, le=1_000_000)
    timeout_s: float = Field(default=30, gt=0, le=3600)
    model: str = Field(default="reference", min_length=1)
    hardware: str = Field(default="undeclared", min_length=1)
    revision: str = Field(default="undeclared", min_length=1)
    model_revision: str = Field(default="undeclared", min_length=1)
    tokenizer_revision: str = Field(default="undeclared", min_length=1)
    engine: str = Field(default="undeclared", min_length=1)
    engine_config: str = Field(default="undeclared", min_length=1)
    cache_policy: str = "engine-default; warmup may populate caches"


class Choice(BaseModel):
    """Only nonempty content establishes client first-content timing."""

    text: str = ""
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class Usage(BaseModel):
    """The server's tokenizer count is authoritative; SSE event count is never a token count."""

    completion_tokens: StrictInt = Field(ge=0)


class ServerTiming(BaseModel):
    """Only an explicit server duration can support server-received TTFT semantics."""

    model_config = ConfigDict(allow_inf_nan=False)
    server_ttft_seconds: float | None = Field(default=None, ge=0)


class StreamEvent(BaseModel):
    """Accept protocol extensions while validating the fields used for measurement."""

    choices: list[Choice] = Field(default_factory=lambda: list[Choice]())
    usage: Usage | None = None
    error: object | None = None
    finserve: ServerTiming | None = None


class StreamState(BaseModel):
    """Accumulate partial evidence even when the stream later times out or fails."""

    output: str = ""
    first_content_s: float | None = None
    generated_tokens: int | None = None
    done: bool = False
    status_code: int | None = None
    server_ttft_s: float | None = None
    finished: bool = False
    maximum_tokens: int = 4096

    def consume(self, payload: str) -> None:
        """Parse complete SSE events; reject structured errors without dropping prior timing."""
        if payload.strip() == "[DONE]":
            if self.first_content_s is None and not (self.generated_tokens == 0 and self.finished):
                raise ValueError("empty_stream_without_valid_finish")
            self.done = True
            return
        event = StreamEvent.model_validate_json(payload)
        if event.error is not None:
            raise ValueError("server_stream_error")
        content = "".join(choice.text for choice in event.choices)
        self.finished = self.finished or any(
            choice.finish_reason is not None for choice in event.choices
        )
        if content and self.first_content_s is None:
            self.first_content_s = time.perf_counter()
        self.output += content
        if len(self.output) > 4_000_000:
            raise ValueError("output_limit_exceeded")
        if event.usage is not None:
            if event.usage.completion_tokens > self.maximum_tokens:
                raise ValueError("usage_exceeds_request_budget")
            if (
                self.generated_tokens is not None
                and event.usage.completion_tokens < self.generated_tokens
            ):
                raise ValueError("usage_decreased")
            self.generated_tokens = event.usage.completion_tokens
        if event.finserve is not None:
            self.server_ttft_s = event.finserve.server_ttft_seconds


async def bounded_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Cap incomplete lines before newline parsing, including hostile streams with no newline."""
    pending = b""
    async for chunk in response.aiter_bytes():
        if len(chunk) > 1_000_000 or len(pending) + len(chunk) > 1_000_000:
            raise ValueError("stream_buffer_limit_exceeded")
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.rstrip(b"\r").decode("utf-8")
    if pending:
        yield pending.decode("utf-8")


async def consume_stream(response: httpx.Response, state: StreamState) -> None:
    """Respect SSE event boundaries and close incomplete/oversized streams as failed records."""
    data: list[str] = []
    size = 0
    async for line in bounded_lines(response):
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
            size += len(line)
            if size > 1_000_000:
                raise ValueError("event_limit_exceeded")
        elif not line and data:
            state.consume("\n".join(data))
            data, size = [], 0
            if state.done:
                break
    if not state.done:
        raise ValueError("stream_missing_done")


async def request_one(
    client: httpx.AsyncClient,
    url: str,
    item: WorkItem,
    logical_id: int,
    scheduled_s: float,
    config: RunConfig,
    phase: str,
) -> RequestRecord:
    """The total deadline includes headers and body; HTTPX read timeouts alone are per chunk."""
    sent = time.perf_counter()
    state = StreamState(maximum_tokens=item.max_tokens)
    error: str | None = None
    try:
        async with asyncio.timeout(config.timeout_s):
            async with client.stream(
                "POST",
                url,
                json={
                    "model": config.model,
                    "prompt": item.prompt,
                    "max_tokens": item.max_tokens,
                    "stream": True,
                    "temperature": 0,
                },
                timeout=config.timeout_s,
            ) as response:
                state.status_code = response.status_code
                response.raise_for_status()
                await consume_stream(response, state)
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        # Keep bounded failure labels; server bodies may include private prompt content.
        error = type(exc).__name__
    except asyncio.CancelledError:
        # The worker persists this partial record before propagating cancellation.
        error = "CancelledError"
    return RequestRecord(
        logical_id=logical_id,
        case_id=item.case_id,
        family=item.family,
        phase=phase,
        scheduled_s=scheduled_s,
        send_s=sent,
        first_content_s=state.first_content_s,
        complete_s=time.perf_counter(),
        success=error is None and state.done,
        status_code=state.status_code,
        error=error,
        output=state.output,
        generated_tokens=state.generated_tokens,
        server_ttft_s=state.server_ttft_s,
    )


async def run_phase(
    client: httpx.AsyncClient,
    url: str,
    workload: Workload,
    config: RunConfig,
    count: int,
    phase: str,
    record: Callable[[RequestRecord], None],
) -> float:
    """Use fixed workers and a bounded queue; open-loop overflow remains in offered accounting."""
    queue: asyncio.Queue[tuple[int, float] | None] = asyncio.Queue(config.concurrency)
    start = time.perf_counter()
    recorded: set[int] = set()
    arrivals: dict[int, float] = {}

    def persist(row: RequestRecord) -> None:
        """Track exactly-once persistence so interruption can account for unsent work."""
        record(row)
        recorded.add(row.logical_id)

    async def worker() -> None:
        """Each worker owns at most one network request, bounding sockets and task count."""
        while True:
            job = await queue.get()
            try:
                if job is None:
                    return
                index, scheduled = job
                item = workload.items[index % len(workload.items)]
                row = await request_one(client, url, item, index, scheduled, config, phase)
                persist(row)
                if row.error == "CancelledError":
                    raise asyncio.CancelledError
            finally:
                queue.task_done()

    try:
        async with asyncio.TaskGroup() as tasks:
            for _ in range(config.concurrency):
                tasks.create_task(worker())
            await enqueue_phase(queue, workload, config, count, phase, start, persist, arrivals)
            await queue.join()
            for _ in range(config.concurrency):
                await queue.put(None)
    except BaseException:
        for index in range(count):
            if index not in recorded:
                item = workload.items[index % len(workload.items)]
                now = time.perf_counter()
                persist(
                    RequestRecord(
                        logical_id=index,
                        case_id=item.case_id,
                        family=item.family,
                        phase=phase,
                        scheduled_s=arrivals.get(index, now),
                        complete_s=now,
                        success=False,
                        offered=index in arrivals,
                        intended_arrival_s=start + index / config.rate
                        if config.mode == "open"
                        else None,
                        error="run_interrupted_before_send",
                    )
                )
        raise
    return time.perf_counter() - start


async def enqueue_phase(
    queue: asyncio.Queue[tuple[int, float] | None],
    workload: Workload,
    config: RunConfig,
    count: int,
    phase: str,
    start: float,
    record: Callable[[RequestRecord], None],
    arrivals: dict[int, float],
) -> None:
    """A deterministic open-loop schedule sheds locally rather than spawning unbounded tasks."""
    for index in range(count):
        scheduled = start + index / config.rate if config.mode == "open" else time.perf_counter()
        if config.mode == "closed":
            arrivals[index] = scheduled
            await queue.put((index, scheduled))
            continue
        await asyncio.sleep(max(0, scheduled - time.perf_counter()))
        arrivals[index] = scheduled
        try:
            queue.put_nowait((index, scheduled))
        except asyncio.QueueFull:
            item = workload.items[index % len(workload.items)]
            record(
                RequestRecord(
                    logical_id=index,
                    case_id=item.case_id,
                    family=item.family,
                    phase=phase,
                    scheduled_s=scheduled,
                    complete_s=time.perf_counter(),
                    success=False,
                    error="client_queue_full",
                )
            )


def prepare_output(output: Path) -> Path:
    """Resolve before mutation; repository-local paths and existing evidence are rejected."""
    resolved = output.resolve()
    repository = Path(__file__).resolve().parents[3]
    if resolved == repository or repository in resolved.parents:
        raise ValueError("benchmark evidence must be outside the source repository")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def write_json(path: Path, value: object) -> None:
    """Forbid nonfinite JSON evidence rather than serializing Python's permissive NaN extension."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def build_summary(
    records: list[RequestRecord], duration: float, workload: Workload
) -> dict[str, object]:
    """All slices use the same run window; finance labels never alter engine routing."""
    summary = summarize(records, duration)
    summary["workload_hash"] = workload.digest()
    summary["slices"] = {
        family: summarize([row for row in records if row.family == family], duration)
        for family in sorted({item.family for item in workload.items})
    }
    finance = {"SEC_QA", "EARNINGS_SUMMARY", "FINANCIAL_TABLE_EXTRACTION", "CHART_REASONING"}
    summary["finance_aggregate"] = summarize(
        [row for row in records if row.family in finance], duration
    )
    summary["nonfinancial_aggregate"] = summarize(
        [row for row in records if row.family not in finance], duration
    )
    return summary


async def run_benchmark(
    client: httpx.AsyncClient,
    url: str,
    workload: Workload,
    config: RunConfig,
    output: Path,
) -> dict[str, object]:
    """Freeze manifest first, flush every raw record, and refuse overwrites or in-repo evidence."""
    output = prepare_output(output)
    manifest: dict[str, object] = {
        "run_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "workload_hash": workload.digest(),
        "workload": workload.model_dump(),
        "configuration": config.model_dump(),
        "url": str(httpx.URL(url).copy_with(username="", password="", query=None)),
        "client_platform": platform.platform(),
        "python": platform.python_version(),
        "timing_clock": "client perf_counter; optional server duration from finserve extension",
        "evidence_scope": "client HTTP observations; no inferred GPU metrics or cost",
    }
    write_json(output / "manifest.json", manifest)
    records: list[RequestRecord] = []
    with (output / "requests.jsonl").open("x", encoding="utf-8") as raw:

        def record(row: RequestRecord) -> None:
            """Flush observations immediately so interrupted runs retain completed evidence."""
            raw.write(row.model_dump_json() + "\n")
            raw.flush()
            # Text is already durable in JSONL; avoid retaining every generated output in RAM.
            records.append(row.model_copy(update={"output": ""}))

        try:
            await run_phase(client, url, workload, config, config.warmup, "warmup", record)
            measured_start = time.perf_counter()
            await run_phase(client, url, workload, config, config.requests, "measured", record)
            measured_end = time.perf_counter()
            duration = measured_end - measured_start
        except BaseException:
            manifest["status"] = "interrupted"
            write_json(output / "manifest.json", manifest)
            raise
    summary = build_summary(records, duration, workload)
    write_json(output / "summary.json", summary)
    manifest["status"] = "completed"
    manifest["measured_seconds"] = duration
    manifest["measured_started_s"] = measured_start
    manifest["measured_finished_s"] = measured_end
    write_json(output / "manifest.json", manifest)
    return summary


class ComparisonInput(BaseModel):
    """Load only comparable fields from immutable run artifacts."""

    model_config = ConfigDict(allow_inf_nan=False)
    workload_hash: str
    workload: Workload
    configuration: RunConfig
    status: Literal["completed"]
    measured_seconds: float = Field(gt=0)
    measured_started_s: float = Field(ge=0)
    measured_finished_s: float = Field(ge=0)

    @model_validator(mode="after")
    def declared(self) -> "ComparisonInput":
        """Hardware/model provenance must be explicit for a headline comparison."""
        for field in (
            "hardware",
            "revision",
            "model_revision",
            "tokenizer_revision",
            "engine",
            "engine_config",
        ):
            if getattr(self.configuration, field) == "undeclared":
                raise ValueError(f"declare {field} before comparison")
        if self.workload.digest() != self.workload_hash:
            raise ValueError("embedded workload hash mismatch")
        if self.measured_finished_s - self.measured_started_s != self.measured_seconds:
            raise ValueError("measurement duration disagrees with clock boundaries")
        return self


def validate_evidence(path: Path) -> ComparisonInput:
    """Recompute aggregates from complete raw artifacts before accepting a comparison input."""
    manifest = ComparisonInput.model_validate_json((path / "manifest.json").read_text())
    records = [
        RequestRecord.model_validate_json(line)
        for line in (path / "requests.jsonl").read_text().splitlines()
    ]
    for phase, count in (
        ("measured", manifest.configuration.requests),
        ("warmup", manifest.configuration.warmup),
    ):
        rows = [row for row in records if row.phase == phase]
        if len(rows) != count or {row.logical_id for row in rows} != set(range(count)):
            raise ValueError(f"incomplete or duplicated {phase} records")
    if len(records) != manifest.configuration.requests + manifest.configuration.warmup:
        raise ValueError("unexpected raw record population")
    for row in records:
        if not row.offered:
            raise ValueError("completed run contains requests never offered")
        item = manifest.workload.items[row.logical_id % len(manifest.workload.items)]
        if row.case_id != item.case_id or row.family != item.family:
            raise ValueError("raw record disagrees with frozen workload")
        if row.generated_tokens is not None and row.generated_tokens > item.max_tokens:
            raise ValueError("raw token count exceeds frozen request budget")
        if row.phase == "measured" and not (
            manifest.measured_started_s
            <= row.scheduled_s
            <= row.complete_s
            <= manifest.measured_finished_s
        ):
            raise ValueError("raw timing outside measurement window")
    expected = build_summary(records, manifest.measured_seconds, manifest.workload)
    actual: object = json.loads((path / "summary.json").read_text())
    if actual != expected:
        raise ValueError("summary disagrees with raw evidence")
    return manifest


def validate_comparison(baseline: Path, candidate: Path) -> None:
    """Reject changed workloads or load envelopes before anyone computes improvement ratios."""
    left = validate_evidence(baseline)
    right = validate_evidence(candidate)
    if left.workload_hash != right.workload_hash:
        raise ValueError("workload mismatch")
    for key in (
        "requests",
        "warmup",
        "concurrency",
        "mode",
        "rate",
        "timeout_s",
        "hardware",
        "model",
        "model_revision",
        "tokenizer_revision",
    ):
        if getattr(left.configuration, key) != getattr(right.configuration, key):
            raise ValueError(f"comparison differs in {key}")


def main() -> None:
    """Expose reproducible defaults and require an explicit external output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--mode", choices=("closed", "open"), default="closed")
    parser.add_argument("--rate", type=float, default=10)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--workload", type=Path, help="Frozen Workload JSON; supersedes default cases"
    )
    parser.add_argument("--model", default="reference")
    parser.add_argument("--hardware", default="undeclared")
    parser.add_argument("--revision", default="undeclared")
    parser.add_argument("--model-revision", default="undeclared")
    parser.add_argument("--tokenizer-revision", default="undeclared")
    parser.add_argument("--engine", default="undeclared")
    parser.add_argument("--engine-config", default="undeclared")
    args = parser.parse_args()
    workload = (
        Workload.model_validate_json(args.workload.read_text())
        if args.workload
        else default_workload(args.max_tokens)
    )
    config = RunConfig(
        requests=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        mode=args.mode,
        rate=args.rate,
        timeout_s=args.timeout,
        model=args.model,
        hardware=args.hardware,
        revision=args.revision,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        engine=args.engine,
        engine_config=args.engine_config,
    )

    async def execute() -> None:
        """One pooled client bounds connection reuse to the declared concurrency."""
        async with httpx.AsyncClient(
            limits=httpx.Limits(max_connections=config.concurrency)
        ) as client:
            result = await run_benchmark(client, args.url, workload, config, args.output)
            print(json.dumps(result, indent=2, allow_nan=False))

    asyncio.run(execute())


if __name__ == "__main__":
    main()
