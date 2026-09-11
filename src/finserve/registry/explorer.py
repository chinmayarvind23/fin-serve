"""Read-only, bounded GraphQL views over registered and checksum-verified serving evidence."""

import asyncio
import hmac
import json
import os
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from graphql import (
    GraphQLObjectType,
    GraphQLResolveInfo,
    build_schema,
    graphql_sync,  # type: ignore[reportUnknownVariableType]  # Upstream middleware type is incomplete.
    parse,
)
from graphql.language import (
    FieldNode,
    IntValueNode,
    OperationDefinitionNode,
    OperationType,
    SelectionSetNode,
    VariableNode,
)
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from finserve.gateway.admission import Admission
from finserve.gateway.body_limit import BodyLimit
from finserve.registry.annotations import Annotation, annotations
from finserve.registry.artifacts import ArtifactRef, ArtifactStore, LocalArtifactStore
from finserve.registry.metadata import Registry, RunBundle, decisions, events, jobs, runs
from finserve.reliability.promotion import PromotionDecision

BODY_SECONDS = 5.0
QUERY_SECONDS = 5.0
SEND_SECONDS = 5.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_METADATA_CHARS = 65536


class ExplorerSendTimeout(RuntimeError):
    """Abort a stalled response without attempting a second status after headers may have left."""


class ExplorerBoundary:
    """Authenticate and admit before receiving bodies, retaining ownership through native drain."""

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        """Keep body buffering inside the four-request boundary and outside GraphQL parsing."""
        self.app = BodyLimit(app, max_bytes=16384)
        self.unbuffered_app = app
        self.credential = ("Bearer " + api_key).encode()
        self.admission = Admission(4)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """A stalled or unauthenticated body cannot allocate an unbounded set of query owners."""
        if scope["type"] != "http" or scope["path"] != "/graphql":
            await self.unbuffered_app(scope, receive, send)
            return
        authorization = Headers(scope=scope).getlist("authorization")
        if len(authorization) != 1 or not hmac.compare_digest(
            authorization[0].encode(), self.credential
        ):
            await JSONResponse({"error": {"code": "UNAUTHORIZED"}}, 401)(scope, receive, send)
            return
        if not self.admission.acquire():
            await JSONResponse({"error": {"code": "QUERY_CAPACITY_EXHAUSTED"}}, 429)(
                scope, receive, send
            )
            return
        deadline = asyncio.get_running_loop().time() + BODY_SECONDS
        send_deadline: float | None = None

        async def bounded_receive() -> Message:
            """Apply one total body budget rather than resetting a timeout for each chunk."""
            async with asyncio.timeout_at(deadline):
                return await receive()

        async def bounded_send(message: Message) -> None:
            """One response deadline covers both headers and body for a stalled client."""
            nonlocal send_deadline
            if send_deadline is None:
                send_deadline = asyncio.get_running_loop().time() + SEND_SECONDS
            try:
                async with asyncio.timeout_at(send_deadline):
                    await send(message)
            except TimeoutError:
                raise ExplorerSendTimeout("evidence response send deadline expired") from None

        try:
            try:
                await self.app(scope, bounded_receive, bounded_send)
            except TimeoutError:
                await JSONResponse({"error": {"code": "BODY_TIMEOUT"}}, 408)(
                    scope, receive, bounded_send
                )
        finally:
            self.admission.release()


SCHEMA = """
type Configuration {
  engine: String!, engineConfig: String!, model: String!, modelRevision: String!,
  tokenizerRevision: String!, sourceRevision: String!, imageDigest: String!, hardware: String!,
  workloadHash: String!, concurrency: Int!, warmup: Int!, cachePolicy: String!
}
type Metrics {
  offered: Int!, succeeded: Int!, failed: Int!, seconds: Float!, requestsPerSecond: Float!,
  tokensPerSecond: Float!, generatedTokens: Int!, successRate: Float!,
  clientTtft: Float, serverTtft: Float, p50: Float, p95: Float, scheduledP95: Float
}
type RequestRecord {
  id: Int!, caseId: String!, family: String!, phase: String!, success: Boolean!,
  error: String, statusCode: Int, generatedTokens: Int, output: String!,
  ttft: Float, latency: Float, outputTruncated: Boolean!
}
type Artifact { kind: String!, sha256: String!, sizeBytes: Int! }
type Slice { name: String!, metrics: Metrics! }
type Gpu {
  averageUtilization: Float, coverage: Float!, deviceIds: [String!]!, method: String!,
  reportSha256: String!, inputSha256: [String!]!
}
type Quality {
  accuracy: Float!, parity: Float!, caseCount: Int!, passed: Boolean!, scope: String!,
  referenceScope: String!, hardFailures: [String!]!, reportSha256: String!, inputSha256: [String!]!
}
type Run {
  id: ID!, configuration: Configuration!, metrics: Metrics!, slices: [Slice!]!,
  gpu: Gpu, quality: Quality,
  artifacts: [Artifact!]!, requests(first: Int = 20, offset: Int = 0): [RequestRecord!]!
}
type Decision { id: ID!, runId: ID!, approved: Boolean!, reasons: [String!]! }
type Lifecycle { id: ID!, status: String!, version: Int!, applyAttempts: Int!, decisionId: String }
type LifecycleEvent { id: ID!, jobId: ID!, observedAt: Float!, status: String!, version: Int! }
type Query {
  runs(first: Int = 20, after: ID): [Run!]!, run(id: ID!): Run,
  decisions(first: Int = 20): [Decision!]!,
  lifecycle(first: Int = 20): [Lifecycle!]!, events(first: Int = 20): [LifecycleEvent!]!
}
"""


class QueryInput(BaseModel):
    """A small operation budget applies before GraphQL parsing or SQL access."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8192)
    variables: dict[str, Any] = Field(default_factory=dict, max_length=16)
    operationName: str | None = Field(default=None, max_length=128)


def bounded_page(first: int, offset: int = 0) -> None:
    """Both SQL lists and request slices have the same explicit small page bounds."""
    if not 1 <= first <= 50 or not 0 <= offset <= 10000:
        raise ValueError("page outside supported bounds")


def bounded_query(query: str, variables: dict[str, Any] | None = None) -> None:
    """Disallow fragments and mutations so depth and field budgets cannot hide recursive work."""
    document = parse(query, max_tokens=1000)
    if len(document.definitions) != 1:
        raise ValueError("one query operation required")
    operation = document.definitions[0]
    if (
        not isinstance(operation, OperationDefinitionNode)
        or operation.operation != OperationType.QUERY
    ):
        raise ValueError("read-only query required")
    count, cost = 0, 0

    def visit(selection: SelectionSetNode, depth: int, multiplier: int = 1) -> None:
        """Count aliases as independent work and reject introspection on this product endpoint."""
        nonlocal count, cost
        if depth > 6:
            raise ValueError("query depth exceeded")
        for node in selection.selections:
            count += 1
            cost += multiplier
            if count > 120 or not isinstance(node, FieldNode) or node.name.value.startswith("__"):
                raise ValueError("query field budget exceeded")
            if cost > 2000:
                raise ValueError("query expanded field budget exceeded")
            if node.selection_set is not None:
                fanout = 1
                if node.name.value in {"runs", "requests", "decisions", "lifecycle", "events"}:
                    fanout = 20
                    for argument in node.arguments:
                        if argument.name.value == "first":
                            if isinstance(argument.value, IntValueNode):
                                fanout = int(argument.value.value)
                            elif isinstance(argument.value, VariableNode):
                                fanout = (variables or {}).get(argument.value.name.value, 50)
                            else:
                                raise ValueError("invalid page argument")
                    if type(fanout) is not int:
                        raise ValueError("invalid page argument")
                    bounded_page(fanout)
                visit(node.selection_set, depth + 1, multiplier * fanout)

    visit(operation.selection_set, 1)


def bounded_metadata(value: object) -> str:
    """Reject clipped registry payloads; SQL substr caps allocation before Python receives rows."""
    text = str(value)
    if len(text) > MAX_METADATA_CHARS:
        raise ValueError("registry metadata exceeds query limit")
    return text


def metric_view(value: dict[str, Any]) -> dict[str, Any]:
    """Expose the published population and engine accounting without rounding the stored values."""
    return {
        "offered": value["offered_requests"],
        "succeeded": value["successful_requests"],
        "failed": value["failed_requests"],
        "seconds": value["measured_seconds"],
        "requestsPerSecond": value["requests_per_second"],
        "tokensPerSecond": value["tokens_per_second"],
        "generatedTokens": value["generated_tokens"],
        "successRate": value["success_rate"],
        "clientTtft": value.get("client_ttft_median_s"),
        "serverTtft": value.get("server_ttft_median_s"),
        "p50": value.get("e2e_p50_s"),
        "p95": value.get("e2e_p95_s"),
        "scheduledP95": value.get("scheduled_to_complete_p95_s"),
    }


class EvidenceReader:
    """Memoization and a verified byte budget bound artifact reads across aliased fields."""

    def __init__(self, registry: Registry, artifacts: ArtifactStore) -> None:
        """Resolvers receive configured storage only; query data cannot name a path or endpoint."""
        self.registry, self.artifacts = registry, artifacts
        self.remaining = 32 * 1024 * 1024
        self.cache: dict[str, bytes] = {}
        self.deadline = time.monotonic() + QUERY_SECONDS
        self.output_remaining = MAX_RESPONSE_BYTES - 16384

    def charge_field(self, value: Any, name: str | int) -> None:
        """Account for every resolved alias before GraphQL retains its serialized output."""
        self.check_deadline()
        cost = len(json.dumps(str(name)).encode()) + 2
        if isinstance(value, str | float | int | bool) or value is None:
            cost += len(json.dumps(value, allow_nan=False).encode())
        elif isinstance(value, list | tuple):
            items = cast(list[Any] | tuple[Any, ...], value)
            cost += 2 + len(items) * 2
            for item in items:
                if isinstance(item, str | float | int | bool) or item is None:
                    cost += len(json.dumps(item, allow_nan=False).encode())
        else:
            cost += 2
        self.output_remaining -= cost
        if self.output_remaining < 0:
            raise ValueError("query response budget exceeded")

    def check_deadline(self) -> None:
        """Stop new work after expiry; native local file I/O itself cannot be forcibly cancelled."""
        if time.monotonic() >= self.deadline:
            raise TimeoutError("evidence query deadline expired")

    def read(self, reference: ArtifactRef) -> bytes:
        """Verify all bytes through the store before interpreting a manifest or response record."""
        self.check_deadline()
        key = reference.model_dump_json()
        if key not in self.cache:
            if reference.size_bytes > self.remaining:
                raise ValueError("query artifact budget exceeded")
            data = self.artifacts.get(reference)
            self.remaining -= len(data)
            self.cache[key] = data
        return self.cache[key]

    def run_view(self, bundle: RunBundle) -> dict[str, Any]:
        """Whitelist public evidence fields; storage paths and service credentials stay private."""
        manifest = json.loads(self.read(bundle.manifest))
        summary = json.loads(self.read(bundle.summary))
        configuration = manifest["configuration"]
        slices = [
            {"name": name, "metrics": metric_view(value)}
            for name, value in summary.get("slices", {}).items()
        ]
        for name in ("finance_aggregate", "nonfinancial_aggregate"):
            if name in summary:
                slices.append({"name": name, "metrics": metric_view(summary[name])})
        return {
            "id": bundle.run_id,
            "_bundle": bundle,
            "configuration": {
                "engine": configuration["engine"],
                "engineConfig": configuration["engine_config"],
                "model": configuration["model"],
                "modelRevision": configuration["model_revision"],
                "tokenizerRevision": configuration["tokenizer_revision"],
                "sourceRevision": configuration["revision"],
                "imageDigest": configuration["image_digest"],
                "hardware": configuration["hardware"],
                "workloadHash": bundle.workload_hash,
                "concurrency": configuration["concurrency"],
                "warmup": configuration["warmup"],
                "cachePolicy": configuration["cache_policy"],
            },
            "metrics": metric_view(summary),
            "slices": slices,
            "artifacts": [
                {"kind": kind, "sha256": ref.sha256, "sizeBytes": ref.size_bytes}
                for kind, ref in (
                    ("manifest", bundle.manifest),
                    ("requests", bundle.requests),
                    ("summary", bundle.summary),
                )
            ],
        }

    def list_runs(self, first: int = 20, after: str | None = None) -> list[dict[str, Any]]:
        """Stable ID cursors bound database work without accepting query-language SQL fragments."""
        bounded_page(first)
        self.check_deadline()
        statement = (
            select(func.substr(runs.c.payload, 1, MAX_METADATA_CHARS + 1))
            .order_by(runs.c.id)
            .limit(first)
        )
        if after is not None:
            if len(after) > 128:
                raise ValueError("invalid cursor")
            statement = statement.where(runs.c.id > after)
        with self.registry.engine.connect() as connection:
            payloads = connection.execute(statement).scalars().all()
        return [
            self.run_view(RunBundle.model_validate_json(bounded_metadata(value)))
            for value in payloads
        ]

    def one_run(self, identity: str) -> dict[str, Any] | None:
        """Missing IDs return null; they never trigger path resolution or fabricate empty runs."""
        if not 1 <= len(identity) <= 128:
            raise ValueError("invalid run identity")
        self.check_deadline()
        with self.registry.engine.connect() as connection:
            payload = connection.execute(
                select(func.substr(runs.c.payload, 1, MAX_METADATA_CHARS + 1)).where(
                    runs.c.id == identity
                )
            ).scalar_one_or_none()
        if payload is None:
            return None
        bundle = RunBundle.model_validate_json(bounded_metadata(payload))
        return self.run_view(bundle)

    def requests(self, source: dict[str, Any], first: int, offset: int) -> list[dict[str, Any]]:
        """Page failures as well as successes and mark output truncation explicitly."""
        bounded_page(first, offset)
        bundle = cast(RunBundle, source["_bundle"])
        records = self.read(bundle.requests).splitlines()
        result: list[dict[str, Any]] = []
        for line in records[offset : offset + first]:
            record = json.loads(line)
            sent, content, complete = (
                record.get("send_s"),
                record.get("first_content_s"),
                record.get("complete_s"),
            )
            output = record.get("output") or ""
            result.append(
                {
                    "id": record["logical_id"],
                    "caseId": record["case_id"],
                    "family": record["family"],
                    "phase": record["phase"],
                    "success": record["success"],
                    "error": record.get("error"),
                    "statusCode": record.get("status_code"),
                    "generatedTokens": record["generated_tokens"],
                    "output": output[:4096],
                    "outputTruncated": len(output) > 4096,
                    "ttft": content - sent if content is not None and sent is not None else None,
                    "latency": complete - sent
                    if complete is not None and sent is not None
                    else None,
                }
            )
        return result

    def control(self, kind: str, first: int) -> list[dict[str, Any]]:
        """Read bounded history without exporting mutable specifications or arbitrary errors."""
        bounded_page(first)
        self.check_deadline()
        table = {"decisions": decisions, "lifecycle": jobs, "events": events}[kind]
        with self.registry.engine.connect() as connection:
            identifiers = {
                "decisions": [table.c.candidate_run_id] if kind == "decisions" else [],
                "lifecycle": [],
                "events": [table.c.job_id, table.c.observed_at] if kind == "events" else [],
            }
            rows = (
                connection.execute(
                    select(
                        table.c.id,
                        *identifiers[kind],
                        func.substr(table.c.payload, 1, MAX_METADATA_CHARS + 1).label("payload"),
                    )
                    .order_by(table.c.id.desc())
                    .limit(first)
                )
                .mappings()
                .all()
            )
        output: list[dict[str, Any]] = []
        for row in rows:
            value = json.loads(bounded_metadata(row["payload"]))
            if kind == "decisions":
                decision = PromotionDecision.model_validate(value)
                output.append(
                    {
                        "id": row["id"],
                        "runId": row["candidate_run_id"],
                        "approved": decision.approved,
                        "reasons": decision.rejection_reasons,
                    }
                )
            elif kind == "lifecycle":
                output.append(
                    {
                        "id": row["id"],
                        "status": value["status"],
                        "version": value["version"],
                        "applyAttempts": value["apply_attempts"],
                        "decisionId": value["decision_digest"],
                    }
                )
            else:
                output.append(
                    {
                        "id": str(row["id"]),
                        "jobId": row["job_id"],
                        "observedAt": row["observed_at"],
                        "status": value["status"],
                        "version": value["version"],
                    }
                )
        return output

    def annotation(self, source: dict[str, Any], kind: str) -> dict[str, Any] | None:
        """Read a nullable registered report through the same verified per-query artifact budget."""
        self.check_deadline()
        bundle = cast(RunBundle, source["_bundle"])
        with self.registry.engine.connect() as connection:
            payload = connection.execute(
                select(func.substr(annotations.c.payload, 1, MAX_METADATA_CHARS + 1)).where(
                    annotations.c.id == f"{bundle.run_id}:{kind}"
                )
            ).scalar_one_or_none()
        if payload is None:
            return None
        reference = Annotation.model_validate_json(bounded_metadata(payload))
        if reference.run_id != bundle.run_id or reference.kind != kind:
            raise ValueError("annotation identity mismatch")
        report = json.loads(self.read(reference.report))
        fields = {
            "averageUtilization": "average_gpu_utilization_percent",
            "coverage": "coverage",
            "deviceIds": "device_ids",
            "method": "method",
        }
        if kind == "quality":
            fields = {
                "accuracy": "candidate_accuracy",
                "parity": "parity",
                "caseCount": "case_count",
                "passed": "passed",
                "scope": "scope",
                "referenceScope": "reference_scope",
                "hardFailures": "hard_failures",
            }
        return {
            **{name: report[key] for name, key in fields.items()},
            "reportSha256": reference.report.sha256,
            "inputSha256": [item.sha256 for item in reference.inputs],
        }


def execute_query(
    registry: Registry, artifacts: ArtifactStore, payload: QueryInput
) -> dict[str, Any]:
    """A read context owns one operation; resolver failures return static errors without paths."""
    bounded_query(payload.query, payload.variables)
    reader = EvidenceReader(registry, artifacts)
    schema = build_schema(SCHEMA)
    query = schema.query_type
    assert query is not None

    def resolve_runs(
        _source: object, _info: GraphQLResolveInfo, first: int = 20, after: str | None = None
    ) -> list[dict[str, Any]]:
        """GraphQL supplies typed pagination to a bounded SQL query."""
        return reader.list_runs(first, after)

    def resolve_run(_source: object, _info: GraphQLResolveInfo, id: str) -> dict[str, Any] | None:
        """Resolve only a registered opaque run identifier."""
        return reader.one_run(id)

    def resolve_control(
        _source: object, info: GraphQLResolveInfo, first: int = 20
    ) -> list[dict[str, Any]]:
        """Use the schema-owned field identity, never a caller SQL table selector."""
        return reader.control(info.field_name, first)

    query.fields["runs"].resolve = resolve_runs
    query.fields["run"].resolve = resolve_run
    for name in ("decisions", "lifecycle", "events"):
        query.fields[name].resolve = resolve_control
    run_type = schema.get_type("Run")
    assert isinstance(run_type, GraphQLObjectType)

    def request_resolver(
        source: dict[str, Any], _info: GraphQLResolveInfo, first: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Keep request pagination explicit instead of exposing all raw model output by default."""
        return reader.requests(source, first, offset)

    run_type.fields["requests"].resolve = request_resolver

    def annotation_resolver(
        source: dict[str, Any], info: GraphQLResolveInfo
    ) -> dict[str, Any] | None:
        """The schema field fixes the annotation kind; callers cannot select a storage location."""
        return reader.annotation(source, info.field_name)

    for kind in ("gpu", "quality"):
        run_type.fields[kind].resolve = annotation_resolver

    def budget_resolver(
        next_resolver: Callable[..., Any], source: object, info: GraphQLResolveInfo, **kwargs: Any
    ) -> Any:
        """Charge scalar aliases and repeated lists during resolution, before full serialization."""
        reader.check_deadline()
        value = next_resolver(source, info, **kwargs)
        reader.charge_field(value, info.path.key)
        return value

    result = graphql_sync(
        schema,
        payload.query,
        variable_values=payload.variables,
        operation_name=payload.operationName,
        middleware=[budget_resolver],
    )
    if result.errors:
        return {"data": None, "errors": [{"message": "EVIDENCE_QUERY_FAILED"}]}
    response = {"data": result.data}
    if len(json.dumps(response).encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("query response budget exceeded")
    return response


def create_explorer_app(registry: Registry, artifacts: ArtifactStore, api_key: str) -> FastAPI:
    """Serve local SQLite/CAS only; four owners drain native work even past query timeout.

    SQLite lock waits are configured to five seconds by Registry. Network stores,
    PostgreSQL and arbitrary storage adapters need explicit transport timeouts before
    this service can support them. Native local file reads cannot be forcibly stopped;
    expired/disconnected HTTP owners retain admission until their worker thread ends.
    """
    if len(api_key) < 16:
        raise ValueError("an evidence service credential is required")
    if registry.engine.dialect.name != "sqlite" or type(artifacts) is not LocalArtifactStore:
        raise ValueError("explorer requires local SQLite and LocalArtifactStore")
    annotations.create(registry.engine, checkfirst=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        """This standalone service owns its database pool and closes it on shutdown."""
        try:
            yield
        finally:
            await asyncio.to_thread(registry.close)

    app = FastAPI(title="FinServe evidence", lifespan=lifespan)
    app.add_middleware(ExplorerBoundary, api_key=api_key)

    @app.post("/graphql", response_model=None)
    async def query(request: Request) -> Response:
        """Authenticate before parsing GraphQL; retain a slot until offloaded SQL work drains."""
        task: asyncio.Task[dict[str, Any]] | None = None
        try:
            payload = QueryInput.model_validate(await request.json())
            task = asyncio.create_task(
                asyncio.to_thread(execute_query, registry, artifacts, payload)
            )
            async with asyncio.timeout(QUERY_SECONDS):
                return JSONResponse(
                    await asyncio.shield(task), headers={"Cache-Control": "private, no-store"}
                )
        except TimeoutError:
            return JSONResponse({"error": {"code": "QUERY_TIMEOUT"}}, status_code=504)
        except Exception:
            return JSONResponse({"error": {"code": "INVALID_EVIDENCE_QUERY"}}, status_code=400)
        finally:
            if task is not None:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                # Consume late native failures without replacing timeout or cancellation semantics.
                if not task.cancelled():
                    task.exception()

    return app


def from_env() -> FastAPI:
    """Only trusted process configuration selects the registry and artifact namespaces."""
    database_url, artifact_root, key = (
        os.environ["FINSERVE_REGISTRY_URL"],
        Path(os.environ["FINSERVE_ARTIFACT_ROOT"]),
        os.environ["FINSERVE_API_KEY"],
    )
    registry = Registry(database_url)
    try:
        return create_explorer_app(registry, LocalArtifactStore(artifact_root), key)
    except BaseException:
        registry.close()
        raise
