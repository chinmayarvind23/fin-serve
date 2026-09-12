# API contracts

The gateway exposes a bounded subset of the OpenAI text/chat protocol. It does not implement every OpenAI parameter. Unknown request fields are rejected by the corresponding typed contract.

| Route | Behavior |
| --- | --- |
| `GET /healthz` | Process liveness; does not prove engine inference readiness |
| `GET /metrics` | Fixed-cardinality Prometheus metrics; keep internal to the service network |
| `POST /v1/completions` | Bounded prompt, model, output count, temperature and deadline; SSE or collected JSON |
| `POST /v1/chat/completions` | System/user/assistant text messages preserved for the backend chat template |
| `POST /v1/vision/completions` | Optional authenticated single-PNG image plus text capability |
| `GET /v1/visual/models` | Optional durable-job worker revision and supported image shape |
| `POST /v1/visual/jobs` | Authenticated durable submission with an idempotency key |
| `GET /v1/visual/jobs/{job_id}` | Tenant-scoped job status |
| `DELETE /v1/visual/jobs/{job_id}` | Explicit fenced cancellation intent |
| `GET /v1/visual/jobs/{job_id}/artifact` | Verified PNG after successful durable completion |

Set `FINSERVE_API_KEY` to protect text inference; image capabilities require it. Current authentication uses a configured service credential, not a user/role management system. Public clients cannot deploy through these routes. Release operations use the separate operator CLI/Airflow path and generation-fenced deployment adapter.

Text bodies are limited to 128 KiB. Chat additionally bounds the duplicated internal prompt/message payload so it fits the Ray transport. The image route permits a larger envelope, but caps the decoded PNG at 1 MiB and 512 pixels per side. Its request fields include `prompt`, `image_png_base64`, `model`, `max_tokens`, `timeout_seconds` and `stream`; arbitrary image URLs and batches are excluded. See `contracts/inference.py`, `contracts/vision.py` and the gateway's generated `/docs` for precise fields.

Successful streaming returns escaped JSON SSE, authoritative engine usage and `[DONE]`. An error after visible output remains a failed stream and is never retried transparently. Nonstream responses consume the same owned iterator. Error codes are stable; status, retryability and request identity depend on the relevant route. A normal text overload is HTTP 429 with code `OVERLOADED`.

The [evidence explorer](../apps/web/README.md) runs as a separate authenticated POST `/graphql` service. Its read-only schema exposes registered runs, bounded request pages, quality/GPU annotations and lifecycle history. It rejects mutations, introspection, fragments and queries exceeding its depth, expansion or byte limits. There are no public REST benchmark/deployment CRUD endpoints.

The gRPC protocol serves the [durable visual worker](../src/finserve/multimodal/README.md). Ray's internal endpoint uses typed NDJSON between trusted services. Neither internal transport is the public SSE protocol.

## Explicit output constraints

Text and chat requests may include `output_constraint`, for example:

```json
{"kind":"json_object","fields":[{"name":"revenue","type":"number"},{"name":"audited","type":"boolean"}]}
```

Supported kinds are `integer`, `decimal`, `scientific`, `yes_no`, `lowercase_word` and `json_object`. Numbers use ordinary decimal syntax with at most 128 integral/fractional digits; the default omits redundant zeros. A decimal may instead specify `decimal_places` from 0 through 12. Scientific output requires explicit `decimal_places` (0–12) and `exponent_digits` (1–4); it uses a single mantissa digit and a lowercase `e` with an explicit exponent sign. These widths describe the requested format, not an expected value. Yes/no permits both lowercase answers. A word contains 1–64 lowercase ASCII letters.

JSON requires 1–16 unique ASCII identifier keys of at most 64 characters. Fields support integer, number, string, boolean, null, or an array with an explicit scalar `item_type`. Arrays contain at most 64 items and cannot contain nested arrays or objects. Strings are limited to 4,096 characters; integer properties require JSON integer tokens. All fields are required, extra fields fail, and JSON whitespace is permitted. Unused options are rejected; absent new options preserve previously frozen constraint identities.

`FINSERVE_ENGINE=vllm` translates these shapes to native `structured_outputs`. SGLang, generic OpenAI-compatible endpoints and fixture/reference engines do not support this FinServe extension. The gateway rejects a locally unsupported capability with HTTP 422. Ray preserves the typed contract; its external-engine configuration must explicitly set `vllm_structured_outputs=true`. A warm route selects capability from its registered engine revision. Unsupported remote engines fail the stream rather than discard the constraint.

The adapter retains at most 128 KiB of constrained output for a final syntax check. Invalid or incomplete output fails before terminal success; any already-streamed fragments remain visible and must be treated as failed output. No repair, trimming or expected-answer comparison occurs. A syntactically valid response can still be factually wrong, and a valid output ending with `finish_reason=length` still reports that limit.

Managed vLLM profiles can pin `structured_output_backend="xgrammar"`, which also explicitly permits JSON whitespace. The API accepts no arbitrary regex, recursive schema, constant, default or caller-provided answer list. Native integration follows the [vLLM 0.29 structured-output API](https://docs.vllm.ai/en/v0.29.0/features/structured_outputs/) and [backend configuration](https://docs.vllm.ai/en/v0.29.0/api/vllm/config/structured_outputs/). Transport, collector and CPU grammar checks pass. Actual constrained GPU runs are complete, including the selected 7B regression score of 55/56 and independent score of 42/48; neither establishes full quality qualification. Isolated first-use overhead remains unmeasured. [Reproducibility](reproducibility.md) describes the separate frozen prompt-to-constraint map used by collectors.


## Managed failed-start reconciliation

The trusted Python operator API is
`runtime_stages.abort_runtime_stage(journal, abort_stage_id, launch_stage_id, expected_attempt_id, runtime)`.
It derives workspace and specification from immutable launch input; callers cannot supply a
container selector. Only an exact incomplete attempt using `posix-flock-abort-v1` is eligible.
Completed launches use the existing receipt-bound stop API. Legacy protocols, wrong attempts,
changed ownership and ambiguous daemon results reject automatic reconciliation.

The returned `RuntimeAbortReceipt` contains the launch-input reference, stage/attempt,
specification digest, observed container/start when available, retained observation references,
terminal `absent` or `stopped-and-removed` outcome, observation time and any auxiliary log error.
It is not a readiness receipt. Partial failures retain intent and evidence without declaring
launch failure. Producer `cleanup_unserved` reports `aborted` only after terminal verification;
`needs_reconciliation` keeps the retired endpoint reserved. POSIX process locking and a shared
trusted runtime workspace are required; Windows launch/abort fails closed.


## Warm-runtime drain ownership

These are trusted Python operations, not public HTTP cleanup endpoints:

- `WarmRouteStore.admit(deployment_id)` atomically returns a pinned `AdmissionLease` and records
  its durable obligation. Only the owning gateway finalizer acknowledges confirmed closure.
- `WarmRouteStore.retire_drained(control, backend)` returns true only after permanent admission
  retirement and zero outstanding obligations are recorded together. Current routes,
  active/known-good targets, legacy protocols and unresolved obligations prevent reclamation.
- `WarmDrainReceipt` binds the store identity, revision/digest, admission protocol, zero
  remaining obligations and observation time in the route database.
- `producer_pipeline.borrowed_collection` owns a direct collector obligation through task and
  client cleanup. Borrowed network helpers require that exact task/execution/store/revision
  context. Failed tasks retain their obligation; no heartbeat or timeout expires it.

Producer cleanup reports `needs_reconciliation` for an incomplete drain and retains the
endpoint reservation. It preserves the existing exact runtime-stop receipt after successful
drain. New protocol stores reject old readers/writers through snapshot schema and SQLite
write guards; existing stores cannot be upgraded in place to claim old streams drained.


## Local capacity control

Trusted Python entrypoints:

- `WarmRouteStore(path, capacity_enabled=True)` creates a new capacity-protocol store.
  Opening an existing store reads its immutable mode; legacy stores cannot be enrolled.
- `freeze_capacity(journal, routes, control, plan)` validates canonical release approval,
  exact stable primary and immutable launch assets, then reserves deployment authority.
- `CapacityController(...).tick()` derives actual tagged gateway observations and executes
  one bounded lifecycle/reconciliation step. `state()` and `observation()` expose durable
  progress and current demand; no caller-supplied desired-load override is accepted.
- `CapacityController.close()` stops new extra-member admission and attempts exact owned
  drain/cleanup. A blocked result retains its runtime slot and stream obligations.

Run `python -m finserve.registry.capacity_cli --plan PLAN.json --routes ROUTES.sqlite
--control CONTROL.sqlite --output RESULT.json` with the existing trusted registry/artifact
environment. The route/control stores must already exist; the plan cannot name a new image
through an HTTP request. Exit 2 preserves a failed or unresolved result. Windows runtime
execution fails closed because the shared attempt fence requires POSIX flock.

Capacity responses retain `x-finserve-revision`, `x-finserve-revision-digest` and
`x-finserve-route-generation` for the actual physical dispatch. They add
`x-finserve-anchor-revision`, `x-finserve-anchor-digest`, `x-finserve-anchor-generation`
and `x-finserve-pool-generation`. A pre-dispatch error has only anchor metadata. Optional
benchmark `RequestRecord.routing` retains this distinction without changing legacy rows.
The private `x-finserve-primary-probe` header accepts only bounded one-use authority minted
by the route store for an exact request and generation; it does not bypass normal auth.
