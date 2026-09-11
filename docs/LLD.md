# Low-level design

The source packages below are implemented. Runtime and acceptance limits are separated from the type contracts they exercise.

## Code map

| Module | Mechanism |
| --- | --- |
| `contracts/inference.py` | Strict bounded text/chat input, request IDs and engine token envelopes |
| `contracts/vision.py` | Single inline PNG, image/text capability, pixel/output/deadline envelope |
| `gateway/app.py` | Application-scoped serving dependencies, lease-owned SSE, metrics and lifespan cleanup |
| `gateway/body_limit.py` | Actual byte counting and exact-route delegation to independently bounded handlers |
| `gateway/vision.py` | Authentication and image admission before body receive, CPU decode ownership, bounded generation/send |
| `engines/openai_adapter.py` | Incremental bounded SSE parsing, finish/usage/DONE state and pooled HTTP transport |
| `engines/ray_http.py` | Typed NDJSON client, fixed replica identity, EOF-gated terminal accounting |
| `engines/pytorch_reference.py` | Reference causal decode with cached/full-prefix paths and native-step cleanup |
| `scheduler/{policy,router}.py` | Eligibility, stale-snapshot rules, score calculation and immutable reservation leases |
| `multimodal/{images,preprocess_http,vision_benchmark}.py` | Strict PNG canonicalization and actual local/HTTP stage measurement |
| `multimodal/{jobs,visual_rpc,visual_wire,jax_generator}.py` | Durable jobs, typed gRPC ownership and the visual reference generator |
| `benchmark/{runner,experiment,metrics,gpu,cost}.py` | Raw request records, provenance, metric populations, GPU integration and explicit cost inputs |
| `evaluation/quality.py` | Frozen exact/typed-JSON grading and separate correctness/parity metrics |
| `registry/{metadata,artifacts,lifecycle,release_gate,pipeline}.py` | Immutable SQL/CAS state and recomputed release decisions |
| `registry/{model_assets,runtime_build,engine_entrypoint}.py` | Frozen model download, committed-source image build and startup verification of actual mounted bytes |
| `registry/{producer_stages,producer_tasks,quality_collection}.py` | Durable attempt ownership, verified stage receipts, unchanged-suite collection and raw-evidence reconstruction |
| `registry/{managed_runtime,runtime_probe,runtime_stages}.py` | Exact Docker launch ownership, bounded real streamed readiness, upstream receipt binding and verified cleanup |
| `registry/{annotations,explorer}.py` | Recomputed GPU/quality associations and bounded read-only GraphQL |
| `reliability/{promotion,rollback,warm_routes}.py` | Performance/quality gates, rollback reconciliation and actual route CAS |
| `telemetry/{metrics,tracing,propagation}.py` | Fixed-cardinality metrics, private trace propagation and sanitized JSONL/OTLP export |
| `telemetry/langfuse_probe.py` | Bounded local ingestion and API verification of exact trace/span identity without prompt content |
| `http_ownership.py` | One retained response-close task through repeated cancellation, with explicit unresolved cleanup failure |

## Stream accounting and ownership

Producer failure cleanup uses immutable launch and build receipts, so damaged model files do
not prevent stopping an exactly identified owned process. It retires only revisions that have
never served a warm route and are absent from every controller's active and known-good targets.
The retirement transaction locks routes before controller state and fences subsequent route
writes. The endpoint remains reserved until exact process stop is verified; only then can a
new revision reuse it. Retired identities and their original receipts remain immutable.
Previously served revisions remain protected because the gateway has no durable proof that
all pinned streams have drained. Incomplete launch attempts require reconciliation.

SSE frames are transport envelopes, not tokens. The adapter accumulates visible output while a state machine validates one supported finish reason, authoritative usage and the terminal completion marker. A missing or malformed terminal sequence fails. The Ray bridge uses newline-delimited JSON and withholds terminal success until complete EOF; another replica cannot take over an already visible response.

The response object and generator share an idempotent lease. Header failure can occur before the generator starts, so generator-finally cleanup alone is insufficient. Response cleanup explicitly closes the iterator and releases the lease. Native work offloaded to a thread is drained before returning its capacity, because task cancellation cannot stop an executing kernel or image decoder.

The main application uses an asynchronous exit stack so a failed visual-worker startup or modality close cannot suppress closure of the remaining engine, quota or tracing resources. Uvicorn owns normal active-request draining before lifespan shutdown.

## Input and queue bounds

Text bodies are limited to 131,072 bytes. Native chat retains the original roles/content through Ray and uses the backend chat template. A deterministic labeled prompt supports reference scheduling; it must match those messages exactly. The duplicated internal prompt/message representation is bounded to 120 KiB, reserving 8 KiB for Ray request metadata. The image route accepts at most 1,450,000 envelope bytes, 1,048,576 decoded PNG bytes and 512 pixels per side. It authenticates and acquires its image slot before receiving the body, validates chunk checksums and format restrictions before native decoding, then publishes a metadata-free canonical PNG. Text engines never receive the image contract as an implicit fallback.

The Bun evidence surface accepts only authenticated POST `/graphql`, with 16 KiB requests and 2 MiB responses. The Python read service authenticates before body buffering, admits four owners, bounds body/query/send time, limits AST/list expansion and charges serialized output during resolution. Artifact input has a 32 MiB query budget; cache keys include the complete artifact reference. SQL metadata is length-limited before Python allocation. The initial service uses local SQLite and local files; a native filesystem stall can outlive the HTTP deadline, but cannot release its owner slot early.

The inference edge has its own coarse bound and passes cancellation upstream. FastAPI, Ray and the engine each own distinct work. A proxy admission count must never be labeled as physical GPU utilization or KV-cache occupancy.

## Durable state and identities

Initial producer rollout freezes its traffic URL and probation settings before collection.
After reconstructing producer receipts and canonical approval, route preparation registers
the gate's immutable profiles and selects only the expected baseline. A successful ingress
smoke is checked again under the route write lock before controller initialization. Candidate
deployment, acknowledgment and probation then use the existing lifecycle action and generation
fences. The DAG requires both successful probation and settled cleanup at its terminal task.
Current producer jobs mint new baseline IDs, so initial rollout rejects an existing deployment
before collection; stable-baseline receipt reuse for later releases remains unimplemented.

Route and rollback stores retain a persistent UUID created during explicit initialization.
Producer execution freezes both UUIDs with their paths. Task reopens require those identities
and do not create schemas. Every transaction opens the existing database with SQLite
`mode=rw` and checks its UUID after acquiring the write lock. A missing file cannot be silently
recreated between validation and open; a replacement database cannot become empty authoritative
state for cleanup. This protects local task retries, and is not a database backup or recovery
protocol.

Visual-job idempotency includes the normalized request and principal. The coordinator persists generation/state and fences worker results. An artifact is complete only after its bytes and expected digest have been verified and the corresponding terminal job update succeeds. Cancellation uses worker-instance identity so an old cancellation cannot affect a replacement process.

Registry objects use immutable content hashes and database constraints. Identical retries succeed; conflicting identity reuse fails. Run registration validates manifest/request/summary consistency before publishing CAS references. Artifact reads verify namespace, size and SHA256. Quality annotation additionally verifies the installed grader source (only LF/CRLF normalization is accepted), the reference's embedded suite and answer linkage. GPU annotation reconstructs the exact measured clock window and rejects shifted epoch boundaries with an absolute tolerance.

Producer stage identity includes canonical input and its local namespace. The SQL journal uses compare-and-swap versions and unique attempt tokens; a running attempt cannot restart merely because time elapsed. Completed model receipts rehash the local snapshot before use. Quality receipts reconstruct all raw rows and suite linkage before reuse. Canonicalization validates nested numeric defaults before hashing, so a JSON round trip cannot change an input's identity. Native disk writes and HTTP close tasks remain owned through cancellation. A failed local transport close preserves the offered request but leaves its stage running for reconciliation; it does not prove that remote GPU work stopped.

Managed runtime stages bind completed model and image receipts to a canonical launch specification. The Docker adapter verifies the actual image, command, environment, mounts, UID and resource bounds before adopting a named attempt. Its local unauthenticated endpoint is loopback-only. A successful readiness receipt requires HTTP 200, visible content, finish reason, authoritative token usage and the terminal SSE marker. Completed replay probes the same container ID and `StartedAt`; it cannot restart a missing or changed process. Cleanup binds the exact launch receipt, preserves logs/inspection and confirms stopped state before removal. Concurrent completion can share a receipt only for the same attempt and container start.

The CPU Hugging Face package reuses the explorer and proxy on port 7860, with Python bound to loopback 8050. The launcher removes inherited Uvicorn worker settings, starts one Python worker and drains both direct children. Distinct runtime-only edge/internal keys prevent browser credentials from becoming internal service credentials. Public routes serve six committed aggregate assets; the initial SQLite/CAS registry is empty and ephemeral. The external package manifest records exact copied bytes, while hosted build and browser acceptance remain separate observations.

A lifecycle specification names baseline/candidate runs, quality and suite references, expected route revision/generation and target Revision. Canonical mode requires both serving profiles. Profile digest includes engine version/parameters, pinned model/tokenizer manifests, served model, endpoint and the name of a credential environment variable; it never contains the credential value. Image identity remains in Revision to avoid a hashing cycle.

Warm routes are separate active traffic truth. Every cutover compares the expected revision and generation, records a stable action fingerprint, and commits the new route plus action receipt atomically. A request pins one snapshot. Lifecycle state versions count persisted transitions; they are not deployment route generations.

## Measurement definitions

The producer task runtime freezes a `ProducerInput` before fetching. After verified model
fetch and image build, it derives both runtime revisions, serving profiles and collection
specifications from those receipts. It verifies source/model/image linkage and the model's
expected workspace path, then freezes the release plan before offering inference. Dispatch
validates task names at runtime and uses one-colon journal IDs such as
`job:baseline-performance`. Only the job ID and task name cross the orchestration boundary;
each task reloads immutable inputs. Launch and collector retry behavior stays with the
existing durable stages. Full DAG deployment, probation and failure cleanup are still being
connected; this task runtime alone is not complete release automation.

Probation completion reconstructs the full monitor observation chain and requires at least
two healthy probes, the configured number of probes, and the configured interval between
adjacent completion times. The CAS evidence includes the frozen monitor policy and ordered
observation references. The completion API rechecks canonical lifecycle approval and binds
the monitor to its deployment, target digest and activation generation. Fresh health and the
unchanged lifecycle route-action receipt are required under the route-to-control lock order
before `mark_stable` persists that evidence digest and advances known-good. A successful
replay uses the same digest; a failed or stale window never becomes probation approval.

Canonical activation acknowledgment is a separate step after lifecycle promotion. It
recomputes the gate, requires the same decision recorded by the promoted lifecycle job, and
performs a fresh warm-route health probe. Under the route store's write transaction it checks
the lifecycle action fingerprint and exact current revision/generation, then calls the control
store's idempotent activation method. Lock order is route then control; the stores use separate
SQLite files and network calls happen before locking. This prevents a concurrent route change
from interleaving with acknowledgment. Known-good remains the previous revision until explicit
probation approval. Failed or unacknowledged activation is still a reconciliation case.

Managed release registration begins with a completed immutable plan stage. Its inputs freeze
policy and both cohort specifications before collection. Registration checks every historical
collection attempt's start time and both retained pre-collection runtime observations against
the plan's completion time. This rejects a new stage carrying an older valid receipt. It then
reconstructs raw performance and quality artifacts, checks their stage inputs and shared launch,
and builds `QualityEvidence`, `LifecycleSpec` and `GateRequest`. The lifecycle is canonical from
creation, so interruption before profile publication cannot downgrade its gate mode. Repeating
registration recomputes evidence and uses immutable registry writes; it does not recollect data.

The managed quality stage shares the completed launch reference with performance collection.
One durable attempt owns the pre-collection runtime observation, raw quality collection,
post-collection observation and receipt publication. Both observations must name the same
container ID, start time, launch digest and attempt. The recorded wall-clock collection interval
must lie between them. Raw quality reconstruction and the frozen specification digest are
checked before stage completion and on replay. The interval is collection provenance; it is
not a per-request latency measurement. Process restarts reject publication, while a failed
transport close retains an unresolved stage. Completed receipts can replay after the runtime
stops because replay validates retained evidence without making new health or inference calls.

`PerformanceCollectionSpec` binds the workload and request configuration to a serving profile,
runtime revision and separate collector commit. The producer journal freezes this specification
and the completed launch reference. Before and after collection, the Docker adapter checks the
same container ID, start time and runtime specification. Publication verifies that the measured
epoch window lies between those observations, recomputes serving and GPU summaries, and
registers the raw artifact references. Replay reconstructs and validates those artifacts without
reoffering inference. A managed GPU annotation follows this verified receipt chain instead of
assuming that collector source and engine-image source are the same commit.

The collector uses an isolated event loop so synchronous evidence writes do not block
orchestration. Its limit is 1,024 workload items, 65,536 total requests and 128 concurrent
workers, with at most two hours and 512 MiB of observed raw artifacts. Limits are checked
between writes; they do not impose a hard disk quota. After a failed HTTP close, the benchmark
fences new offers before persisting the failure row, drains peer tasks, and preserves cleanup
uncertainty even if a raw write also fails. The producer leaves that attempt running for explicit
reconciliation. Successful local draining permits a failed attempt; it does not establish remote
inference termination.

`requests_per_second = successful_measured_requests / measured_seconds`.

`tokens_per_second = sum(authoritative_generated_tokens_for_successful_measured_requests) / measured_seconds`.

Client TTFT is send-to-first-nonempty-content on the client's clock. Server TTFT uses the server's own receipt-to-first-content duration. End-to-end percentiles use successful request send-to-complete durations; failures remain in the denominator for success rate and in raw records. Warmup is retained separately and excluded from measured summaries. Open-loop scheduled-to-complete delay includes queueing before send; the configured arrival rate is unused in closed-loop mode.

GPU utilization integrates sample-held physical-device observations over a declared epoch-mapped measured interval, caps stale gaps and reports coverage. Below 95% coverage, mean utilization is unknown. Multiple processes sharing one GPU do not increase the physical device count. Cost requires a declared price and billed/modelled time; local throughput alone cannot establish a cloud cost reduction.

The quality grader distinguishes agreement with a reference from correctness against expected answers. A pair of identical wrong answers can have high parity. Missing/invalid structured outputs and exact-format failures remain explicit, and a bad result does not justify editing the frozen suite afterward.

Trace context attaches only during generator execution, never across a consumer yield. Private Ray RPC arguments carry bounded W3C identity; the gateway starts a fresh root. Exporter configuration, queue/response bounds and native shutdown limitations are documented in [observability](observability.md). Actual CPU Ray actors and HTTP fixtures verify the causal parent chain. A pinned local Langfuse stack accepted and returned exact trace/span IDs with null prompt/output fields. The six-process fixture experiment retained all 6,144 successful requests but did not identify an isolated tracing penalty. vLLM kernel spans and cloud ingestion remain separate work.
