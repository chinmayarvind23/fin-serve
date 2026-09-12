# Low-level design

The source packages below are implemented. Runtime and acceptance limits are separated from the type contracts they exercise.

## Code map

| Module | Mechanism |
| --- | --- |
| `contracts/inference.py` | Strict bounded text/chat input, request IDs and engine token envelopes |
| `contracts/output_constraint.py` | Bounded caller output shapes, fixed native grammars and terminal syntax validation |
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

## Durable warm-runtime drain

Fresh route stores initialize `durable-http-close-v1`. Their route snapshots carry an
`admission_protocol` field that older extra-forbid schemas reject. SQLite triggers reject
legacy route, event and action writes; a racing old initializer cannot publish an untracked
route. Existing stores are never retrofitted. Ordinary legacy gateway and route replay remain
available, but historical-runtime reclamation is disabled for those stores.

`WarmRouteStore.admit` selects the route, binds its immutable backend and inserts a unique
`warm_admissions` row under the retirement transaction fence. `WarmRouteMiddleware` holds the
obligation through the whole ASGI response. Direct `create_app(WarmRouteEngine(...))` plus
middleware composition participates too; an untracked engine composition rejects. The lease
contains the endpoint binding so an already admitted stream can finish after retirement
blocks ordinary backend lookup. `AdmissionTransport` counts entry before backend send and
wraps each response in the shielded owned-close stream. An obligation is released only after
ASGI work unwinds and every entered backend response has a positively completed close task,
or no backend send was entered. Receiving DONE alone is insufficient. A failed send, failed
close, lost acknowledgement or process death leaves a durable row. Outstanding rows are
bounded at 4,096; their age never authorizes removal.

`retire_drained` acquires route then control transactions, rejects any current route or
active/known-good controller target, and persists an irreversible retirement tombstone.
It writes an immutable `WarmDrainReceipt` only when zero obligations remain, in that same
transaction. No transaction waits for a stream to finish. A later cleanup call can complete
a pending drain. `cleanup_unserved` uses this path when historical traffic prevents the
never-served path, keeps endpoints reserved while obligations remain, and reports
`needs_reconciliation` instead of treating that pending cleanup as settled. Exact runtime
stop and endpoint release follow the drain proof; retired identities cannot be reactivated.

A new producer `FrozenExecution.collection_protocol` rejects older task parsers.
`producer_pipeline.borrowed_collection` atomically verifies the current stable borrowed
revision and registers an obligation before any borrowed task work. The obligation outlives
both collection and HTTP-client closure. The helper checks exact frozen input, store identity,
revision and owning asyncio task; copied child-task context cannot authorize another task.
Successful completion acknowledges the row, while any exception, cancellation or process
death conservatively retains it for explicit reconciliation. Borrower cleanup still never
stops its foreign baseline. Once successful borrowers finish and another release is stable,
the original owner may reclaim its now-inactive runtime.

## Stream accounting and ownership

Explicit output constraints survive chat conversion and Ray serialization. Unconstrained requests omit the new field to preserve existing wire identities. The vLLM adapter translates fixed typed shapes into `structured_outputs`; unknown native capabilities and fixture/reference engines reject them. Warm routing selects the adapter from the pinned backend revision. These choices do not change engine batching or the release evaluator.

For constrained requests, the native adapter retains at most 128 KiB of text while streaming original fragments. At DONE it validates syntax before publishing terminal success. This catches incomplete JSON at a token limit, duplicate keys and wrong JSON value types without repairing the answer. The buffer applies only to explicitly constrained requests. Invalid streamed content remains visible as partial failed output. Managed profiles optionally pin xgrammar with whitespace permitted; absent configuration preserves historical profile hashes.

`benchmark.constraint_mapping` binds exact original prompt hashes to caller-declared shapes. The complete bounded map and selected wire transport enter the shared request-mapping digest. Both collection specs preflight every prompt; raw reconstruction rechecks successful output syntax. No constraint derives from an expected answer or case label. Producer templates freeze the map before image construction, require xgrammar in both parameter sets, and resolve executable runtime identities from receipts. Direct payloads, collection specs and evidence reject an unresolved template. JSON receipt comparisons use JSON-mode serialization so nested tuple fields reconstruct as their recorded arrays. Actual constrained GPU collection is complete, with separate regression and independent quality results retained. Isolated grammar first-use cost remains unmeasured.

Producer failure cleanup uses immutable launch and build receipts, so damaged model files do
not prevent stopping an exactly identified owned process. It retires only revisions that have
never served a warm route and are absent from every controller's active and known-good targets.
The retirement transaction locks routes before controller state and fences subsequent route
writes. The endpoint remains reserved until exact process stop is verified; only then can a
new revision reuse it. Retired identities and their original receipts remain immutable.
Previously served revisions in legacy stores remain protected because those gateways have
no durable proof that all pinned streams have drained. Incomplete launches with the immutable
`operation_protocol=posix-flock-abort-v1` input use explicit abort reconciliation. Legacy or
unknown protocols remain unresolved; adding a marker later cannot prove an old executor drained.

`runtime_fence.attempt_fence` holds a never-unlinked kernel `flock` file in the canonical
attempt directory. Reentrance is limited to the same asyncio task; copied child-task context
cannot bypass the lock. `launch_runtime_stage` holds this fence through create/start,
readiness and completion publication. Owned command workers drain through cancellation before
the descriptor closes. Process death releases the kernel lock, but does not cancel an
accepted Docker daemon request.

The adapter therefore freezes `create-request.json` before creation, `create-complete.json`
after successful command return, `allocation.json` after exact inspection, and
`start-request.json` before starting. `first-start.json` binds ID and `StartedAt` before any
health probe. Abort persists `abort-intent.json` before querying Docker. An unacknowledged
create with no observed allocation cannot become an absence claim; an unbound requested start
requires operator reconciliation. Known allocations must match image, specification, mounts,
resources, ID and start. Abort observes stopped state and unchanged start, removes by full ID,
and verifies both ID and attempt-name absence before publishing `RuntimeAbortReceipt` and
`RuntimeAborted` failure. Log capture errors are recorded without skipping safe cleanup;
inspection or evidence-publication failures leave the attempt unresolved. Completed abort
replay only checks absence and refuses reappearance. Automatic launch replay of a
`RuntimeAborted` stage requires a new producer identity.

SSE frames are transport envelopes, not tokens. The adapter accumulates visible output while a state machine validates one supported finish reason, authoritative usage and the terminal completion marker. A missing or malformed terminal sequence fails. The Ray bridge uses newline-delimited JSON and withholds terminal success until complete EOF; another replica cannot take over an already visible response.

The response object and generator share an idempotent lease. Header failure can occur before the generator starts, so generator-finally cleanup alone is insufficient. Response cleanup explicitly closes the iterator and releases the lease. Native work offloaded to a thread is drained before returning its capacity, because task cancellation cannot stop an executing kernel or image decoder.

The main application uses an asynchronous exit stack so a failed visual-worker startup or modality close cannot suppress closure of the remaining engine, quota or tracing resources. Uvicorn owns normal active-request draining before lifespan shutdown.

## Input and queue bounds

Text bodies are limited to 131,072 bytes. Native chat retains the original roles/content through Ray and uses the backend chat template. A deterministic labeled prompt supports reference scheduling; it must match those messages exactly. The duplicated internal prompt/message representation is bounded to 120 KiB, reserving 8 KiB for Ray request metadata. The image route accepts at most 1,450,000 envelope bytes, 1,048,576 decoded PNG bytes and 512 pixels per side. It authenticates and acquires its image slot before receiving the body, validates chunk checksums and format restrictions before native decoding, then publishes a metadata-free canonical PNG. Text engines never receive the image contract as an implicit fallback.

The Bun evidence surface accepts only authenticated POST `/graphql`, with 16 KiB requests and 2 MiB responses. The Python read service authenticates before body buffering, admits four owners, bounds body/query/send time, limits AST/list expansion and charges serialized output during resolution. Artifact input has a 32 MiB query budget; cache keys include the complete artifact reference. SQL metadata is length-limited before Python allocation. The initial service uses local SQLite and local files; a native filesystem stall can outlive the HTTP deadline, but cannot release its owner slot early.

The inference edge has its own coarse bound and passes cancellation upstream. FastAPI, Ray and the engine each own distinct work. A proxy admission count must never be labeled as physical GPU utilization or KV-cache occupancy.

## Durable state and identities

Producer rollout freezes its traffic URL and probation settings before collection.
After reconstructing producer receipts and canonical approval, route preparation registers
the gate's immutable profiles and selects only the expected baseline. A successful ingress
smoke is checked again under the route write lock before controller initialization. Candidate
deployment, acknowledgment and probation then use the existing lifecycle action and generation
fences. The DAG requires both successful probation and settled cleanup at its terminal task.
An update names an existing completed baseline launch stage. Its profile and runtime identity
remain unchanged while new quality/performance stages collect fresh evidence. The initial
launch task becomes an observation of that original start; it creates no new baseline container
or launch receipt. Collection entry checks the borrowed revision against simultaneous registered
route/controller truth and requires active=known-good at the expected generation with no rollback.
Cleanup excludes borrowed runtimes. Omitted baseline references retain the original initial-job
serialization so existing frozen inputs remain replayable.

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
existing durable stages. The 18-task DAG integrates deployment, probation and failure cleanup. Actual GPU
acceptance remains incomplete: run 01 failed during model rehash with ENOMEM and
verified owned cleanup; run 02 was prepared in native WSL but shared-GPU preflight
prevented launch. It produced no new quality or performance evidence.

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


## KubeRay CPU worker autoscaler

ray.autoscaling.enabled selects enableInTreeAutoscaling and the V2 Conservative autoscaler. Enabled initial/min/max workers are1/1/2; disabled remains2/2/2. idleTimeoutSeconds accepts60?600 with default60. The sidecar inherits the pinned head image, requests100mCPU/512Mi and limits500mCPU/512Mi. Head500m+sidecar100m+worker1000m fits the declared1600m per-node staging allowance; actual system requests, memory and overlapping upgrade clusters still require live checks. Unknown autoscaler options fail schema validation.

Enabled head templates omit serviceAccountName so KubeRay1.6.1 creates a cluster-named ServiceAccount, Role and RoleBinding. Its Role permits namespace-wide Pod get/list/watch/patch, Pod resize patch, and RayCluster get/patch. Head containers share the projected token. Engine and worker templates retain the separate runtime account with automountServiceAccountToken=false. This opt-in does not duplicate Serve actors or infer GPU capacity from CPU Pods. Active actors may keep a worker non-idle; no consolidation or request-driven throughput gain is claimed. Actual Helm rendering and the pinned RayService CRD validate both modes, but API admission and observed scale/drain cycles remain pending.


## EKS node scaling identities and topology

`enable_node_autoscaling` defaults to false. Enabled configuration creates nine ASG tags: three for CPU and six for GPU. Both groups carry cluster discovery tags; GPU metadata also carries the pool, accelerator, taint and GPU count required for scale-from-zero scheduling. Tags target the ASG names returned by the managed node-group resources. IAM scaling writes require these actual ASG name patterns plus both cluster tag conditions. Only the exact `kube-system/finserve-cluster-autoscaler` service account and STS audience can assume the role. Capacity discovery is read-only; DescribeNodegroup is restricted to the two node-group ARNs.

Both node groups ignore subsequent `scaling_config[0].desired_size` drift. CPU initial/min/max is 2/2/3 in both modes; GPU initial defaults to zero with bounds 0/1. Disabling role/tag creation cannot reduce the CPU maximum below an autoscaler-selected desired size of three. Stop the controller before disabling, and use an explicit safe EKS desired-size update if shrinking is required. GPU subnet selection is fixed to private subnet zero so a node returning from zero can mount the retained model PV in that AZ. Existing installations must review node-group replacement and PV topology; no volume migration is implied. Terraform validate and five mocked plan runs pass, but no live API acceptance, controller installation or scale cycle has been verified.


## Pinned node controller contract

`infra/kubernetes/node-autoscaler` accepts only clusterName, awsAccountId and awsRegion. Release and service account are fixed to finserve-cluster-autoscaler in kube-system; Kubernetes minor must be 1.35. It derives the Terraform role ARN and uses the official CA 1.35.2 image index aac369dc283927a623deb1af54696efcc722ae79255aa07788422e495bab887d. Explicit leader-lock and status ConfigMap names match the namespaced write permissions. Creation is namespace-scoped without resourceNames, as required by Kubernetes. Cluster-wide reads support scheduling and mandatory DRA metadata; node updates and Pod evictions support drain. Optional provisioning-request and capacity-buffer clients are disabled, and their write permissions are absent.

The single controller requests 100m CPU/600Mi on CPU nodes, caps total nodes at four and drains at most one at a time. It runs without host mounts and disables EC2 metadata credential fallback; the API token and EKS-injected IRSA token retain distinct purposes. Conservative local-storage/custom-controller checks can prevent scale-down. The system-Pod check has an upstream one-hour timeout, not an absolute exemption. A real container parsed the exact rendered flags plus --help with no credentials and networking disabled. This verifies the binary interface, not API/RBAC admission or live reconciliation.

The warm gateway caches at most 32 backend clients. At capacity it closes an idle
retired pool before allocating another; local stream ownership remains pinned when
HTTP closure is uncertain. Pool eviction supplies no remote runtime drain proof.


## Local model capacity

`CapacityPlan` freezes the approved primary launch/model/build references, canonical
approval job, exact route generation, one extra endpoint, up to eight distinct replica
specifications, per-member gateway request limit, sample budget and hysteresis policy.
Enrollment calls `approved_activation`, re-evaluates the canonical gate, verifies the
candidate profile and completed primary launch, and requires active=known-good at the
same route/control generation. Manual bootstrap is insufficient. Replica equivalence
permits only revision ID, endpoint and the resulting full profile digest to differ;
model/tokenizer/image/source/engine parameters and resource limits remain identical.
This explicit replication authorization does not evaluate a changed model or relax quality.

A fresh `WarmRouteStore(..., capacity_enabled=True)` installs immutable protocol metadata,
snapshot fields and write guards. Existing stores cannot opt in. A unique deployment
allocation authority prevents different plans from each creating an extra runtime. Its
slot is reserved before any daemon operation and retained through warming, draining or
uncertain cleanup. The per-plan POSIX fence drains owned operations before cancellation
unlocks it. Store transactions are released before daemon, HTTP or readiness waits.

Capacity middleware does not reserve a pre-authentication stream. After normal auth,
validation, quota and application admission, the engine atomically selects a ready physical
member and writes its durable reservation. Positive return of an upstream HTTP response
marks dispatched serving occupancy; pre-dispatch and collector uncertainty cannot prove
low load. Collector and primary-probe rows are separate populations. Per-member serving
limits count outstanding reservations conservatively; they are a gateway budget, not a
measurement of all native work from arbitrary direct clients.

The controller invokes `launch_runtime_stage`, then rechecks the unchanged stable anchor
before publishing ready membership. On lower sustained load it removes eligibility before
calling `retire_drained` and exact `stop_runtime_stage`. An incomplete startup uses explicit
`abort_runtime_stage`; unresolved daemon mutations retain the global slot. Crashes never
expire stream rows. Ordinary promotion immediately invalidates old pool selection, and
stale warming work is reclaimed without being published. The primary is borrowed and is
never stopped by capacity cleanup. Each generation uses a new frozen revision ID; endpoint
reuse waits for terminal lifecycle cleanup. Generic retirement refuses an eligible pool
member and treats its pool history as served history.

Pool generation is a durable counter with retained history, advanced atomically on route
cutover, enrollment, ready publication and removal. Responses distinguish physical revision,
logical anchor and pool generation. Capacity middleware delays response-start forwarding
until dispatch identity is known. A pre-dispatch error has anchor metadata only. Physical
container/start identity is bound by the membership's immutable RuntimeReceipt.

`WarmRouteAdapter.health` mints a bounded, one-use token tied to the exact route snapshot
and canonical inference request. Normal authenticated gateway dispatch consumes it and
selects the primary. Replay/substitution fails; probe work retains a drain obligation but
is excluded from scaling demand. Token revocation removes unused authority only.

The CLI runs a frozen plan with the real DockerRuntime and attempts explicit extra-member
drain at exit. Its JSON result and route-store event/receipt history retain blocked cleanup.
Synthetic integration covers actual HTTP load, dispatch attribution, held-stream downscale,
concurrent controllers, competing plans, stale anchors and lost create responses. This is
implementation evidence; no hardware throughput, utilization or cost gain follows from it.
