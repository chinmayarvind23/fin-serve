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
| `registry/{annotations,explorer}.py` | Recomputed GPU/quality associations and bounded read-only GraphQL |
| `reliability/{promotion,rollback,warm_routes}.py` | Performance/quality gates, rollback reconciliation and actual route CAS |
| `telemetry/{metrics,tracing}.py` | Fixed-cardinality Prometheus collectors and bounded sampled OTel export |

## Stream accounting and ownership

SSE frames are transport envelopes, not tokens. The adapter accumulates visible output while a state machine validates one supported finish reason, authoritative usage and the terminal completion marker. A missing or malformed terminal sequence fails. The Ray bridge uses newline-delimited JSON and withholds terminal success until complete EOF; another replica cannot take over an already visible response.

The response object and generator share an idempotent lease. Header failure can occur before the generator starts, so generator-finally cleanup alone is insufficient. Response cleanup explicitly closes the iterator and releases the lease. Native work offloaded to a thread is drained before returning its capacity, because task cancellation cannot stop an executing kernel or image decoder.

The main application uses an asynchronous exit stack so a failed visual-worker startup or modality close cannot suppress closure of the remaining engine, quota or tracing resources. Uvicorn owns normal active-request draining before lifespan shutdown.

## Input and queue bounds

Text bodies are limited to 131,072 bytes. Native chat retains the original roles/content through Ray and uses the backend chat template. A deterministic labeled prompt supports reference scheduling; it must match those messages exactly. The duplicated internal prompt/message representation is bounded to 120 KiB, reserving 8 KiB for Ray request metadata. The image route accepts at most 1,450,000 envelope bytes, 1,048,576 decoded PNG bytes and 512 pixels per side. It authenticates and acquires its image slot before receiving the body, validates chunk checksums and format restrictions before native decoding, then publishes a metadata-free canonical PNG. Text engines never receive the image contract as an implicit fallback.

The Bun evidence surface accepts only authenticated POST `/graphql`, with 16 KiB requests and 2 MiB responses. The Python read service authenticates before body buffering, admits four owners, bounds body/query/send time, limits AST/list expansion and charges serialized output during resolution. Artifact input has a 32 MiB query budget; cache keys include the complete artifact reference. SQL metadata is length-limited before Python allocation. The initial service uses local SQLite and local files; a native filesystem stall can outlive the HTTP deadline, but cannot release its owner slot early.

The inference edge has its own coarse bound and passes cancellation upstream. FastAPI, Ray and the engine each own distinct work. A proxy admission count must never be labeled as physical GPU utilization or KV-cache occupancy.

## Durable state and identities

Visual-job idempotency includes the normalized request and principal. The coordinator persists generation/state and fences worker results. An artifact is complete only after its bytes and expected digest have been verified and the corresponding terminal job update succeeds. Cancellation uses worker-instance identity so an old cancellation cannot affect a replacement process.

Registry objects use immutable content hashes and database constraints. Identical retries succeed; conflicting identity reuse fails. Run registration validates manifest/request/summary consistency before publishing CAS references. Artifact reads verify namespace, size and SHA256. Quality annotation additionally verifies the installed grader source (only LF/CRLF normalization is accepted), the reference's embedded suite and answer linkage. GPU annotation reconstructs the exact measured clock window and rejects shifted epoch boundaries with an absolute tolerance.

A lifecycle specification names baseline/candidate runs, quality and suite references, expected route revision/generation and target Revision. Canonical mode requires both serving profiles. Profile digest includes engine version/parameters, pinned model/tokenizer manifests, served model, endpoint and the name of a credential environment variable; it never contains the credential value. Image identity remains in Revision to avoid a hashing cycle.

Warm routes are separate active traffic truth. Every cutover compares the expected revision and generation, records a stable action fingerprint, and commits the new route plus action receipt atomically. A request pins one snapshot. Lifecycle state versions count persisted transitions; they are not deployment route generations.

## Measurement definitions

`requests_per_second = successful_measured_requests / measured_seconds`.

`tokens_per_second = sum(authoritative_generated_tokens_for_successful_measured_requests) / measured_seconds`.

Client TTFT is send-to-first-nonempty-content on the client's clock. Server TTFT uses the server's own receipt-to-first-content duration. End-to-end percentiles use successful request send-to-complete durations; failures remain in the denominator for success rate and in raw records. Warmup is retained separately and excluded from measured summaries. Open-loop scheduled-to-complete delay includes queueing before send; the configured arrival rate is unused in closed-loop mode.

GPU utilization integrates sample-held physical-device observations over a declared epoch-mapped measured interval, caps stale gaps and reports coverage. Below 95% coverage, mean utilization is unknown. Multiple processes sharing one GPU do not increase the physical device count. Cost requires a declared price and billed/modelled time; local throughput alone cannot establish a cloud cost reduction.

The quality grader distinguishes agreement with a reference from correctness against expected answers. A pair of identical wrong answers can have high parity. Missing/invalid structured outputs and exact-format failures remain explicit, and a bad result does not justify editing the frozen suite afterward.
