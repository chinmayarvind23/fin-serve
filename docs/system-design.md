# System design walkthrough

FinServe separates online inference from durable jobs, release decisions and evidence browsing. The [high-level design](HLD.md) describes the implemented ownership boundaries; the [low-level design](LLD.md) maps them to source. This walkthrough explains the main design decisions and their limits.

## Follow one request

A Bun edge bounds the allowed route, upload, response and forwarding duration. FastAPI validates the typed request, checks its configured credential and optional Redis quota, then acquires an application slot. Text inference uses a configured external engine directly or the private Ray HTTP bridge. Ray selects one eligible endpoint and admits a worker lease before generation. The engine owns model weights, continuous batching and KV memory.

The response emits visible content and authoritative final token usage. A network chunk is not a token. Failure after visible output remains a failed stream; another replica does not transparently restart it. Header/body failures and disconnects close the owned iterator, while ambiguous Ray retirement retains capacity. Trace context propagates across private calls without remaining attached while a generator yields to its consumer.

An image/text request has its own admission and PNG contract, then invokes a pretrained vision-capable engine. The measured stage split moves CPU normalization behind HTTP; it does not split the vision encoder from language decoding. Durable visual generation follows a separate SQLite/gRPC job path and uses an explicitly untrained JAX/Flax reference model.

## Capacity and scheduling

Throughput counts successful completed work over the whole measured interval. Token throughput uses authoritative output-token totals. Latency and quality constrain whether a gain is useful. Increasing active sequences may improve batching while consuming KV memory and increasing delay.

The router checks health, model/GPU compatibility, snapshot age, memory headroom and reserved/reflected leases. Least-load routing provides the baseline. Adaptive scoring adds a physical-memory term and an explicitly known prefix-affinity bonus. It does not infer per-engine GPU utilization or invent a cache-hit forecast. When two endpoints share one physical GPU and no prefix affinity is declared, that memory term is identical for both; the experiment should not presume an adaptive advantage.

Adding Ray CPU proxies does not add GPU capacity. Engine process capacity, Ray worker placement and cloud GPU node capacity are different controls. The current staging chart reserves one GPU for a separate engine and uses a disruptive Recreate update. An overlapping warm canary requires enough measured capacity for both versions.

## State and control

Redis owns optional quotas and ephemeral cache bytes. SQLite owns durable visual jobs and the tested registry/controller stores. Immutable artifact references bind namespace, size and digest. PostgreSQL and S3 adapters have separate deployment scopes; an RDS/S3 deployment is not implied by local SQLite tests.

The shared gate recomputes raw performance and frozen quality before a candidate may activate. Canonical profiles bind the measured endpoint, model/tokenizer manifests and engine configuration; a Revision binds the image. A route CAS changes active traffic only when its expected revision and generation match. In-flight work keeps its original route. Recovery is verified through real inference and exact revision identity.

Airflow currently coordinates evidence registration, gate evaluation and the trusted deployment callback. Complete producer orchestration and managed probation remain under construction. The first produced GPU candidate was registered as rejected after failing its small development smoke; that smoke cannot replace the unchanged 32-case release suite.

## Evidence and unresolved limits

The completed sustained native comparison retained 6,144 requests. Compiled execution increased token throughput 2.64-fold and reduced p95, but worsened median server TTFT and failed strict quality. Those results are not a production availability or cloud cost claim. [Results](results.md) gives the complete populations and limitations.

Cloud deployment, cold recovery, full hosted observability and the browser demo require their own execution evidence. Local configuration validation and real CPU/GPU integration tests answer narrower questions. The system keeps those boundaries visible instead of treating each installed component as a completed deployment.
