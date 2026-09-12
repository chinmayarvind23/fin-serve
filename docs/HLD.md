# High-level design

FinServe separates active inference, durable jobs, release control, and browsing. Each boundary has its own admission, identity, and failure rules.

## Request paths

The Bun edge authenticates requests, bounds bytes, and propagates disconnects. FastAPI validates contracts and deadlines, applies optional Redis quotas, and holds admission through stream cleanup. Text requests reach an external vLLM/SGLang engine directly or through the Ray HTTP router. The engine owns model weights, prefill, decode, continuous batching, and KV memory.

The Ray router filters unhealthy, stale, full, or incompatible endpoints before choosing a backend. Immutable leases bind request ownership to one endpoint. CPU proxy replicas, model engine processes, and physical GPU nodes are separate capacities.

Image requests use a separate admission pool. The gateway validates and canonicalizes an inline PNG before calling a pretrained vision-language engine. Encoding and language decoding remain in that engine.

Visual-generation jobs use durable SQLite intent and a gRPC JAX/Flax reference worker. Submission is idempotent within a tenant. Worker-instance and generation fences prevent late results from completing a newer attempt. This worker demonstrates untrained autoregressive image-token generation.

## State ownership

| State | Owner |
| --- | --- |
| Streaming admission and request leases | Gateway and Ray router |
| Shared quotas | Redis |
| Visual job lifecycle | SQLite coordinator |
| Models, runs, revisions, and decisions | SQLAlchemy registry |
| Immutable artifacts | Content-addressed local store or S3 adapter |
| Active endpoint and known-good revision | Generation-checked route store |
| Experiment tracking copies | MLflow |

Artifacts have a namespace, digest, and byte length. SQL stores references rather than mutable output blobs. Readers verify bytes before use. MLflow is an operator-controlled mirror; SQL and artifact identities remain authoritative.

## Release workflow

The producer freezes model files, builds an image from committed source, launches the exact runtime, and checks readiness through an inference stream. Collection stages retain all offered requests. The shared gate rechecks the baseline/candidate artifacts and configuration identities before recording approval or rejection.

Airflow orchestrates the same operations used by the CLI. Attempt journals prevent retries from silently creating duplicate engines. Interrupted stages reconcile their owned resources before proceeding.

Promotion switches a registered route with expected revision and generation checks. Existing streams finish on their original backend. Activation acknowledgment binds controller state to the observed route. Healthy probation advances the known-good revision; detected regressions initiate rollback to a verified running endpoint.

## Capacity control

The local controller can add an equivalent engine to a canonically approved primary. A durable global slot includes warming and uncertain allocations. Membership changes select physical endpoints and stop new admission before removal. The controller waits for stream obligations to drain, then stops the exact runtime it owns. Health probes select the primary without contributing to scaling demand.

KubeRay manages Ray process placement and optional CPU worker autoscaling. Kubernetes engine workloads and the AWS node autoscaler have distinct owners. Terraform defines node bounds and workload identities; an operator deploys and verifies those definitions in the target account.

## Read and observation paths

The explorer uses a bounded read-only GraphQL API. It reads registered artifacts and lifecycle records without deployment authority. OpenTelemetry carries sanitized request causality across private hops; Prometheus exposes fixed-cardinality operational signals. Export work runs outside token iteration.

See [low-level design](LLD.md), [API contracts](api-contracts.md), [deployment](deployment.md), and [security](security.md) for the concrete interfaces.
