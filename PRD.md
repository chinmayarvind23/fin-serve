# Product Requirements Document

## Problem

A naive model server can produce correct outputs while wasting GPU capacity, creating long queues, producing poor time-to-first-token, failing under burst traffic, and making model regressions hard to detect or roll back.

FinServe is a production-shaped inference platform for engineers who need to answer which serving configuration should run a model on given hardware, how much throughput exists before latency collapses, when batching/speculation help, how requests should route across GPU replicas, and whether optimizations preserve quality.

## User

Primary: ML systems / inference engineer. Secondary: ML engineer, platform engineer, applied AI engineer, reviewer reproducing the benchmark.

## Inputs

- model/tokenizer artifact and revision,
- engine/config,
- GPU class/topology,
- workload manifest,
- concurrency/arrival distribution,
- prompt/output length buckets,
- modality,
- quality suite,
- deployment policy.

## Outputs

- streamed inference result,
- request-level timing,
- latency/throughput distributions,
- token throughput,
- GPU utilization/memory,
- speculative acceptance metrics,
- cost per generated-token unit,
- quality comparison,
- deployment decision and rollback evidence,
- reproducible benchmark manifest.

## Product value

The product is not an LLM behind FastAPI. It is a repeatable system for measuring, selecting, deploying, and operating inference configurations.

## Feature Criteria

### Feature A

`38 -> 94 requests/s` and `2.4x token throughput` require a fixed workload, same declared quality acceptance, hardware declaration, warmup policy, arrival process, request-length distribution, run duration, raw requests, failure accounting, and GPU telemetry.

### Feature B

`6,000+ requests`, `690 -> 295 ms median TTFT`, `3.8 -> 1.9 s p95`, `37% lower GPU cost / 1M generated tokens`, and `99.2% quality parity` require request-level records, pricing snapshot, generated-token counts, and a versioned quality evaluator.

### Feature C

`99.95% successful requests`, `81% average GPU utilization`, and rollback within `94 s` require a load-test success definition, raw GPU samples and aggregation rule, and an induced bad-release experiment with timestamps from detection to restored healthy revision.

## Functional requirements

1. Streaming text inference through an OpenAI-compatible HTTP surface.
2. A small PyTorch reference path that exposes decode/KV-cache/batching semantics for first-principles learning.
3. vLLM primary optimized engine and SGLang comparison path.
4. Continuous batching and bounded admission.
5. Speculative decoding with proposal/acceptance telemetry.
6. Adaptive routing using bounded features such as load, queue state, GPU class, prefix/cache affinity, modality, length bucket, and SLO class.
7. JAX/Flax multimodal reference and at least one production-compatible multimodal engine path.
8. Reproducible benchmark harness with deterministic workload manifests, warmup, cancellation, timeout, streaming clients, and per-request records.
9. Versioned output-quality parity evaluation.
10. RDS/S3/MLflow model/deployment/benchmark registry.
11. Airflow offline lifecycle: fetch -> verify -> optimize -> build -> evaluate -> performance gate -> deploy -> promote/rollback.
12. Rollback to immutable known-good revision.
13. OTel, Prometheus, Langfuse, Grafana/CloudWatch observability.
14. Vercel TypeScript/Bun dashboard; GraphQL for nested read-oriented exploration only.

## Non-functional requirements

### Performance

- streaming path does not require Redis queue traversal,
- bounded request admission,
- backpressure before unbounded GPU queueing,
- benchmark overhead measured separately,
- tracing sampling under high load.

### Reliability

- bounded queues,
- timeout/cancellation propagation,
- safe retries,
- readiness/health,
- graceful drain,
- idempotent deployments,
- rollback.

### Scalability

Scale three layers independently: ingress/router replicas, Ray Serve replicas, and Ray/Kubernetes GPU nodes.

### Security

Authenticate control APIs, use least-privilege IAM, verify artifacts, keep secrets out of images/repos, enforce quotas/size limits, and pin model/image revisions.

### Reproducibility

Every benchmark records git SHA, image digest, model/tokenizer revision, engine/runtime/CUDA versions, GPU model/count, Ray/K8s config, workload hash, seed, warmup, load shape, metric version, and pricing snapshot.

## Technology roles

| Technology               | Role                                 | Main tradeoff                       |
| ------------------------ | ------------------------------------ | ----------------------------------- |
| Python                   | serving, benchmark, ML systems       | CPU hot paths/concurrency need care |
| TypeScript/Bun           | web/edge                             | second runtime                      |
| FastAPI                  | public/control API                   | not GPU scheduler                   |
| gRPC                     | selected internal typed streaming    | schema/ops complexity               |
| GraphQL                  | dashboard reads                      | off hot path                        |
| PyTorch                  | reference/mainstream runtime         | framework overhead                  |
| JAX/Flax                 | visual-token reference/comparison    | compile/separate ecosystem          |
| vLLM                     | primary text engine                  | engine-specific tuning              |
| SGLang                   | alternative engine                   | more integration surface            |
| Ray Serve                | distributed serving/routing          | cluster complexity                  |
| Redis                    | rate limits/cache/async state        | not durable truth                   |
| RDS                      | metadata/audit truth                 | not large artifacts                 |
| S3                       | large durable artifacts              | object semantics                    |
| MLflow                   | eval/optimization/promotion evidence | process discipline                  |
| Airflow                  | offline lifecycle DAG                | wrong for online path               |
| EKS/KubeRay              | GPU/Ray orchestration                | operational cost                    |
| Terraform                | AWS reproducibility                  | state/provider maintenance          |
| OTel/Prometheus/Langfuse | tracing/metrics/eval exploration     | overhead/cardinality                |
| Elasticsearch            | optional ops search                  | not needed for MVP                  |

## Failure cases

- unbounded queue growth,
- throughput gain that violates p95,
- low speculative acceptance,
- prefix affinity overloads one replica,
- autoscaling colder than traffic spike,
- GPU OOM/KV pressure,
- Ray/node loss,
- model corruption,
- quantization quality loss,
- benchmark compares different workloads,
- telemetry distorts result,
- rollback artifact incompatibility,
- multimodal stage imbalance,
- client disconnect leaks GPU work,
- retry duplicates generation.

## MVP definition

A clean clone can launch one text engine, stream through FastAPI, run a fixed benchmark, persist per-request timing, compute TTFT/p95/throughput, run a versioned quality eval, expose OTel/Prometheus, and demonstrate one normal and one failure case.

## Approval questions

- What GPU hardware is realistically available?
- Which model family has a compatible speculation pairing/method?
- What quality metric fits the text workload?
- Is the multimodal model autoregressive over discrete visual tokens or another family? Do not force text-decoding assumptions onto diffusion.
- Which workload mix owns the headline numbers?
- Are baseline and optimized measurements on identical hardware?
- What pricing source/date defines GPU cost?
- Which requested technologies are core versus comparison labs?
