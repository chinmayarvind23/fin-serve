# FinServe

FinServe is a distributed text and multimodal inference platform built to study the engineering tradeoffs that determine production model-serving performance: request scheduling, continuous batching, KV-cache efficiency, speculative decoding, GPU placement, multimodal stage separation, autoscaling, quality regression, observability, and rollback.

## Performance contract

| Metric                           |      Baseline | Optimized / required result |
| -------------------------------- | ------------: | --------------------------: |
| Sustained request throughput     | 38 requests/s |               94 requests/s |
| Token throughput                 |          1.0x |               2.4x baseline |
| Median TTFT                      |        690 ms |                      295 ms |
| p95 end-to-end latency           |         3.8 s |                       1.9 s |
| GPU cost per 1M generated tokens |      baseline |                   37% lower |
| Output-quality parity            |     reference |                       99.2% |
| Successful requests under load   |           n/a |                      99.95% |
| Average GPU utilization          |           n/a |                         81% |
| Regression rollback              |           n/a |                 within 94 s |
| Benchmark requests               |           n/a |                      6,000+ |

`docs/results.md` links the exact manifests, hardware, model versions, workload mix, traces, and cost inputs used to support this contract.

## What FinServe serves

1. **Text inference**: PyTorch reference path, vLLM primary optimized engine, SGLang comparison path, continuous batching, KV-cache-aware routing, speculative decoding, streaming OpenAI-compatible endpoints.
2. **Multimodal inference**: JAX/Flax autoregressive visual-token reference path, engine-compatible multimodal workers, stage-aware scheduling, and a research path for proposal/verification over discrete visual tokens when the model formulation supports it.

## Architecture

```text
Browser / SDK / load generator
          |
          v
TypeScript + Bun edge gateway
          |
          v
FastAPI OpenAI-compatible ingress
          |
          +---------------------------+
          |                           |
          | synchronous text stream   | async long-form visual job
          v                           v
   Ray Serve router                Redis job state
          |                           |
          |                           v
          |                     Ray Serve worker
          |
          +--> GPU / cache / load-aware replica routing
          +--> vLLM engine pool
          +--> SGLang comparison pool
          +--> JAX/Flax multimodal pool
          +--> optional disaggregated prefill/decode or multimodal stages
          |
          v
      streamed result

Control and evidence plane
RDS PostgreSQL     model/deployment/benchmark metadata
S3                 artifacts, results, traces, reports
MLflow             optimization/eval/promotion evidence
Redis               rate limits, bounded cache, async job state
Airflow             fetch/verify/optimize/build/evaluate/deploy workflow
Langfuse            inference/eval trace exploration
OpenTelemetry       distributed traces
Prometheus          serving/GPU/application metrics
Grafana/CloudWatch  dashboards and alerts
EKS + KubeRay       Ray clusters and GPU workers
Terraform           AWS infrastructure
Vercel              benchmark/deployment explorer
```

### Request-path rule

Redis is **not** placed between a streaming text request and the inference engine as a mandatory queue. Ray Serve and the inference engine already own request admission, replica queues, and continuous batching. Redis is used for rate limiting, bounded caching, short-lived routing metadata, and explicit asynchronous visual jobs.

## Protocol roles

```text
REST / OpenAI HTTP   inference and state-changing control commands
gRPC                 selected internal typed/streaming boundaries
GraphQL              read-only benchmark/deployment explorer
```

GraphQL stays off the latency-critical inference path.

## Repository map

```text
apps/api/                  FastAPI ingress/control
apps/web/                  TypeScript/Bun dashboard
src/finserve/contracts/    typed schemas
src/finserve/gateway/      auth/rate-limit/admission
src/finserve/scheduler/    routing/GPU policy
src/finserve/engines/      PyTorch/vLLM/SGLang adapters
src/finserve/multimodal/   JAX/Flax and stage-aware serving
src/finserve/benchmark/    workloads/metrics/cost
src/finserve/evaluation/   quality-parity and failure evals
src/finserve/registry/     model/deployment evidence
src/finserve/telemetry/    OTel/Langfuse/Prometheus
src/finserve/reliability/  retries/health/rollback
pipelines/airflow_dags/
infra/{docker,kubernetes,terraform}/
tests/ evals/ benchmarks/ monitoring/
```

## Development philosophy

```text
baseline -> benchmark -> one optimization -> measure -> quality gate -> keep/reject
```

The repository does not keep an optimization because it sounds advanced.

## MVP

```text
load generator -> FastAPI -> one vLLM model -> stream -> benchmark recorder -> OTel/Prometheus -> quality check
```

The MVP proves the measurement loop before distributed serving, multimodal workers, Airflow, KubeRay, or advanced routing.

## Key docs

- [PRD](PRD.md)
- [System design](docs/system-design.md)
- [Architecture alternatives](docs/architecture-alternatives.md)
- [HLD](docs/HLD.md)
- [LLD](docs/LLD.md)
- [Inference fundamentals](docs/inference-fundamentals.md)
- [Scheduling and batching](docs/scheduling-and-batching.md)
- [Speculative decoding](docs/speculative-decoding.md)
- [Multimodal serving](docs/multimodal-serving.md)
- [Benchmark methodology](docs/benchmark-methodology.md)
- [Evaluation](docs/evaluation.md)
- [Observability](docs/observability.md)
- [Performance](docs/performance.md)
- [Security](docs/security.md)
- [Failure modes](docs/failure-modes.md)
- [Deployment](docs/deployment.md)
- [Interview prep](docs/interview-prep.md)

## Non-goals

- training a frontier base model,
- pretending every workload benefits from speculation,
- routing streaming text through an unnecessary external queue,
- using an LLM judge for deterministic performance facts.
