# System Design

## Design question

Design an inference platform that serves streaming text and multimodal workloads, compares serving configurations reproducibly, scales GPU work safely, preserves output quality, and rolls back regressions quickly.

## Non-functional conflicts

```text
low TTFT <-> large batches
high utilization <-> queueing headroom
warm capacity <-> GPU cost
cache affinity <-> load balance
speculation speedup <-> proposal overhead
stage separation <-> transfer/orchestration cost
```

## Capacity intuition

`request_throughput = successful_completed_requests / measured_wall_time`

`token_throughput = generated_tokens / measured_wall_time`

`gpu_cost_per_1m = gpu_hourly_cost * gpu_hours / generated_tokens * 1_000_000`

Queueing becomes extremely sensitive when offered load approaches effective service capacity. Simple queueing math is intuition only because token lengths, batching, cache state, and scheduling violate simple assumptions, so the real system is benchmarked.

## Architecture candidates

### A. Single vLLM server

Best first benchmark. Low integration overhead. Weak distributed-platform signal. Selected for MVP/baseline.

### B. FastAPI + Redis queue for every request

Explicit async semantics, but poor fit for low-TTFT streaming and duplicates Ray/engine scheduling. Rejected for synchronous text.

### C. Ray Serve over engine replicas

Supports distributed replica routing, custom policies, autoscaling, placement, and KubeRay. Selected after single-node evidence is stable.

### D. Fully disaggregated multimodal stages

Powerful but premature. Stage-disaggregation is a later measured experiment inspired by modern multimodal serving work.

## Selected architecture

```text
                           +--------------------+
                           | Vercel dashboard   |
                           | TypeScript + Bun   |
                           +---------+----------+
                                     |
                               GraphQL reads
                                     |
+---------+   HTTPS/OpenAI   +-------v---------+
| clients | ---------------->| FastAPI ingress |
+---------+                  +-------+---------+
                                     |
                               auth/admission
                                     |
                              +------v-------+
                              | Ray Serve    |
                              +---+-------+--+
                                  |       |
                          route by load/cache/GPU/modality
                     +------------+        +----------------+
                     |                                     |
                 +---v----+                           +----v----+
                 | vLLM   |                           | JAX/Flax|
                 | pool   |                           | pool    |
                 +---+----+                           +----+----+
                     |                                     |
                 +---v----+                           +----v----+
                 | SGLang |                           | VLM/etc |
                 | compare|                           | pool    |
                 +--------+                           +---------+

Ephemeral: Redis
Evidence/control: RDS + S3 + MLflow + Airflow
Observability: OTel + Prometheus + Langfuse + Grafana/CloudWatch
Infrastructure: EKS + KubeRay + Terraform
```

## Online text flow

1. client opens stream,
2. gateway auth/validation,
3. admission checks request size/concurrency,
4. model routing,
5. replica routing,
6. engine scheduler admission,
7. continuous batching,
8. stream tokens,
9. telemetry emitted asynchronously,
10. sampled metadata recorded.

No Airflow task, RDS query, Elasticsearch query, or Redis job queue is required to emit the first token.

## Adaptive routing

Start with a simple/default load-aware router. Later compare a score such as:

```text
score =
  a*normalized_load
+ b*queue_penalty
+ c*expected_length_penalty
- d*cache_affinity
+ e*health_penalty
```

Coefficients are configuration. Compare policies on held-out traffic traces.

## Continuous batching

```text
step 1: [A B C D]
step 2: [A B C]     D finishes
step 3: [A B C E]   E joins
```

This raises utilization with variable output lengths, but excessive active work can pressure KV memory and worsen latency.

## Speculative decoding

If `D` proposal tokens are generated, `A` accepted, and `V` verification steps occur:

```text
draft_acceptance_rate = A / D
mean_acceptance_length = 1 + A / V
```

Speculation is kept only when saved verifier work exceeds proposer/coordination overhead for the workload bucket.

## Multimodal

Two paths:

1. JAX/Flax autoregressive visual-token lab for first principles and compile/batch/speculation-feasibility research.
2. Engine-supported production-compatible multimodal/VLM path.

Later stage-disaggregation can separate encode/language/visual generation if measured stage imbalance justifies transfer cost.

## Storage

- RDS: authoritative metadata/deployment/benchmark state.
- S3: large artifacts/results/models/traces.
- MLflow: optimization/eval/promotion evidence.
- Redis: ephemeral counters/cache/job state only.

## Autoscaling

Three levels:

```text
Serve replicas -> Ray worker Pods -> EKS/EC2 GPU nodes
```

Cold start can include node provisioning, image pull, Ray startup, model load, compile, GPU memory profile. Interactive pools keep a warm floor when required.

## Deployment

```text
Airflow candidate DAG
 -> verify artifact
 -> optimize/build
 -> offline eval
 -> performance gate
 -> deploy canary
 -> smoke/load
 -> promote
 -> regression? restore known-good revision
```

## Excluded from hot path unless earned

Kafka, Spark, Elasticsearch, RDS-per-request reads, Airflow online orchestration, a separate microservice per engine feature, and a Redis queue for streaming text.
