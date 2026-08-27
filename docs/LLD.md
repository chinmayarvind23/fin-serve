# Low-Level Design

## Package map

```text
src/finserve/
  contracts/{inference,routing,benchmark,deployment,evaluation}.py
  gateway/{auth,rate_limit,admission}.py
  scheduler/{policy,features,capacity}.py
  engines/{base,pytorch_reference,vllm_adapter,sglang_adapter}.py
  multimodal/{jax_generator,router,stage_graph}.py
  cache/{redis_client,keys}.py
  benchmark/{workload,runner,recorder,metrics,cost}.py
  evaluation/{parity,graders,golden}.py
  registry/{models,deployments,benchmarks}.py
  telemetry/{tracing,metrics,logging}.py
  reliability/{retries,circuit_breaker,health,rollback}.py
```

## Core contracts

`InferenceRequest`: request ID, model, modality, prompt/media reference, output limit, sampling, stream, timeout, SLO class.

`ReplicaSnapshot`: replica/model/engine/GPU, ongoing and queued requests, memory ratio, health, cache-affinity score.

`RoutingDecision`: request ID, replica ID, policy version, feature snapshot, reason codes.

`BenchmarkManifest`: git SHA, image digest, model/tokenizer revisions, engine/version, GPU, workload hash, seed, warmup, request count, arrival mode, concurrency, pricing snapshot.

## Request states

```text
RECEIVED -> VALIDATED -> ADMITTED -> ROUTED -> ENGINE_QUEUED -> PREFILL -> DECODING -> STREAMING -> COMPLETED
terminal: REJECTED / CANCELLED / TIMED_OUT / ENGINE_FAILED / OVERLOADED
```

## Cancellation

Client disconnect propagates to ingress/Ray/engine abort where supported and gives back KV/cache resources.

## Retry

Transparent retry is allowed only before externally visible output and only for retryable failures. Once tokens have streamed, return a structured stream error rather than duplicating/diverging output.

## Rollback record

Store deployment, bad revision, restored revision, detected time, rollback start, healthy time, detector, reason.

`rollback_seconds = healthy_at - detected_at`
