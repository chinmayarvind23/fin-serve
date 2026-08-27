# Performance Engineering

## Method

`measure -> classify bottleneck -> change one variable -> benchmark -> quality gate -> keep/revert`

## Bottleneck classes

Queue-bound: scale/routing/admission.

GPU underutilized: batching/concurrency/ingress/preprocessing.

KV-memory bound: token limits/cache config/active sequences/more memory.

Decode-latency bound: speculation/faster engine/quality-safe quantization/hardware.

Prefill-heavy: prefix cache/chunking/disaggregation if measured.

## Headline arithmetic

`94 / 38 ~= 2.47x` request throughput.

The separate `2.4x` token-throughput must come from token counts, not be inferred.

Median TTFT reduction is about `57.2%` from 690 to 295 ms.

p95 reduction is `50%` from 3.8 to 1.9 s.

## Utilization

100% GPU utilization can be bad if queueing violates latency. Optimize useful throughput/cost subject to latency/success/quality gates.

## Cold starts

Measure node provision, image pull, Ray start, model download/load, compile, memory profile, readiness separately.

## Caching

Prefix cache is workload-specific. Response cache must be declared and must not contaminate inference benchmarks. Redis is not GPU KV cache.

## Quantization

Compare memory, throughput, latency, and quality. Reject quality regressions.

## Engine comparison

Use same model/workload/hardware where possible. Do not mix different models/hardware and call it an engine-only result.
