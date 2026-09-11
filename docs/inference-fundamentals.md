# Inference fundamentals

An autoregressive model factors sequence probability as `P(x_1, ..., x_T) = product_t P(x_t | x_<t)`. Each generated token becomes input to a later step. Serving performance depends on this sequential work and the surrounding admission, transport and cleanup.

## Attention and KV cache

[pytorch_reference.py](../src/finserve/engines/pytorch_reference.py) contains a seeded tiny decoder with causal attention. Its cached path stores previous keys and values; the uncached path recomputes the prefix. Deterministic tests require matching generated token IDs. Random weights provide an educational reference, without a language-quality claim.

Approximate KV storage is `2 * layers * tokens * KV_heads * head_dimension * bytes_per_element` per sequence, before engine and allocator overhead. The factor two covers keys and values. Grouped-query attention changes KV-head count, so query heads alone are insufficient for this estimate.

The reference uses simple tensors. Production engines own allocation, batching and token scheduling. Paged KV memory maps logical sequence blocks onto physical storage to reduce fragmentation; FinServe observes engine gauges rather than implementing that allocator.

## Prefill, decode and batching

Prefill processes the prompt; decode advances generated sequences. Long-input/short-output requests exercise a different balance from short-input/long-output requests. The frozen routing workload covers all four combinations of short/long input and 16/128-token budgets. A budget is an upper bound; authoritative output counts determine actual work.

Continuous batching lets an engine replace completed sequences while others continue. Proxy admission controls how many requests reach the engine. Native sequence and token limits govern its internal batches; a proxy capacity is not a GPU batch shape.

## Measurement boundaries

| Quantity | Recorded boundary |
| --- | --- |
| Client TTFT | First nonempty content received minus recorded client-send boundary |
| Server TTFT | First content produced minus trusted ingress receipt, where available |
| End-to-end latency | Client completion minus client send |
| Successful-request throughput | Successful measured requests divided by measured interval |
| Generated-token throughput | Authoritative generated tokens divided by that interval |
| Success rate | Successful measured requests divided by all offered logical requests, including local rejection |

An SSE content delta can contain several tokens or no tokens. HTTP adapters emit zero-count content frames and take the final count from engine usage. Counting frames as tokens corrupts throughput and inter-token latency. These adapters expose content-arrival timing; genuine token-level ITL requires a source that provides token timestamps.

Percentiles state their population and interpolation rule. Failed requests remain in the availability denominator. Successful-request p95 alone can appear favorable when most work was rejected. Missing TTFT and GPU observations remain missing.

[benchmark/gpu.py](../src/finserve/benchmark/gpu.py) identifies physical devices by UUID. Integration holds samples over bounded gaps and reports coverage; collection failures break coverage. Two processes sharing one GPU count as one physical device. High utilization matters alongside useful completed work, latency and correctness.
