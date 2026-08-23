# Inference Fundamentals

## Autoregressive decode

`P(x_1,...,x_T) = product_t P(x_t | x_<t)`

Production engines avoid recomputing the whole prefix every decode step with a KV cache.

## KV cache

Store attention key/value tensors for prior tokens and reuse them. Memory scales roughly with layers, sequence length, KV heads, head dimension, bytes per element, and a factor of 2 for keys/values.

## Prefill versus decode

Prefill processes the prompt in larger compute-heavy work. Decode generates token-by-token and is often KV/memory-bandwidth sensitive. Long-prompt/short-output and short-prompt/long-output workloads must be benchmarked separately.

## TTFT

`first_streamed_token - request_accepted`

Includes queueing, routing, prefill, and stream overhead.

## ITL

`token_time_i - token_time_(i-1)`

## End-to-end latency

`client_complete - client_send`

## Throughput

`successful_requests / wall_time` and `generated_tokens / wall_time`.

Never compare throughput without prompt/output distributions.

## Continuous batching

Completed sequences are replaced by waiting work while longer sequences continue, increasing occupancy under variable generation lengths.

## Paged KV memory

Block/page-based KV management reduces fragmentation. Interview mental model: logical sequence KV can map onto non-contiguous physical blocks, similar to virtual-memory paging.

## GPU utilization

Document source, sampling interval, per-GPU samples, warmup exclusion, and aggregation. 100% utilization is not automatically better if tail latency collapses.
