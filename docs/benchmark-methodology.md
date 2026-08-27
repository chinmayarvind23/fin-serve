# Benchmark Methodology

## Rules

1. Freeze workload manifest before headline comparison.
2. Record baseline before tuning.
3. Use the same quality rules.
4. Exclude warmup consistently.
5. Preserve failed/timed-out requests.
6. Never count dropped requests as successful throughput.
7. Record hardware/software versions.
8. Repeat runs and keep raw records.
9. Separate policy-tuning workloads from final workloads.
10. Publish failed optimizations and regressions.

## Workloads

- W1 interactive short: short prompt, short/medium output.
- W2 decode-heavy: short prompt, long output.
- W3 prefill-heavy: long prompt, short output.
- W4 frozen mixed production-like distribution for headline comparison.
- W5 repeated-prefix for cache/routing.
- W6 failure/adversarial: excess input, timeout, cancellation, burst overload, malformed input, engine/node failure.
- W7 multimodal: size/resolution/output buckets.

## 6,000+ request rule

Count real logical requests across declared configurations. Streaming chunks are not requests.

## Arrival modes

Closed-loop concurrency and open-loop arrivals are both supported and explicitly labeled because they answer different queueing questions.

## Timestamps

Record client send, server received, route, engine admit, prefill if available, first token, last token, client complete.

## Metrics

`TTFT = first_token - server_received`

`E2E = client_complete - client_send`

`token_throughput = sum(generated_tokens) / measured_seconds`

`success_rate = successful_requests / accepted_requests`

`gpu_cost_per_1m = sum(instance_hour_cost * instance_hours) * 1_000_000 / generated_tokens`

Quality parity has a versioned formula matching the selected task. Do not invent a generic 99.2% formula after seeing results.

## Ablations

```text
B0 reference/simple server
B1 vLLM default
B2 batching/engine tuning
B3 cache-aware routing
B4 speculative decoding
B5 Ray Serve distributed routing
B6 autoscaling/heterogeneous placement
B7 multimodal stage experiment where applicable
```

This prevents attributing the whole gain to the last technology added.

## Artifact layout

```text
benchmarks/results/<run_id>/
  manifest.json
  requests.parquet
  gpu_metrics.parquet
  service_metrics.json
  eval_results.json
  cost.json
  summary.json
  notes.md
```
