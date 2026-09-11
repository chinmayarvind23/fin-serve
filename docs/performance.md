# Performance engineering

No optimized release has passed FinServe's required correctness gate. Measurements retain faster candidates that failed quality and routing cohorts that rejected most offered requests. [Recorded results](results.md) contains the central result table.

## Current evidence

The sustained eager/compiled comparison used the same pinned Qwen2.5-0.5B model, one RTX 4070 Laptop GPU, concurrency 16 and 3,072 measured requests per configuration. Successful-request throughput rose from 12.99 to 34.59 requests/s and token throughput from 373.52 to 987.54 tokens/s. Client median TTFT worsened from 112.95 to 159.45 ms; successful-request end-to-end p95 fell from 2.131 to 0.804 seconds. The separate quality gate failed. One ordered pair leaves thermal and workstation confounds.

Three-token n-gram speculation lost throughput and failed quality at the tested envelope. The preferred profile leaves it disabled. The VLM CPU preprocessing hop added measured round-trip cost while preserving paired output, but its uniform-color semantic gate failed.

The first two-engine routing comparison used two actual vLLM processes on one physical GPU. Four 64-request cohorts completed only 21, 19, 24 and 24 requests. All other requests failed before content; Ray logs identify admission capacity rejection. These results provide no adaptive policy gain. Scale-down succeeded, then restart preflight failed. See [scheduling and batching](scheduling-and-batching.md) for the separate admission-wait correction.

The original targets, including 94 requests/s, improved TTFT, 99.2% quality parity, 81% mean GPU utilization and 37% lower GPU cost, remain unachieved or unmeasured. Local observations contain no cloud invoice or production availability window.

## Experiment discipline

Freeze revisions, inputs, seed/order, output budgets, concurrency, warmup and acceptance criteria before measurement. Record actual source and launch identity. A copied profile is a declaration; model discovery, process identity and observed hardware supply separate evidence. A later container build cannot retrospectively identify an unrecorded native-run image digest.

Retain every offered request and failed attempt. Compare successful throughput alongside availability and actual generated lengths. Report paired output agreement over a declared denominator and task correctness separately. Development tuning, held-out comparisons and functional smoke tests answer different questions.

GPU means require physical UUIDs, bounded sample gaps, declared warmup exclusion and sufficient coverage. Clock drift can invalidate wall-clock integration. Drain the observer before calculating summaries so a final in-flight sample cannot alter the saved raw population afterward.

## Diagnose before tuning

| Observation | Evidence to inspect |
| --- | --- |
| Admission rejection | Pending requests, leases, reflected leases, native-gauge age and health |
| Slow first content | Upload, queue, routing, prompt length, prefill and transport boundaries |
| Slow completion | Generated length, native batch load, decode rate and cleanup |
| Memory pressure | Physical VRAM, engine KV, active sequences and context budgets |
| Poor quality | Exact input, template, model identity, grader and independent correctness cases |

Increasing queue depth can turn rejection into latency. Dropping conservative telemetry can invent capacity. The admission-wait correction preserves the request deadline and waits for eligible capacity; it needs a separately identified GPU rerun before any performance conclusion.

Cold-start measurements separate process launch, image pull, weights, compilation, memory profiling, readiness and cloud provisioning. Warm-route rollback does not measure those stages. Prefix-cache benefits require a declared prefix distribution. Response caching must be declared separately. Quantization or a different model requires its own correctness comparison.
