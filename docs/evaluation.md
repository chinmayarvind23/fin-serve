# Evaluation

FinServe separates transport correctness, task correctness, serving agreement and performance. A configuration may complete every request faster while returning wrong answers. The release gate checks these properties independently and retains rejected evidence.

## Deterministic task quality

The frozen 32-case suite uses `exact-or-typed-json-v1`. Exact answers strip leading/trailing whitespace, then compare case-sensitive strings. JSON answers must be objects: key order and insignificant JSON whitespace may differ, but field types, extra fields, duplicate keys and nonfinite values are not accepted. Markdown fences are not stripped. Consequently, `No` and `no`, or exact strings `0.20` and `0.2`, differ under this declared grader.

For every case, the evaluator records reference correctness, candidate correctness and reference/candidate parity. Missing or invalid responses fail. Each incorrect candidate answer is a hard failure, so reducing a numeric accuracy threshold cannot average away wrong deterministic answers. Default minimum parity is 0.992 and minimum accuracy is 1.0. At 32 cases, even one disagreement misses that parity threshold.

The original sustained compiled candidate scored 10/32 correct and 24/32 parity, failing the gate. Invalid reference JSON also fails self-parity; identical malformed output is not a valid structured answer. The failures remain in the recorded evidence. The three-case development smoke has a separate suite hash and narrower scope.

Chat collection additionally freezes API mode, system prompt and template identity. Both benchmark and quality evidence must use the same declared request mapping. A corrected prompt, stronger model or constrained decoder is a new candidate experiment; it cannot rewrite an earlier result or weaken the frozen grader after failure.

## Serving agreement and numerical checks

Load-output parity compares equally sized, nonempty sequences byte for byte. It does not use a correctness oracle. The sustained pair matched 2,479 of 3,072 outputs while failing the independent task suite.

The PyTorch reference decoder has separate cache-versus-full-prefix, batching and sampling checks. They verify implementation mechanics with reference weights, not pretrained language quality. The JAX/Flax reference generator and pretrained image-understanding route likewise have distinct tests and populations. The [results](results.md) keep the failed color suite separate from three successful chart probes.

## Performance and lifecycle

The benchmark retains scheduled, offered, completed and failed work, authoritative generated-token counts, warmup boundaries and raw timestamps. Its immutable configuration controls the arrival process, concurrency, timeouts and workload identity. The gate recomputes summaries from raw records and checks identity, throughput, latency, transport success and quality before activation.

Lifecycle tests cover revision/generation fencing, retries, route activation and recovery. The actual local rollback drill switches already-running fixture endpoints; cloud node recovery and GPU cold start require their own evidence. No local fixture result establishes production availability or billed GPU cost.

## Source verification and remaining scope

CI runs Python static checks, unit/contract/integration tests, aggregate coverage and semantic mutation checks, plus Bun HTTP and DOM-emulated tests and builds. Native GPU experiments run separately with explicit manifests. There is no continuous GPU performance job or calibrated semantic-judge service in the current CI workflow.

Browser visual review, hosted telemetry, AWS/Hugging Face operation and extended quality populations remain separately tracked work. See [quality gates](quality-gates.md), [benchmark populations](finance-benchmark-suite.md) and [reproducibility](reproducibility.md) for the boundaries of each claim.
