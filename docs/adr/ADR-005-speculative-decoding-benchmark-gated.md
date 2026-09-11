# ADR 005: Gate speculation on measured benefit and quality

Status: accepted; tested n-gram profile rejected.

## Decision

Keep speculation disabled by default. Enable an engine-supported proposal method only after a declared workload comparison improves useful serving performance and passes independent correctness. Record draft/accepted counts and their interval; acceptance alone is insufficient.

## Evidence and consequences

Three-token n-gram proposals at concurrency 16 reduced development throughput from 40.29 to 20.59 requests/s and increased p95 from 0.533 to 1.387 seconds. Acceptance across two recordings including warmup was 30.04%. The correctness suite failed. Raw failures remain in `ngram-load-c16-01` and `quality-ngram-01`; [speculative decoding](../speculative-decoding.md) explains accounting.

Always enabling speculation would retain measured overhead at this envelope. A learned draft model could change cost and acceptance, but adds memory and another model identity without an accepted result here. FinServe delegates verification to the engine and has no adaptive proposal-length controller. The untrained JAX visual decoder has no verified speculative path.
