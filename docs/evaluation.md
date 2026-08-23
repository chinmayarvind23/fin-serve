# Evaluation

## Layers

1. Functional correctness: schema, streaming, model selection, cancellation, errors.
2. Output quality: catches quantization/engine/decoding/artifact regressions.
3. Performance: latency, throughput, cost.
4. Reliability/security: overload and failure behavior.

## Golden set

Stable case IDs contain input, task, reference expectation, hard checks, soft scoring config, modality, and length bucket.

## Failure-oriented cases

Malformed inputs, max-context boundary, unsupported modality, huge image, timeout before first token, cancellation mid-stream, server restart, saturation, Redis unavailable, RDS unavailable while serving, telemetry unavailable, artifact checksum mismatch, quality regression, latency regression, GPU OOM/recovery, duplicate Airflow/deploy retry.

## Gate logic

Examples:

```text
FAIL if p95 exceeds allowed limit
FAIL if TTFT regression exceeds allowed delta
FAIL if throughput falls below allowed floor
FAIL if success rate falls below floor
FAIL if quality parity falls below floor
```

All values are versioned configuration.

## Semantic judges

Only when deterministic/task metrics are insufficient. Use fixed rubric/version, blind ordering when possible, calibration reviewed by a human, and store judge output. Never use a judge for deterministic latency/correctness facts.

## CI tiers

PR CI: unit/contract/static checks and small eval.

GPU integration CI: engine startup, small performance/quality/speculation/multimodal smoke.

Release: 6,000+ request suite, load/failure, full quality, cost, rollback drill.
