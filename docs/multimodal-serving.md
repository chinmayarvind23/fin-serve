# Multimodal Serving

## Principle

Multimodal model components can have different batching, memory, compute, input-size, output, and accelerator behavior.

## Path 1: JAX/Flax autoregressive visual-token reference

Use it to understand visual-token generation, JAX compilation, fixed/dynamic shapes, batch behavior, and proposal/verification feasibility.

## Path 2: production-compatible multimodal serving

Use engine support for the selected VLM/multimodal model. Reuse FinServe request validation, routing, GPU placement, telemetry, eval, deployment, and rollback.

## Stage disaggregation

Conceptually:

`input -> encode -> autoregressive language stage -> visual/audio generation -> postprocess`

If a monolithic replica shows stage imbalance, benchmark separate pools with per-stage batching and GPU allocation. Keep it only if job completion/utilization improves after cross-stage transfer cost.

## Metrics

Job completion time, per-stage queue/execution time, resolution/output size, stage utilization, transfer time, task quality, cancellation, artifact integrity.

## Async job semantics

Long image jobs may use `POST /v1/image-jobs -> 202 + job_id`; Redis may keep short-lived status, while durable evidence remains in RDS/S3.
