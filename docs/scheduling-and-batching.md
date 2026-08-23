# Scheduling and Batching

## Admission

Validate size, auth/quota, model/modality, concurrency, and bounded queue capacity before routing. Explicit overload is better than infinite latency.

## Routing baseline

Start simple/load-aware. Do not start with a complicated learned router.

## Prefix/cache affinity

Prefer a replica with reusable prefix state only while load imbalance remains acceptable. Cache affinity that overloads one replica is a bad optimization.

## Length/SLO awareness

Later experiments can separate interactive short work from long/batch work if the held-out workload shows benefit.

## Adaptive policy

Feature vector can include ongoing requests, queue depth, GPU memory, cache affinity, request length buckets, and SLO class. Score eligible replicas and log features/reason codes for every decision.

## Backpressure

Use max queued/ongoing requests, per-user concurrency, size limits, timeouts. Saturation returns an overload error rather than silently buffering forever.

## Autoscaling chain

`traffic -> Serve replica demand -> Ray resource demand -> Pod demand -> EC2 GPU node demand`

Each layer has a different signal and reaction time.

## GPU placement

Use accelerator labels and placement groups for single-GPU models, multi-GPU tensor parallel groups, and modality-specific pools. Actual GPU types depend on AWS quota/budget.
