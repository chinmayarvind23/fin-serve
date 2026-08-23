# ADR 004 redis not on streaming hot path

## Decision

Ray/engine queues handle synchronous text; Redis handles rate limits, bounded cache, ephemeral state, explicit async jobs.

## Evidence

Validate the decision with the benchmark, eval, telemetry, or failure test appropriate to the component.
