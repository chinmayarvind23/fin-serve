# ADR008: Airflow stays outside inference

Status: evidence DAG and local producer DAG implemented and tested with scoped fixtures.

Airflow 3.3.0 coordinates registration, dual-gate evaluation and a verified deployment callback. Admission, routing and token generation never wait for an Airflow task. Tasks pass immutable job IDs and reconstruct trusted state from the registry; database state survives retries.

The evidence DAG consumes existing artifacts. A separate 18-task producer DAG connects model
fetch, build, paired collection, canonical gates, initial route preparation, activation,
controller acknowledgment and probation. All-done cleanup is followed by an all-success leaf
that also depends on probation, preserving failed release status. Actual `dag.test()` runs
verify scheduler success and failed-gate cleanup using synthetic callbacks. Integration tests
exercise real lifecycle code with fixture transports. Neither establishes a live GPU/cloud
rollout. Updates reuse the existing stable baseline's completed launch receipt and collect new
evidence without changing its identity or starting another baseline container. Initial requests
without a baseline reference still require an unused deployment ID at generation zero.

Keeping Airflow off the token path avoids coupling request latency to orchestration availability. The tradeoff is a separate fenced control-plane integration. [Airflow pipeline](../airflow-pipeline.md) records its implemented stages and remaining work.
