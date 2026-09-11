# ADR008: Airflow stays outside inference

Status: three-task evidence DAG implemented and locally tested; producer DAG expansion pending.

Airflow 3.3.0 coordinates registration, dual-gate evaluation and a verified deployment callback. Admission, routing and token generation never wait for an Airflow task. Tasks pass immutable job IDs and reconstruct trusted state from the registry; database state survives retries.

The current DAG consumes existing build/run/quality artifacts. Model verification and Docker build APIs run locally but are not yet DAG tasks. Runtime collection, controller acknowledgment and probation orchestration remain work. Actual local `dag.test()` execution validates Airflow's public API with fixture evidence, not an AWS rollout or a healthy model release.

Keeping Airflow off the token path avoids coupling request latency to orchestration availability. The tradeoff is a separate fenced control-plane integration. [Airflow pipeline](../airflow-pipeline.md) records its implemented stages and remaining work.
