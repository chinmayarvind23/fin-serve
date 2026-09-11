# Registry and release control

The registry stores immutable model, run, quality and revision references in SQL. Artifact bytes live in a content-addressed store and are checked for namespace, length and SHA256 on read. The release gate reconstructs metrics and quality from those bytes before persisting its decision.

`model_assets.py` freezes and verifies a model snapshot. `runtime_build.py` builds an engine image from committed source and records actual Docker identities. `engine_entrypoint.py` rehashes the mounted snapshot and checks the serving profile before starting vLLM. `producer_stages.py` journals attempts; `producer_tasks.py` connects download, build and quality collection to verified receipts. A running attempt requires reconciliation before retry. A completed quality collection can still fail the release gate.

`pipeline.py` and `lifecycle.py` share gate and deployment behavior with the Airflow DAG. `explorer.py` exposes bounded, authenticated, read-only GraphQL; it cannot mutate a deployment. `annotations.py` recomputes GPU and quality details for the evidence UI.

Use the commands in [deployment](../../../docs/deployment.md), [evaluation](../../../docs/evaluation.md) and [API documentation](../../../docs/api-contracts.md). The [low-level design](../../../docs/LLD.md) defines ownership and persistence boundaries. Full artifact production through cloud activation remains in progress; the existing Airflow evidence DAG is not proof of that complete path.
