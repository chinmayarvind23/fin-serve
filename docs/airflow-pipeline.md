# Airflow lifecycle

Airflow coordinates evidence work outside the request path. The implemented Airflow 3.3.0 DAG has three tasks:

```text
register_evidence → evaluate_gates → deploy_verified_candidate
```

`pipelines/airflow_dags/finserve_lifecycle.py` calls `registry.pipeline`. The DAG has no schedule, permits one active run, and passes only an immutable job ID between tasks. Actual local Airflow execution is tested with controlled evidence and a deployment fixture. That test does not establish a cloud rollout.

## Inputs and gate

The managed producer registration entry point is `pipeline.register_produced_stage(plan_stage_id)`.
Before collecting either cohort, `freeze_produced_release` persists a `ProducedReleasePlan`
containing both performance and quality specifications, four distinct collection stage IDs,
the target deployment generation and the promotion policy. Registration reconstructs completed
managed receipts, requires the same launch for each cohort's performance and quality, and
rejects collection attempts or retained observations predating the plan. It constructs canonical
gate inputs from raw outputs; callers do not supply a quality result file. Replay performs no
inference. The original directory-based registration entry point remains for existing evidence.

The managed entry point is tested from fixture launch and collection through canonical gate
rejection. It is not yet wired into the full Airflow producer DAG, and does not acknowledge
traffic activation or complete probation.

The trusted worker configures `FINSERVE_PIPELINE_REQUEST`, `FINSERVE_REGISTRY_URL` and `FINSERVE_ARTIFACT_ROOT`. The request identifies existing baseline/candidate run directories, raw quality outputs, a frozen suite, immutable revisions, policy and both canonical serving profiles. Database and artifact references must retain their original namespace across retries.

Registration verifies raw run evidence and persists the lifecycle specification. Canonical jobs record their required gate mode before publishing profile inputs; a missing profile after a crash cannot fall back to a legacy drill gate. Evaluation checks profile/run identities and recomputes performance and quality. Chat cohorts also bind API, system instruction and template digest between performance and quality. Rejection or invalid evidence prevents deployment.

The same gate is available to a trusted CI worker:

```sh
uv run --no-sync python -m finserve.registry.release_gate --job-id JOB_ID --output /external/evidence/new-outcome.json
```

It exits 0 for approval, 2 for rejection and 3 for invalid evidence. The manual `performance-gate.yml` workflow requires a trusted Linux self-hosted runner with registry/artifact access. It does not run GPU work or deploy from pull requests. Adding the workflow is not proof of an executed CI run.

## Activation and recovery

Deployment loads the server-configured `FINSERVE_DEPLOYMENT_ADAPTER` factory and recomputes the gate before invoking it. Lifecycle leases, immutable request identities and adapter idempotency govern retries. Uncertain external action is reconciled through exact-revision health rather than blindly repeated. A decision supplied by an arbitrary caller is not an authorization capability.

The warm adapter switches traffic between already running registered backends and probes the active route. It does not pull images or start pods. After lifecycle promotion, `release_activation.acknowledge_release` recomputes the canonical gate, verifies the recorded lifecycle decision, probes current traffic and records the exact activation in the rollback controller. It requires the persisted lifecycle route-action receipt and holds the route write lock during controller acknowledgment. Repeating it does not increment the generation again. The previous known-good revision remains unchanged. After a complete healthy monitor window, `release_activation.complete_probation` rechecks approval and current traffic before advancing known-good with retained observation evidence. Wiring these calls into the full producer DAG remains pending. [ADR014](adr/ADR-014-rollback-known-good-revision.md) describes restoration rules.

## Producer integration still to complete

Model fetch/verification and committed-source runtime build APIs now run locally. Actual image preflight, GPU measurement and canonical rejection evidence are retained. These producer steps are not yet Airflow tasks: the current DAG consumes their completed artifacts.

The local producer implements durable model/image/quality receipts, managed runtime startup and cleanup, and frozen performance collection bound to a runtime start. Remaining work is to connect these stages into the full Airflow producer DAG with image publication where required, controller acknowledgment, probation and rollback wiring, then exercise that complete path. The 32-case release suite remains frozen. A three-case development smoke cannot substitute for it, and failed quality cannot be waived to demonstrate activation.
