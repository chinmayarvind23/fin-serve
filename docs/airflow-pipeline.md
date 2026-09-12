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

The separate `finserve_producer_lifecycle` DAG now connects collection, canonical registration,
evaluation, route preparation, deployment, acknowledgment, probation and cleanup for local
deployments. Producer integration tests cover rejection and two consecutive synthetic approved
releases. Actual Airflow execution with synthetic task callbacks verifies success and gate-failure
cleanup. The full live GPU run through Airflow passed in `gpu-producer-airflow-05`.

The trusted worker configures `FINSERVE_PIPELINE_REQUEST`, `FINSERVE_REGISTRY_URL` and `FINSERVE_ARTIFACT_ROOT`. The request identifies existing baseline/candidate run directories, raw quality outputs, a frozen suite, immutable revisions, policy and both canonical serving profiles. Database and artifact references must retain their original namespace across retries.

Registration verifies raw run evidence and persists the lifecycle specification. Canonical jobs record their required gate mode before publishing profile inputs; a missing profile after a crash cannot fall back to a legacy drill gate. Evaluation checks profile/run identities and recomputes performance and quality. Chat cohorts also bind API, system instruction and template digest between performance and quality. Rejection or invalid evidence prevents deployment.

The same gate is available to a trusted CI worker:

```sh
uv run --no-sync python -m finserve.registry.release_gate --job-id JOB_ID --output /external/evidence/new-outcome.json
```

It exits 0 for approval, 2 for rejection and 3 for invalid evidence. The manual `performance-gate.yml` workflow requires a trusted Linux self-hosted runner with registry/artifact access. It does not run GPU work or deploy from pull requests. Adding the workflow is not proof of an executed CI run.

## Activation and recovery

Deployment loads the server-configured `FINSERVE_DEPLOYMENT_ADAPTER` factory and recomputes the gate before invoking it. Lifecycle leases, immutable request identities and adapter idempotency govern retries. Uncertain external action is reconciled through exact-revision health rather than blindly repeated. A decision supplied by an arbitrary caller is not an authorization capability.

The warm adapter switches traffic between already running registered backends and probes the active route. It does not pull images or start pods. After lifecycle promotion, `release_activation.acknowledge_release` recomputes the canonical gate, verifies the recorded lifecycle decision, probes current traffic and records the exact activation in the rollback controller. It requires the persisted lifecycle route-action receipt and holds the route write lock during controller acknowledgment. Repeating it does not increment the generation again. The previous known-good revision remains unchanged. After a complete healthy monitor window, `release_activation.complete_probation` rechecks approval and current traffic before advancing known-good with retained observation evidence. The local producer DAG calls these stages for initial deployments and updates; full live GPU execution passed in `gpu-producer-airflow-05`. [ADR014](adr/ADR-014-rollback-known-good-revision.md) describes restoration rules.

## Local producer workflow

The producer DAG has 18 tasks. It serializes fetch, image build, plan freeze and both cohorts'
launch/quality/performance stages. After gate approval, it registers the verified profiles,
selects the initial baseline, checks actual traffic health and seeds controller state under a
route fence. Separate tasks activate the candidate, acknowledge the exact action and run
probation. Credentials come from `FINSERVE_API_KEY` at runtime; the request freezes a `rollout`
object containing `traffic_url` and bounded probe settings. A missing rollout object fails
before collection. The traffic gateway must already run against the same route store and
deployment ID. These tasks do not provision the gateway or cloud resources.

`cleanup_unused` uses `all_done` and depends on the collection and release chain. The terminal
`release_complete` task requires both probation and cleanup to succeed, so cleanup cannot hide
an upstream failure. This follows Airflow's [leaf-task status semantics](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dag-run.html).
Ambiguous launches and failed/unacknowledged cutovers still require reconciliation.

An initial request creates new baseline and candidate identities and requires an unused
deployment ID at generation zero. An update supplies `producer.existing_baseline_stage`, such
as `previous-job:candidate-launch`, and the current `expected_generation`. Its baseline port,
engine parameters and served model must match that original completed launch. Freeze and
collection task entry points check the registered backend, route and controller's active and
known-good revision under route-then-controller locks. A stale baseline or active rollback
blocks new collection tasks.

The update reuses the baseline's original source, image, profile and container start. It
collects new quality and performance evidence after freezing the new plan; earlier results
are not relabeled. The candidate uses the new build. Cleanup never stops a borrowed baseline.
Previously served backends also remain protected because durable stream-drain evidence is
not implemented. Their retained memory must be included in host capacity planning; task
serialization does not free resident models.

`producer_pipeline.freeze_stage()` reads `FINSERVE_PRODUCER_REQUEST`, a server-owned JSON
`ProducerExecution` containing `producer` (`ProducerInput`), `routes` and `control`. The read
is capped at four MiB. Route and controller databases must already exist at distinct canonical
paths outside the checkout. Their paths and persistent database identities are frozen with
the job before collection. Existing stores need the current store schema, including identity
metadata; task entry points never migrate or initialize them.
`collection_stage(job_id, step)` reopens the configured registry and artifact store for one
producer action. `cleanup_stage(job_id)` uses the frozen route/control paths, ignores later
changes to the request file, and refuses missing or recreated stores. Every store transaction
opens with SQLite `mode=rw` and verifies identity before reading protected state. It stops only verified never-served
producer runtimes. The registry and artifact environment must retain their original namespace
across tasks. These callable entries are verified with actual fixture-byte fetch and replay.

`producer_runtime.freeze_producer` now freezes server-owned source/model inputs, paired typed
engine configurations, load, suite, policy and workspace. `produce_step` accepts that job ID
and an explicit fetch/build/freeze/launch/quality/performance task name. It derives image and
model identities from completed producer receipts, then freezes the collection plan before
launching either cohort. Runtime identity fields in the load template must be `undeclared`;
the verified build fills them. Hardware must be declared. Ports must be distinct and bound
to loopback. The implementation currently supports the pinned local vLLM/Docker runtime.

The task runtime is tested from actual fixture-byte fetch through synthetic Docker build and
launch, real collector protocol handling, registration, rejection and approved fixture rollout.
The approved fixture uses an explicit answer oracle and broad predeclared timing thresholds;
it is an orchestration test, not evidence of model quality or performance.

Model fetch/verification and committed-source runtime build APIs run locally and now have producer DAG tasks. Earlier actual image preflight, GPU measurement and canonical rejection evidence remain separate from scheduler fixtures.

Remaining work includes live GPU execution of the complete DAG, reclamation of drained historical backends, image publication where required and cloud deployment. The 32-case release suite remains frozen. A three-case development smoke cannot substitute for it, and failed quality cannot be waived to demonstrate activation.
