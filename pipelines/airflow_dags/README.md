# Evidence lifecycle

`finserve_producer_lifecycle` is the separate local producer workflow. It reads
`FINSERVE_PRODUCER_REQUEST` (`ProducerExecution`) using the same registry/artifact settings,
requires frozen rollout settings and a runtime `FINSERVE_API_KEY`, and connects collection
through probation with failure-preserving cleanup. It requires an existing traffic gateway,
existing route/control stores. Initial deployments use an unused deployment ID at generation
zero; updates supply the stable `producer.existing_baseline_stage` and current generation.
A complete live GPU Airflow run remains pending. See
[the producer workflow](../../docs/airflow-pipeline.md#local-producer-workflow).

For a constrained job, freeze `producer.load.output_constraints` as one complete prompt-hash
map shared by quality and performance, and set `constraint_transport="native_vllm"`. Both
engine parameter sets must pin `structured_output_backend="xgrammar"`. Load runtime identities
remain `undeclared` until the producer derives them from build receipts; that template cannot
execute directly. Missing bindings reject the job before collection. See the
[constraint mapping contract](../../docs/reproducibility.md#frozen-output-constraint-maps).

`finserve_evidence_lifecycle` registers existing run artifacts, recomputes the canonical
profile gate, and invokes a server-configured deployment adapter only after approval.
The DAG uses the Airflow 3 public SDK. Install the `orchestration` extra in Linux.

The worker needs these environment settings:

| Setting | Meaning |
| --- | --- |
| `FINSERVE_REGISTRY_URL` | SQLAlchemy metadata database URL |
| `FINSERVE_ARTIFACT_ROOT` | Original local content-addressed artifact namespace |
| `FINSERVE_PIPELINE_REQUEST` | Private JSON file matching `PipelineRequest` |
| `FINSERVE_DEPLOYMENT_ADAPTER` | Server-owned `module:factory` for an idempotent deployment adapter |

`PipelineRequest` includes the two run directories, exact baseline/target revisions,
quality and suite files, policy, deployment generation, and both
`baseline_profile_file` and `candidate_profile_file`. Release evaluation fails if a
profile is missing. Each profile uses `ServingProfileV1`; its canonical digest must
equal the revision and benchmark configuration digests. Engine parameters, model and
tokenizer commits, endpoint, and served model must match the measured evidence.
Image identity remains a separate revision field.

Registration freezes `gate_mode=canonical-profile-v1` in the job before publishing
its profile references. Interrupted publication therefore blocks both the DAG and
the core lifecycle service until the same frozen inputs are available. Existing
legacy drill specifications retain their original canonical bytes.

Task handoffs contain only the immutable job ID. Every gate invocation verifies CAS
bytes and recomputes the raw benchmark comparison and quality result. Faster output
with failed quality is rejected. Deployment rechecks the gate; retries reconcile the
existing lifecycle operation and require verified health for the exact target.

For the same decision without deploying:

```bash
uv sync --frozen --extra registry
uv run python -m finserve.registry.release_gate \
  --job-id REGISTERED_JOB_ID --output /private/evidence/gate-attempt-001.json
```

Exit codes are `0` approved, `2` measured rejection, and `3` missing or invalid evidence.
Output must be outside the source checkout and use a new filename for every attempt.
The registry also retains immutable outcome records. Approval describes evidence
consistency and thresholds; it is not an authorization token for a public deploy API.

The manual `.github/workflows/performance-gate.yml` runs this command on a trusted
Linux self-hosted runner labelled `finserve-evidence`, using only the default branch.
Provision that runner with read/write registry access and the original artifact
namespace before dispatching a registered job ID. A copied SQLite database with a
different artifact root is insufficient because references include their namespace.
The workflow uploads the bounded outcome JSON; raw prompts and endpoint profiles stay
on the evidence runner. No GPU job or deployment runs on pull-request workflows.

Current scope: this DAG consumes existing evidence. Model-manifest hashes in a profile
are identity references; this gate does not download weights, inspect an OCI image,
run optimization experiments, or complete managed probation. Those producer and
activation stages must supply new artifact-bound runs before a full release workflow
can be claimed. Historical host-only experiments retain their original identities.
The manual CI workflow has not yet been executed on a configured evidence runner.
