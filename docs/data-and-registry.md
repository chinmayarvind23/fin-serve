# Data and registry

The tested local registry uses SQLAlchemy with SQLite and a local content-addressed artifact store. PostgreSQL-compatible schemas and an S3 adapter are implemented; an RDS/S3 deployment has not been exercised.

## Persisted records

`Registry` stores model identities, deployment revisions, workload definitions, benchmark runs, promotion decisions, lifecycle jobs/events, canonical gate inputs and gate outcomes. A `RunBundle` references the exact manifest, raw request JSONL and summary bytes. Registration recomputes the workload and summary from raw records. A run may have no deployment revision, which keeps host observations distinct from deployable image-bound evidence.

`Revision` binds model, tokenizer, source commit, image digest and engine configuration. `ServingProfileV1` supplies a common configuration digest for measurement and warm routing. The profile includes the fixed backend endpoint and a credential environment-variable name; credential values do not enter evidence.

Immutable insertions reject different content under an existing identity. Lifecycle transitions read persisted state inside a transaction and use version/lease fencing. Decision identity includes its candidate run, so identical rejection reasons from different runs remain separate entries. Optional typed annotations attach verified quality and GPU reports to runs for the read-only explorer.

## Artifact bytes

`LocalArtifactStore` publishes SHA256-addressed bytes with immutable references and verifies their digest on read. Publication is atomic; supported POSIX systems also sync the publication directory. The volume must be controlled by the trusted producer. Path checks cannot defend against a malicious process concurrently replacing files.

`S3ArtifactStore` follows the same digest contract and bounded reads. Its request behavior is tested against boto3 with a stubbed service, not a live bucket. Raw benchmark records are JSONL; manifests and summaries are JSON. There is no implemented Parquet metrics pipeline.

The model producer keeps weights on a bounded local volume and records a small verified manifest. It checks fixed Hub commits, Git-blob or LFS source checksums and raw SHA256 for every file. The runtime builder archives a full Git commit, builds the pinned engine image and verifies actual Docker identities and labels. See [deployment](deployment.md) for runtime and infrastructure scope.

## Separate control stores

The managed performance stage freezes the workload, load configuration, collector commit,
serving profile and completed launch receipt before collection. It observes the same container
ID and start time before and after measurement, then recomputes request and GPU summaries
before publishing an immutable receipt. Completed replay verifies those stored bytes without
sending new requests. The collector must run from its declared clean commit; that commit is
recorded separately from the engine image's source commit. Missing GPU samples remain missing.

Collection runs on a dedicated event loop with bounded request population, elapsed-time and
raw-byte limits. Cancellation drains local work. An unresolved HTTP close immediately stops
new offers, drains active workers and leaves the stage unresolved, including when evidence
persistence also fails. Byte limits are observed between writes and native cleanup may outlive
the deadline. These controls do not prove remote GPU cancellation or release approval. The
complete producer-to-activation Airflow path and a live GPU run through this collection stage
remain pending.

The quality collector also preserves unresolved HTTP cleanup if recording the failed request,
closing the raw file or publishing terminal evidence fails. Its stage remains running for
reconciliation. Partial files cannot authorize retry or serve as completed quality receipts;
local transport cleanup does not establish remote inference termination.

`DeploymentStore` uses SQLite for known-good revisions, decisions, detector signals and rollback state. `WarmRouteStore` is a separate SQLite store representing the external traffic route, with its own generation and idempotency receipts. Registry approval does not imply traffic activation or verified recovery.

MLflow is an optional reporting mirror with verified decision-to-run binding; it cannot authorize deployment. Local MLflow integration is tested. Redis owns ephemeral serving state, never evidence or recovery truth. [Airflow pipeline](airflow-pipeline.md) describes the implemented lifecycle and remaining producer orchestration.
