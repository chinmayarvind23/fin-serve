# Security boundaries

FinServe currently uses separately configured service credentials and trusted operator processes. It does not implement the previously proposed five-role RBAC system. The public inference API and read-only explorer cannot invoke deployment operations; the release CLI/Airflow worker and its configured adapter hold that authority.

## Requests and ownership

Bearer authentication protects configured text serving and is required for image/job capabilities. The local transport fixture can run without a key on loopback. External deployment requires authenticated TLS ingress. Request/body/output bounds, separate admission pools and deadlines limit accepted work; optional Redis quotas are shared across replicas only with matching principal and quota configuration.

The serving adapters close HTTP streams on cancellation and retain local/proxy ownership until cleanup is acknowledged. Native thread or GPU work cannot be stopped merely by cancelling an asyncio task. Closing an engine HTTP socket is not proof that every external GPU kernel has stopped; kernel/process drain needs runtime evidence.

## Model and artifact inputs

The model producer uses fixed Hub commits, bounded file lists and source checksums. It records raw file SHA256 values and rechecks the mounted snapshot before engine startup. The runtime builder extracts a bounded exact Git archive, rejects links/path aliases and records actual image identities. Engine parameters are typed; arbitrary remote-code or destination overrides are excluded.

Local model/artifact volumes belong to trusted operators. Static path checks do not protect against a malicious process replacing those files concurrently. CAS reads verify namespace, length and digest. Invalid evidence, a missing canonical profile or failed quality blocks promotion.

## Configuration and deployment

Credentials remain in private process configuration, not images, manifests or browser bundles. Browser access uses a separate web credential retained only in memory. Pinned bases/lockfiles and non-root runtime configuration improve reproducibility; they are not evidence that vulnerability scans or SBOM generation ran.

Terraform and Helm define private data services, scoped workload identities, resource limits and NetworkPolicies. Those configurations have local validation evidence. Their enforcement, application secret provisioning, TLS and cloud service health remain unverified until deployment. The local SQLite stores do not imply a highly available production control plane.

## Data visibility

Operational traces retain a bounded metadata allowlist and remove prompt/event/exception content before JSONL or OTLP export. Metric labels exclude request IDs and prompts. Benchmark artifacts intentionally retain raw synthetic prompts and outputs for independent grading; they stay in the private evidence workspace. Do not confuse trace redaction with deletion of evaluation evidence.

The [threat model](threat-model.md) records limits and the [failure guide](failure-modes.md) distinguishes fail-closed control decisions from recoverable transport failures.
