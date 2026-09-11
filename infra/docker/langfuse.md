# Local Langfuse integration

This separate Docker Compose project verifies FinServe OTLP ingestion into a real local Langfuse database. It uses six services: Langfuse web/worker, PostgreSQL, ClickHouse, Redis and MinIO. It is a small functional verification footprint, below the vendor's production sizing, with no high availability, backup service or cloud deployment claim.

The project name is `finserve-langfuse-local`. Its containers, networks and named volumes are independent of the base `finserve` project. Only the web service publishes a port: `127.0.0.1:3037`. Storage and queue services have no host ports and use an internal network. Web also joins an ingress bridge because Docker Desktop does not publish ports on an internal-only network; that bridge permits web egress. Langfuse telemetry, open signups, AI features and mail integration are disabled. A reserved `example.invalid` address identifies the local bootstrap owner; no email is sent.

The stack pins Langfuse **4.33.0** web/worker by image digest, with upstream source `81bbfd169b72ea2ed53639699cc6632e8f908ce8`. GitHub release 4.34.0 existed during preparation, but its versioned images were not yet published. PostgreSQL 17.11, ClickHouse 25.12, Redis 7.2 and the upstream-recommended Chainguard MinIO image are also digest-pinned. The MinIO publisher exposes a rolling tag; the digest fixes its bytes. Record local image IDs after pulling instead of treating the tag as evidence.

## Prepare and start

Install the locked telemetry optional dependency. Generate secrets into an access-controlled directory outside the repository:

```sh
uv sync --frozen --extra telemetry
uv run --no-sync python -m finserve.telemetry.langfuse_probe prepare --secrets-file /private/finserve/langfuse.env
docker compose --env-file /private/finserve/langfuse.env -f infra/docker/compose.langfuse.yaml config --quiet
docker compose --env-file /private/finserve/langfuse.env -f infra/docker/compose.langfuse.yaml pull
docker compose --env-file /private/finserve/langfuse.env -f infra/docker/compose.langfuse.yaml up -d --wait --wait-timeout 360
```

Run this outside any GPU measurement interval. The configured aggregate memory cap is 6,336 MiB for a functional test. ClickHouse background pools are reduced for the local PID cap. The generator creates independent random credentials, an API key pair and a local owner password. It creates the file exclusively with POSIX mode 600 where supported; Windows directory ACLs remain the operator's responsibility. Use `docker compose config --quiet` for validation: the full resolved configuration includes secrets. Docker administrators can inspect container environment variables.

Headless bootstrap creates the organization, project, project API keys and owner together. Reusing existing volumes preserves those resources; changing an environment file does not constitute credential rotation. The secret generator never overwrites an existing file.

## Verify stored data

```sh
uv run --no-sync python -m finserve.telemetry.langfuse_probe verify --secrets-file /private/finserve/langfuse.env --base-url http://127.0.0.1:3037 --output /private/finserve/langfuse-verification-01
```

The probe exports one explicitly synthetic FinServe span, then queries `/api/public/v2/observations` for its exact trace/span IDs and a bounded time window. Verification requires the stored observation with its expected name and no input, output, injected private event/status text or credentials. The evidence directory retains a terminal manifest and the queried observation, including failures. The test covers ingestion and storage; model inference and serving throughput are measured separately.

For application export, set `FINSERVE_OTLP_ENDPOINT=http://127.0.0.1:3037/api/public/otel/v1/traces`, `FINSERVE_OTLP_PROTOCOL=langfuse-v4`, and `FINSERVE_OTLP_AUTHORIZATION` to `Basic ` followed by base64 of the project public/secret key pair separated by a colon. Construct this value in memory from the private file. Do not paste it into source, shell history or evidence. Set the desired `FINSERVE_TRACE_SAMPLE_RATIO`; omit `FINSERVE_TRACE_PATH` when using OTLP. The fixed protocol option adds `x-langfuse-ingestion-version:4` and validates the bounded JSON queue acknowledgement returned by the pinned server. Default `otlp` still requires the standard protobuf acknowledgement and rejects partial success. Arbitrary OTEL header environment variables are not forwarded.

## Stop after verification

Stop only this project's services to free CPU/RAM before further GPU experiments; volumes retain database evidence:

```sh
docker compose --env-file /private/finserve/langfuse.env -f infra/docker/compose.langfuse.yaml stop --timeout 30
```

This command does not stop `finserve-redis-1` or other projects. Do not add volume-deletion options when retaining evidence. Before final reporting, retain source hashes, resolved image IDs, container health/status, startup failures and the queried trace artifact. A healthy local stack does not establish hosted Langfuse ingestion or AWS readiness.

The retained local run verified a stored FinServe span with seven synthetic output tokens, its configured model, and no input/output or injected private content. All six services stopped afterward. Web and worker exceeded the 30-second graceful-stop budget and exited with code 137; storage services exited with code 0. This demonstrates ingestion, not graceful worker shutdown under load. Docker Hub CDN pulls initially failed; an external Compose override used `mirror.gcr.io` images with the exact same pinned index digests. The original pull failures, matching manifest inspections, runtime image IDs and stopped-container receipts remain in private evidence.

## References

The configuration follows the [versioned upstream Compose source](https://github.com/langfuse/langfuse/blob/v4.33.0/docker-compose.yml), [headless initialization dependencies](https://langfuse.com/self-hosting/administration/headless-initialization), and [Docker deployment guide](https://langfuse.com/self-hosting/deployment/docker-compose). The [OTLP integration guide](https://langfuse.com/integrations/native/opentelemetry) specifies Basic authentication and the v4 header; the [observations API](https://langfuse.com/docs/api-and-data-platform/features/observations-api) specifies the query used for verification. Consult [production sizing](https://langfuse.com/self-hosting/configuration/scaling) before deploying beyond this local proof.
