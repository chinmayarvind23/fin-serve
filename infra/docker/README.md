# Local container runtime

The base stack runs the Python gateway and ephemeral Redis. Its default `fixture`
engine checks HTTP streaming and control behavior; it is not a language model.
GPU engines and the Flax worker use separate runtimes.

Create a private environment file outside the repository with independent random
values of at least 32 characters:

```dotenv
FINSERVE_API_KEY=replace-with-a-random-service-credential
REDIS_PASSWORD=replace-with-an-independent-random-credential
REDIS_KEY_SECRET=replace-with-an-independent-random-HMAC-secret
SOURCE_REVISION=your-exact-source-commit
```

From the repository root:

```sh
docker compose --env-file /path/to/private.env -f infra/docker/compose.yaml config --quiet
docker compose --env-file /path/to/private.env -f infra/docker/compose.yaml up --build -d --wait
```

The API binds to `127.0.0.1:8030`. Send an authenticated OpenAI-style request to
`/v1/completions` with model `reference`. `/healthz` is process liveness. An
external-engine deployment also needs actual inference and exact revision checks
before promotion.

For an existing engine, set `FINSERVE_ENGINE=vllm` or `sglang`, `FINSERVE_MODEL`
to its served model name, and `FINSERVE_ENGINE_URL` to its internal `/v1` URL.
`host.docker.internal:8020` is the Docker Desktop host bridge; engine binding and
host firewall rules must permit the connection. The base stack reserves no GPU.

Add `compose.monitoring.yaml` as a second `-f` argument for Prometheus
(`127.0.0.1:9097`) and Grafana (`127.0.0.1:3007`). This explicit override needs an
independent `GRAFANA_PASSWORD`; the base stack does not. A Prometheus datasource
is provisioned automatically.

The gateway runs as UID 10001 with a read-only root, dropped capabilities, bounded
memory/CPU/process count and a small temporary filesystem. Redis has no published
port or disk persistence, a 64 MiB data limit and a 128 MiB process limit. Its
`noeviction` policy preserves quota keys under pressure; quota failure returns 503.
Optional cache read failures are misses. Replicas share a quota only when they use
the same HMAC secret, principal, limit and window. Use separate secrets between
unrelated deployments.

Record source identity and resolved image digest for each build. The build uses
the lockfile and pinned uv; resolve the Python base tag to a digest for release
images. Local functional checks do not establish EKS health or performance.

The gateway image includes the gRPC client required by durable visual jobs. Set
`FINSERVE_VISUAL_GRPC_TARGET`, `FINSERVE_VISUAL_DB` (a writable external volume),
`FINSERVE_VISUAL_SERVICE_KEY` and `FINSERVE_API_KEY` to enable them. The worker runs
separately and uses the same service credential. This image contains no JAX runtime.

Build the CPU Ray runtime from the repository root:

```sh
docker build -f infra/docker/Dockerfile.ray --build-arg SOURCE_REVISION=<commit> -t <ray-image> .
```

The image installs the locked distributed dependencies, Bash and wget, and runs as
UID/GID 1000 to match the Helm chart. GPU engines remain separate. The public API
uses `FINSERVE_ENGINE=ray-http`, `FINSERVE_ENGINE_URL=<private-Ray-HTTP-URL>` and an
optional `FINSERVE_RAY_API_KEY` matching the routing actor. This endpoint speaks
internal NDJSON; callers continue using the public OpenAI-style SSE API.

Both Dockerfiles pin the Python and uv base images by digest. The Ray build still
resolves Debian package indexes for runtime utilities, so retain the built image
digest for deployment reproducibility rather than claiming byte-identical rebuilds.
