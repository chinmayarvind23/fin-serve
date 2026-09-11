# Evidence explorer

The Bun website compares registered serving runs through a read-only GraphQL service. It shows measured throughput, latency, workload slices, GPU coverage, failed quality gates, raw request pages and lifecycle history. It never deploys from the browser. Unregistered evidence is absent; missing measurements display as unknown.

Install `uv sync --extra explorer` and `bun install --frozen-lockfile`. The initial service supports a local SQLite registry and a `LocalArtifactStore` outside this repository. Import a completed experiment with the same immutable native identities used during measurement:

```sh
python -m finserve.registry.annotations \
  --database-url sqlite:////absolute/private/evidence.db \
  --artifact-root /absolute/private/artifacts \
  --experiment /absolute/private/experiment \
  --quality /absolute/private/quality-candidate \
  --reference-quality /absolute/private/quality-baseline
```

The experiment directory contains `run/{manifest.json,requests.jsonl,summary.json}`, `environment.json`, `experiment-status.json`, `gpu.jsonl` and `gpu-summary.json`. Quality directories contain their manifest, raw requests, answers and quality report. GPU integration and quality grading are recomputed during import. Quality association matches source, model, tokenizer and engine configuration; its suite is separate from the performance workload. An association does not authorize promotion or establish an image identity.

Set `FINSERVE_REGISTRY_URL`, `FINSERVE_ARTIFACT_ROOT` and a newly generated `FINSERVE_API_KEY` of at least 16 characters. Start the registry read service:

```sh
uv run --no-sync uvicorn finserve.registry.explorer:from_env --factory --host 127.0.0.1 --port 8050
```

In another terminal, set the same internal `FINSERVE_API_KEY`, a different `FINSERVE_WEB_KEY`, and optionally `FINSERVE_EXPLORER_UPSTREAM` (default `http://127.0.0.1:8050`). Run `bun run start:web`, visit `http://127.0.0.1:8051`, and enter the web key. Keys remain process configuration and browser memory; no key is embedded in the bundle or local storage. Both listeners default to loopback. An external deployment requires authenticated TLS ingress.

`bun run build:web` produces `dist/web/server.js` and its bundled assets. Keep the complete output directory together. `FINSERVE_WEB_HOST` and `FINSERVE_WEB_PORT` configure the web listener. The GraphQL endpoint accepts POST only, with a 16 KiB body, a four-request capacity limit, bounded query shape and verified artifact reads. The edge allows only `/graphql`, injects the internal credential, and rejects responses exceeding 2 MiB. Native file reads cannot be forcibly interrupted; the read service retains capacity until its worker drains.

The website loads up to 20 registered runs and pages individual request records. The API also supports a stable `after` cursor for run lists. The browser does not export private artifacts or issue GraphQL mutations. Browser visual review is separate from the HTTP contract tests.
