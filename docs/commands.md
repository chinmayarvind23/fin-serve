# Running and measuring FinServe

Commands run from the repository root. Shell examples use POSIX syntax; Windows users can run them in WSL. Keep credentials and generated evidence outside the checkout. Python 3.12 and Bun 1.3.10 are the tested versions.

## Development checks

```sh
uv sync --locked --extra reference --extra multimodal --extra cache --extra registry \
  --extra visual-worker --extra visual-proto --extra explorer --extra vision --extra telemetry
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest --cov=finserve --cov-branch --cov-fail-under=85
bun install --frozen-lockfile
bun run lint
bun run typecheck
bun run test
bun run build
```

Optional runtimes are separate: `distributed` installs Ray and `orchestration` installs Airflow on supported Linux Python versions. GPU engine packages have their own environments and images. The base gateway does not install vLLM, SGLang or CUDA.

## HTTP fixture

```sh
uv sync --locked
uv run --no-sync uvicorn finserve.gateway.app:from_env --factory --host 127.0.0.1 --port 8000
```

In a second terminal:

```sh
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hello","max_tokens":5}'
uv run --no-sync python -m finserve.benchmark.runner \
  --url http://127.0.0.1:8000/v1/completions \
  --output ../resources/fin_serve/evidence/fixture-example-01 \
  --requests 64 --warmup 8 --concurrency 4 --max-tokens 16
```

This default fixture tests transport and accounting. Its character tokens and timings do not measure a language model. An output directory must be new; retain failed attempts under their original names.

Set `FINSERVE_API_KEY` in the gateway environment to require authentication. The runner and experiment clients read the same variable without storing it in artifacts. Authenticated HTTP requests send `Authorization: Bearer <service credential>`. Never put a real credential in a committed command or workload.

## Real engines and frozen experiments

Start a separately managed model engine first, with pinned model/tokenizer files and recorded arguments. Configure the gateway with `FINSERVE_ENGINE=vllm` or `sglang`, `FINSERVE_ENGINE_URL=http://<private-engine>:8020/v1`, and `FINSERVE_MODEL=<served model name>`. `FINSERVE_ENGINE_API_KEY` supplies its engine credential. Ray uses `FINSERVE_ENGINE=ray-http` and a private Ray HTTP URL; see the [container setup](../infra/docker/README.md).

The experiment wrapper accepts a serialized `RunConfig`, freezes the workload/configuration and collects raw GPU samples with the HTTP records:

```sh
uv run --no-sync python -m finserve.benchmark.experiment \
  --url http://127.0.0.1:8000/v1/completions \
  --output /absolute/private/experiment-01 \
  --workload benchmarks/configs/text-release-v1.json \
  --config /absolute/private/frozen-run-config.json
```

Create and review the config before running. It declares requests, warmup, concurrency, arrival mode/rate, timeout, hardware, model and tokenizer revisions, engine settings and source/image identities. The 6,144-request published comparison used 3,072 measured requests and 64 warmups per configuration. A current rerun uses current source and becomes a new experiment; it cannot recreate an old run identity.

Quality is a separate measurement against the unchanged suite. The quality client reads `FINSERVE_API_KEY` when configured:

```sh
uv run --no-sync python scripts/run_quality.py \
  --suite evals/golden/correctness-32-v1.json \
  --output /absolute/private/quality-candidate-01 \
  --url http://127.0.0.1:8000/v1/completions \
  --model <served-model> --model-revision <pinned-commit> \
  --tokenizer-revision <pinned-commit> --engine <engine-version> \
  --engine-config <recorded-configuration> \
  --reference /absolute/private/quality-baseline-01
```

Run the baseline without `--reference` first. Retain its answers and invalid outputs. Do not change expected answers after inspecting failures. [Results](results.md) explain the failed gate in the recorded comparison.

## Separate services

- [Bun edge](../apps/api/README.md): bounded inference proxy and public API key.
- [Evidence explorer](../apps/web/README.md): import verified experiments, then start the SQLite-backed GraphQL service and Bun website.
- [Visual jobs](../src/finserve/multimodal/README.md): gRPC worker, durable job database and generation artifacts.
- [Containers](../infra/docker/README.md): explicit `infra/docker/compose.yaml`, private environment file and optional monitoring override.
- [Airflow](../pipelines/airflow_dags/README.md): isolated orchestration environment and shared release gate.
- [Terraform](../infra/terraform/README.md) and [Kubernetes](../infra/kubernetes/README.md): foundation and workload commands, prerequisites and current deployment limits.

These service guides identify their actual paths and required configuration. A passing local fixture or Terraform mock test does not establish a deployed cloud service.
