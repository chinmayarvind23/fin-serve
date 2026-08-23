# Commands

Keep synchronized with actual runnable scripts.

```bash
uv sync
uv run ruff check .
uv run pyright
uv run pytest
bun install
bun test
bun run build
docker compose up -d
uv run python -m finserve.benchmark.runner --config benchmarks/configs/baseline.yaml
uv run python -m finserve.evaluation.run --suite evals/golden/text.yaml
ray start --head
serve run deploy/ray/serve.yaml
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform plan
kubectl get rayservices
```

Do not document commands that are not working yet.
