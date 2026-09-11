# Run FinServe without paid hosting

The [Hugging Face page](https://huggingface.co/spaces/chinmayarvind/finserve)
publishes static aggregate results. It runs no inference or API. The full explorer
and serving processes run on your own computer. No AWS account, Hugging Face PRO,
paid inference endpoint or Kubernetes cluster is needed for these local recipes.

## Full evidence explorer on Windows

Install Git, Python 3.12 and Docker Desktop with Linux containers, then open PowerShell:

```powershell
git clone https://github.com/chinmayarvind23/fin-serve.git
Set-Location fin-serve
docker build -f infra/huggingface/Dockerfile -t finserve-explorer:local .
$env:FINSERVE_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:FINSERVE_WEB_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
$finserveExplorerId = docker run --detach --publish 127.0.0.1:7860:7860 --memory 512m --cpus 2 --pids-limit 128 --env FINSERVE_API_KEY --env FINSERVE_WEB_KEY finserve-explorer:local
if ($LASTEXITCODE -ne 0) { throw "Explorer container did not start" }
```

Open <http://127.0.0.1:7860>. The public figures need no key. Open the explorer
and use the value of `$env:FINSERVE_WEB_KEY`; inspect it only in your private
terminal. The internal API key is different. Both keys are generated locally and
passed by environment variable, not stored in Git or baked into the image.

The registry initially has no runs. This container demonstrates the complete
Bun-to-GraphQL read path, not a prepopulated experiment database. Follow the
[explorer guide](../apps/web/README.md) to import your retained experiment and
quality artifacts and run against a persistent external registry. Removing this
basic container discards its local data.

Check readiness and stop only this container:

```powershell
curl.exe --fail http://127.0.0.1:7860/healthz
docker logs --tail 30 $finserveExplorerId
docker stop $finserveExplorerId
docker rm $finserveExplorerId
Remove-Item Env:FINSERVE_API_KEY, Env:FINSERVE_WEB_KEY
```

For Linux, use the same build/run flags and generate the variables with
`export FINSERVE_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')`
and the equivalent command for `FINSERVE_WEB_KEY`. Capture the ID returned by `docker run` and use that exact ID for cleanup.

## Real text inference on a local NVIDIA GPU

Use Linux or Ubuntu in WSL2, an NVIDIA driver with working Docker GPU support,
Python 3.12, uv, and an idle GPU with at least about 8 GB VRAM. The commands below
are a functional single-engine recipe, not the frozen benchmark protocol.
CPU users can still run the explorer and the [transport fixture](../README.md#run-locally).

In the Linux checkout, download the public pinned weights to a directory outside
the repository. No Hugging Face login or paid API is required for these weights:

```bash
uv sync --frozen
export FINSERVE_MODEL_DIR="$HOME/finserve-local/qwen25-15b"
mkdir -p "$FINSERVE_MODEL_DIR"
uvx --from huggingface_hub hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --local-dir "$FINSERVE_MODEL_DIR"
docker pull mirror.gcr.io/vllm/vllm-openai:v0.29.0@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
unset FINSERVE_ENGINE_CONTAINER
uv run --no-sync python scripts/check_gpu.py && \
FINSERVE_ENGINE_CONTAINER=$(docker run --detach --gpus all \
  --publish 127.0.0.1:8020:8000 --shm-size 1g \
  --mount "type=bind,source=$FINSERVE_MODEL_DIR,target=/model,readonly" \
  --entrypoint vllm \
  mirror.gcr.io/vllm/vllm-openai:v0.29.0@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b \
  serve /model --served-model-name finserve-qwen-15b --host 0.0.0.0 --port 8000 \
  --dtype float16 --enforce-eager --max-model-len 1024 \
  --max-num-batched-tokens 1024 --max-num-seqs 4 \
  --gpu-memory-utilization 0.60 --no-enable-prefix-caching)
docker logs --tail 40 "$FINSERVE_ENGINE_CONTAINER"
curl --fail http://127.0.0.1:8020/v1/models
```

The guard refuses when another compute process is present or more than 512 MiB
is occupied. It checks availability once; it cannot reserve the GPU against other
threads. Coordinate an idle window with whoever owns Ollama or another model
server. Do not kill another project or start two engines on this small device.
If the engine exits, retain its logs and check VRAM before retrying. Model loading
may take several minutes; a running container alone does not mean inference is ready.

Start the FinServe gateway in the same Linux shell:

```bash
export FINSERVE_ENGINE=vllm
export FINSERVE_ENGINE_URL=http://127.0.0.1:8020/v1
export FINSERVE_MODEL=finserve-qwen-15b
uv run --no-sync uvicorn finserve.gateway.app:from_env --factory --host 127.0.0.1 --port 8000
```

In another terminal, send a request through FinServe:

```bash
curl --fail -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"finserve-qwen-15b","prompt":"The capital of France is","max_tokens":16}'
```

This loopback-only development gateway has no API key unless you set
`FINSERVE_API_KEY`; if set, send its bearer token. Do not expose either listener
publicly. Stop the gateway with Ctrl+C, then stop and remove only the engine:

```bash
docker stop "$FINSERVE_ENGINE_CONTAINER"
docker rm "$FINSERVE_ENGINE_CONTAINER"
```

The model still has known correctness failures on the retained FinServe suites.
Successful inference is not release approval. See [results](results.md) for the
actual measurements and failed gates. The runtime image is large; download time,
disk capacity and local GPU memory are the practical requirements for this path.

## Publish your own free static Space

```bash
uv run --no-sync python infra/huggingface/package.py --static --output /absolute/outside/repo/static-space
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf repos create your-account/finserve --type space --space-sdk static
uvx --from huggingface_hub hf upload your-account/finserve /absolute/outside/repo/static-space --type space
```

The package includes one HTML page, CSS, six committed aggregate files, a Space
README and a file-hash manifest. It contains no credentials, request records,
model weights or backend. [Static Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-static)
use `sdk: static` and `app_file: index.html`. Keep the Docker package for local use
when your account cannot host Docker Spaces without a subscription.
