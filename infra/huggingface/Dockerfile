FROM ghcr.io/astral-sh/uv:0.8.4@sha256:40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0 AS uv
FROM oven/bun:1.3.10@sha256:b86c67b531d87b4db11470d9b2bd0c519b1976eee6fcd71634e73abfa6230d2e AS web
WORKDIR /build
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile
COPY apps ./apps
COPY infra/huggingface ./infra/huggingface
RUN bun build infra/huggingface/space_server.ts --target=bun --entry-naming='[name].[ext]' --outdir=/web

FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254 AS python
COPY --from=uv /uv /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra explorer --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra explorer --no-editable

FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
ARG SOURCE_REVISION=unrecorded
LABEL org.opencontainers.image.source="https://github.com/chinmayarvind23/fin-serve"
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
LABEL org.opencontainers.image.description="CPU evidence explorer; no GPU inference"
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN groupadd --gid 1000 space && useradd --uid 1000 --gid space --create-home space \
    && mkdir /data && chown 1000:1000 /data
WORKDIR /app
COPY --from=python /app/.venv /app/.venv
COPY --from=web /usr/local/bin/bun /usr/local/bin/bun
COPY --from=web /web /app/web
COPY infra/huggingface/launch.py /app/launch.py
COPY infra/huggingface/landing.html infra/huggingface/landing.css /app/
COPY docs/assets/inference-demo.gif /app/public/inference-demo.gif
USER 1000:1000
EXPOSE 7860
HEALTHCHECK --interval=10s --timeout=3s --start-period=65s --retries=3 \
  CMD python -c "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:7860/healthz', timeout=2)"
CMD ["python", "/app/launch.py"]
