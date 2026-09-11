# Build the small HTTP/control boundary separately from GPU engine images.
FROM ghcr.io/astral-sh/uv:0.8.4@sha256:40775a79214294fb51d097c9117592f193bcfdfc634f4daa0e169ee965b10ef0 AS uv
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254 AS build
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra cache --extra visual-worker --extra vision --extra telemetry --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra cache --extra visual-worker --extra vision --extra telemetry --no-editable

FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
ARG SOURCE_REVISION=unrecorded
LABEL org.opencontainers.image.source="https://github.com/chinmayarvind23/fin-serve"
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV FINSERVE_REVISION=$SOURCE_REVISION
RUN groupadd --gid 10001 finserve && useradd --uid 10001 --gid finserve --no-create-home finserve
WORKDIR /app
COPY --from=build --chown=10001:10001 /app/.venv /app/.venv
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "finserve.gateway.app:from_env", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
