"""Authenticated durable visual-job HTTP contract; polling has no execution ownership."""

import asyncio
import hashlib
import hmac
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from finserve.contracts.visual import VisualJobRequest
from finserve.multimodal.jobs import JobCapacityError, JobConflict, VisualJobCoordinator
from finserve.multimodal.visual_rpc import MODEL_REVISION


def register_visual_routes(app: FastAPI, coordinator: VisualJobCoordinator, api_key: str) -> None:
    """The configured service credential defines one tenant for this local reference deployment."""
    tenant = hashlib.sha256(api_key.encode()).hexdigest()

    def authorized(request: Request) -> bool:
        """Authenticate before reading tenant-scoped job state or accepting durable intent."""
        return hmac.compare_digest(
            request.headers.get("authorization", "").encode(), f"Bearer {api_key}".encode()
        )

    def error(code: str, status: int) -> JSONResponse:
        """Stable error codes exclude SQLite paths, backend addresses and worker credentials."""
        return JSONResponse({"error": {"code": code}}, status_code=status)

    @app.get("/v1/visual/models", response_model=None)
    async def models(request: Request) -> Response:
        """Expose the accepted immutable revision so clients need not guess a worker identifier."""
        if not authorized(request):
            return error("UNAUTHORIZED", 401)
        return JSONResponse(
            {
                "models": [
                    {
                        "revision": MODEL_REVISION,
                        "width": 8,
                        "height": 8,
                        "scope": "untrained image-conditioned reference",
                    }
                ]
            }
        )

    @app.post("/v1/visual/jobs", status_code=202, response_model=None)
    async def submit(payload: VisualJobRequest, request: Request) -> Response:
        """A repeated key returns the same durable intent; changed bytes under that key conflict."""
        if not authorized(request):
            return error("UNAUTHORIZED", 401)
        if payload.model_revision != MODEL_REVISION:
            return error("MODEL_NOT_FOUND", 404)
        key = request.headers.get("idempotency-key", "")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key) is None:
            return error("INVALID_IDEMPOTENCY_KEY", 422)
        try:
            job = await coordinator.submit(tenant, key, payload)
        except JobConflict:
            return error("IDEMPOTENCY_CONFLICT", 409)
        except JobCapacityError:
            return error("JOB_CAPACITY_EXHAUSTED", 429)
        except Exception:
            return error("JOB_STORE_UNAVAILABLE", 503)
        return JSONResponse(job.model_dump(), status_code=202)

    @app.get("/v1/visual/jobs/{job_id}", response_model=None)
    async def status(job_id: str, request: Request) -> Response:
        """Status polling does not attach to or cancel the coordinator's owned RPC."""
        if not authorized(request):
            return error("UNAUTHORIZED", 401)
        try:
            job = await asyncio.to_thread(coordinator.store.get, tenant, job_id)
            return JSONResponse(job.model_dump())
        except KeyError:
            return error("JOB_NOT_FOUND", 404)
        except Exception:
            return error("JOB_STORE_UNAVAILABLE", 503)

    @app.delete("/v1/visual/jobs/{job_id}", status_code=202, response_model=None)
    async def cancel(job_id: str, request: Request) -> Response:
        """Explicit cancellation changes intent; ambiguous RPC cleanup remains pending."""
        if not authorized(request):
            return error("UNAUTHORIZED", 401)
        try:
            job = await coordinator.cancel(tenant, job_id)
            return JSONResponse(job.model_dump(), status_code=202)
        except KeyError:
            return error("JOB_NOT_FOUND", 404)
        except Exception:
            return error("JOB_STORE_UNAVAILABLE", 503)

    @app.get("/v1/visual/jobs/{job_id}/artifact", response_model=None)
    async def artifact(job_id: str, request: Request) -> Response:
        """Return verified committed PNG bytes only after a successful fenced completion."""
        if not authorized(request):
            return error("UNAUTHORIZED", 401)
        try:
            png = await asyncio.to_thread(coordinator.store.artifact, tenant, job_id)
        except KeyError:
            return error("JOB_NOT_FOUND", 404)
        except JobConflict:
            return error("ARTIFACT_NOT_READY", 409)
        except Exception:
            return error("ARTIFACT_UNAVAILABLE", 503)
        return Response(
            png,
            media_type="image/png",
            headers={
                "ETag": f'"{hashlib.sha256(png).hexdigest()}"',
                "Cache-Control": "private, no-store",
            },
        )
