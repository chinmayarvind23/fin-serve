"""Exercise authenticated HTTP intent/status/artifact against a real loopback gRPC boundary."""

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest

from finserve.contracts.visual import VisualArtifact, VisualJobRequest
from finserve.gateway.app import create_app
from finserve.multimodal.benchmark import conditioning_image
from finserve.multimodal.jobs import VisualJobCoordinator, VisualJobStore
from finserve.multimodal.visual_rpc import MODEL_REVISION, VisualRPCClient, VisualWorker

KEY = "test-only-http-visual-service-key"


def fixture_output(_: VisualJobRequest) -> VisualArtifact:
    """Keep this transport test independent of JAX compilation, which has separate real evidence."""
    png = b"\x89PNG\r\n\x1a\nhttp-fixture"
    return VisualArtifact(
        png=png,
        sha256=hashlib.sha256(png).hexdigest(),
        model_sha256="a" * 64,
        initialization_ns=0,
        generation_ns=1,
        rendering_ns=1,
    )


async def test_http_job_acceptance_idempotency_and_artifact(tmp_path: Path) -> None:
    """Only durable accepted work reaches the worker; completed bytes are verified on read."""
    pytest.importorskip("grpc")
    worker = VisualWorker(KEY, fixture_output)
    server, port = await worker.start()
    coordinator = VisualJobCoordinator(
        VisualJobStore(tmp_path / "jobs.db"), VisualRPCClient(f"127.0.0.1:{port}", KEY)
    )
    app = create_app(api_key=KEY, visual_jobs=coordinator)
    payload = VisualJobRequest(
        image=conditioning_image(), model_revision=MODEL_REVISION
    ).model_dump()
    headers = {"Authorization": f"Bearer {KEY}", "Idempotency-Key": "immutable-job-key"}
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                assert (await client.post("/v1/visual/jobs", json=payload)).status_code == 401
                models = await client.get("/v1/visual/models", headers=headers)
                assert models.json()["models"][0]["revision"] == MODEL_REVISION
                invalid = await client.post(
                    "/v1/visual/jobs",
                    json={**payload, "model_revision": "unknown"},
                    headers=headers,
                )
                assert invalid.status_code == 404
                accepted = await client.post("/v1/visual/jobs", json=payload, headers=headers)
                assert accepted.status_code == 202
                job_id = accepted.json()["job_id"]
                repeated = await client.post("/v1/visual/jobs", json=payload, headers=headers)
                assert repeated.json()["job_id"] == job_id
                conflict = await client.post(
                    "/v1/visual/jobs", json={**payload, "seed": 18}, headers=headers
                )
                assert conflict.status_code == 409
                response = accepted
                for _ in range(100):
                    response = await client.get(f"/v1/visual/jobs/{job_id}", headers=headers)
                    if response.json()["state"] == "succeeded":
                        break
                    await asyncio.sleep(0.01)
                assert response.json()["state"] == "succeeded"
                artifact = await client.get(f"/v1/visual/jobs/{job_id}/artifact", headers=headers)
                assert artifact.status_code == 200
                assert (
                    artifact.content == fixture_output(VisualJobRequest.model_validate(payload)).png
                )
                assert (
                    artifact.headers["etag"] == f'"{hashlib.sha256(artifact.content).hexdigest()}"'
                )
                assert (await client.get(f"/v1/visual/jobs/{job_id}/artifact")).status_code == 401
                assert (
                    await client.get("/v1/visual/jobs/unknown", headers=headers)
                ).status_code == 404
    finally:
        await server.stop(0)
        await worker.close()


async def test_visual_mode_requires_authentication(tmp_path: Path) -> None:
    """An accidental deployment configuration cannot expose durable visual jobs anonymously."""
    pytest.importorskip("grpc")
    coordinator = VisualJobCoordinator(
        VisualJobStore(tmp_path / "jobs.db"), VisualRPCClient("127.0.0.1:1", KEY)
    )
    try:
        with pytest.raises(ValueError, match="credential"):
            create_app(visual_jobs=coordinator)
    finally:
        await coordinator.close()
