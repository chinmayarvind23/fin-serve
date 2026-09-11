"""CPU transport evidence tests for the frozen cohort recorder; no engine/GPU claims."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from finserve.benchmark.routing_cohort import RoutingTopology, gpu_summary, observe, run_cohort
from finserve.benchmark.routing_workload import MODEL, REVISION, frozen_workload


def topology() -> RoutingTopology:
    """Supply clearly synthetic process identities only for the CPU transport fixture."""
    return RoutingTopology.model_validate(
        {
            "physical_gpu_uuid": "GPU-cpu-fixture",
            "backends": [
                {"replica_id": "a", "endpoint": "http://a/v1", "pid": 100, "start_ticks": 1},
                {"replica_id": "b", "endpoint": "http://b/v1", "pid": 101, "start_ticks": 2},
            ],
            "profile": {"model": MODEL, "model_revision": REVISION},
        }
    )


class Fixture:
    """Fake only the network; production framing, accounting and artifact code execute unchanged."""

    def __init__(self, mode: str = "success") -> None:
        """A bounded failure switch exercises a specific evidence boundary in each test."""
        self.mode = mode
        self.decisions: list[dict[str, Any]] = []
        self.started = asyncio.Event()

    async def respond(self, request: httpx.Request) -> httpx.Response:
        """Preserve real NDJSON envelopes and explicit synthetic status without acquiring a GPU."""
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {"replica_id": name, "model": MODEL, "healthy": True} for name in ("a", "b")
                    ],
                    "physical_gpu_observation": {
                        "epoch_s": time.time(),
                        "collection_seconds": 0,
                        "error": None,
                        "configured_shared_uuid": "GPU-cpu-fixture",
                        "router_observed_at": 1,
                        "devices": [
                            {
                                "uuid": "GPU-cpu-fixture",
                                "name": "CPU test fake",
                                "utilization_percent": 0,
                                "memory_used_mib": 1,
                                "memory_total_mib": 2,
                            }
                        ],
                    },
                    "recent_decisions": self.decisions,
                },
            )
        payload = json.loads(request.content)
        self.started.set()
        if self.mode == "cancel":
            await asyncio.sleep(60)
        policy = "adaptive" if self.mode == "wrong-policy" else "least_load"
        self.decisions.append({"request_id": payload["request_id"], "decision": {"policy": policy}})
        frames = [{"replica_id": "a", "token": {"text": "one", "generated_tokens": 0}}]
        if self.mode != "partial":
            frames.append(
                {
                    "replica_id": "a",
                    "token": {"text": "", "generated_tokens": 1, "finish_reason": "stop"},
                }
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content="".join(json.dumps(frame) + "\n" for frame in frames),
        )


def install(monkeypatch: pytest.MonkeyPatch, fixture: Fixture) -> None:
    """Inject socket responses while retaining independent real HTTPX client pool lifetimes."""
    original = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        """Share fixture transport without changing parser or runner code."""
        kwargs["transport"] = httpx.MockTransport(fixture.respond)
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setenv("FINSERVE_RAY_API_KEY", "cpu-test-routing-key")


async def test_complete_cohort_preserves_exact_population_and_artifact_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Eight warmups and all 64 measured requests retain usage and source/hash evidence."""
    fixture = Fixture()
    install(monkeypatch, fixture)
    output = tmp_path / "complete"
    result = await run_cohort("http://router", frozen_workload(), "least_load", output, topology())
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert result["status"] == "complete" and len(records) == 72
    assert len(result["artifacts"]) >= 18 and len(result["sources"]) >= 13
    assert all(row["record"]["generated_tokens"] == 1 for row in records)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["requests"]["all"]["offered_requests"] == 64
    assert summary["requests"]["declared_slo_denominator"] == 64


@pytest.mark.parametrize("mode", ["partial", "wrong-policy"])
async def test_failed_cohort_keeps_attempts_and_marks_manifest(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Partial warmup and mismatched-policy results cannot become successful evidence."""
    install(monkeypatch, Fixture(mode))
    output = tmp_path / mode
    with pytest.raises(ValueError):
        await run_cohort("http://router", frozen_workload(), "least_load", output, topology())
    manifest = json.loads((output / "manifest.json").read_text())
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert manifest["status"] == "failed" and manifest["error"] == "ValueError"
    assert len(records) == (8 if mode == "partial" else 72)
    assert all(row["record"]["output"] == "one" for row in records)
    if mode == "partial":
        assert all(not row["record"]["success"] for row in records)


async def test_cancelled_cohort_drains_and_persists_interrupted_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation cannot leave running metadata or become a completed denominator."""
    fixture = Fixture("cancel")
    install(monkeypatch, fixture)
    output = tmp_path / "cancelled"
    task = asyncio.create_task(
        run_cohort("http://router", frozen_workload(), "least_load", output, topology())
    )
    await fixture.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    manifest = json.loads((output / "manifest.json").read_text())
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert manifest["status"] == "interrupted" and len(records) == 8
    assert all(not row["record"]["success"] for row in records)


def test_topology_rejects_aliases_or_different_model_identity() -> None:
    """Distinct URLs alone cannot turn one PID into two engines or change the held-out model."""
    original = topology().model_dump()
    original["backends"][1]["pid"] = original["backends"][0]["pid"]
    with pytest.raises(ValueError, match="distinct"):
        RoutingTopology.model_validate(original)


def test_clock_drift_invalidates_gpu_mean() -> None:
    """A wall-clock jump cannot turn GPU samples into a plausible unrelated request interval."""
    result = gpu_summary([], 100, 110, 10, 19)
    assert result["average_gpu_utilization_percent"] is None
    assert result["clock_drift_seconds"] == 1 and "clock_warning" in result


async def test_known_observation_failure_breaks_previous_sample_hold(tmp_path: Path) -> None:
    """A failed status observation becomes an explicit missing-device sample in aggregate input."""
    from finserve.benchmark.gpu import TelemetrySample

    stop = asyncio.Event()
    samples: list[TelemetrySample] = []

    def fail(request: httpx.Request) -> httpx.Response:
        """Stop after one failed HTTP call to test failure recording without polling delays."""
        stop.set()
        raise httpx.ConnectError("fixture offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        await observe(client, "http://router", stop, tmp_path / "status.jsonl", samples)
    assert len(samples) == 1 and samples[0].error == "ConnectError" and not samples[0].devices
