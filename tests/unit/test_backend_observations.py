"""CPU adversarial tests for actual engine gauges and shared physical-device identity."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from finserve.benchmark.gpu import DeviceSample, TelemetrySample
from finserve.contracts.inference import InferenceRequest
from finserve.contracts.routing import RoutingRequest
from finserve.engines.backend_observations import (
    PhysicalObservation,
    SharedGpuSampler,
    parse_engine_metrics,
)
from finserve.engines.ray_backends import BackendConfiguration, OpenAIEngineReplica, RoutedBackends
from finserve.scheduler.router import NoReplicaAvailable


def exposition(running: str = "1", waiting: str = "2", kv: str = "0.25") -> bytes:
    """Mirror the official gauge names and model label without simulating a GPU."""
    return "".join(
        f'# TYPE vllm:{name} gauge\nvllm:{name}{{model_name="model"}} {value}\n'
        for name, value in (
            ("num_requests_running", running),
            ("num_requests_waiting", waiting),
            ("kv_cache_usage_perc", kv),
        )
    ).encode()


def telemetry(*, error: str | None = None) -> TelemetrySample:
    """One physical device is intentionally shared by all logical worker fixtures."""
    return TelemetrySample(
        epoch_s=time.time(),
        collection_seconds=0.01,
        error=error,
        devices=[]
        if error
        else [
            DeviceSample(
                uuid="GPU-one",
                name="RTX4070",
                utilization_percent=42,
                memory_used_mib=6000,
                memory_total_mib=8000,
            )
        ],
    )


def test_vllm_gauges_keep_kv_and_request_counts_distinct() -> None:
    """A fractional KV occupancy must never become request load or physical memory usage."""
    result = parse_engine_metrics(exposition(), "model")
    assert result.running == 1 and result.waiting == 2 and result.kv_cache_utilization == 0.25


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"x" * 262145,
        exposition().replace(b"model_name", b"wrong_label"),
        exposition().replace(b'"model"', b'"other"'),
        exposition(running="nan"),
        exposition(waiting="inf"),
        exposition(kv="1.1"),
        exposition(running="1.5"),
        exposition(waiting="-1"),
        exposition(running="65537"),
        exposition() + b'vllm:num_requests_running{model_name="model",engine="1"} 1\n',
        exposition().replace(b"gauge", b"untyped"),
    ],
    ids=[
        "missing",
        "oversized",
        "label",
        "model",
        "nan",
        "inf",
        "kv-range",
        "fractional",
        "negative",
        "count-range",
        "duplicate",
        "type",
    ],
)
def test_invalid_engine_observations_fail_closed(body: bytes) -> None:
    """Partial, ambiguous or impossible observations cannot look like idle engines."""
    with pytest.raises(ValueError):
        parse_engine_metrics(body, "model")


async def test_sampler_coalesces_and_drains_after_caller_cancellation() -> None:
    """A cancelled router request cannot orphan the native thread or start duplicate samples."""
    entered, release = threading.Event(), threading.Event()
    calls = 0

    def collect_fixture() -> TelemetrySample:
        """Block a CPU thread to observe ownership across multiple cancelled asyncio callers."""
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(3)
        return telemetry()

    sampler = SharedGpuSampler(collect_fixture)
    first = asyncio.create_task(sampler.get())
    assert await asyncio.to_thread(entered.wait, 2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(sampler.get())
    close = asyncio.create_task(sampler.close())
    await asyncio.sleep(0)
    close.cancel()
    await asyncio.sleep(0)
    assert not close.done() and calls == 1
    release.set()
    observation = await second
    await close
    assert observation.snapshot_fields("GPU-one")["gpu_memory_utilization"] == 0.75
    with pytest.raises(RuntimeError, match="closed"):
        await sampler.get()


@pytest.mark.parametrize("failure", ["missing", "collector", "impossible"])
def test_missing_or_invalid_physical_observation_quarantines(failure: str) -> None:
    """Unknown device state is not zero memory; both colocated workers must be unavailable."""
    sample = telemetry(error="OSError" if failure == "collector" else None)
    if failure == "impossible":
        sample.devices[0].memory_used_mib = 9000
    result = PhysicalObservation(10, sample).snapshot_fields(
        "missing" if failure == "missing" else "GPU-one"
    )
    assert result == {"healthy": False}


async def test_unexpected_collector_error_is_missing_then_recovers() -> None:
    """A failed native observation must not poison the sampler's cached task permanently."""
    calls = 0

    def transient_failure() -> TelemetrySample:
        """Raise once outside ordinary nvidia-smi errors to test the ownership boundary."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("private diagnostic not copied to evidence")
        return telemetry()

    sampler = SharedGpuSampler(transient_failure)
    failed = await sampler.get()
    assert failed.sample.error == "RuntimeError" and not failed.sample.devices
    await asyncio.sleep(1.01)
    assert (await sampler.get()).sample.error is None and calls == 2
    await sampler.close()


async def test_engine_state_requires_health_and_metrics_and_avoids_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy leases overlap backend execution; preserve both raw counts without summing them."""
    worker = OpenAIEngineReplica("a", 4, "model", "http://engine/v1", True)
    await worker.probe.aclose()
    valid = True

    def respond(request: httpx.Request) -> httpx.Response:
        """Exercise real bounded HTTP parsing while controlling only the remote observation."""
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        assert request.url.path == "/metrics"
        return httpx.Response(200, content=exposition() if valid else b"invalid")

    worker.probe = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        worker.admit("lease")
        state = await worker.state()
        assert state["healthy"] and state["ongoing_requests"] == 3
        assert state["engine_running_requests"] == 1 and state["engine_waiting_requests"] == 2
        assert state["reflected_lease_ids"] == ["lease"]
        assert state["kv_cache_utilization"] == 0.25
        valid = False
        monkeypatch.setattr(worker, "_checked_at", None)
        state = await worker.state()
        assert not state["healthy"] and "kv_cache_utilization" not in state
    finally:
        await worker.close()


async def test_shared_device_attaches_once_and_missing_state_blocks_both_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each refresh gives identical physical telemetry to all workers, counted once in status."""
    handles: dict[str, MagicMock] = {}
    for name in ("a", "b"):
        handle = MagicMock()
        handle.state.remote = AsyncMock(
            return_value={
                "replica_id": name,
                "model": "model",
                "capacity": 4,
                "ongoing_requests": 0,
            }
        )
        handles[name] = handle
    routed = RoutedBackends(handles, "model", "adaptive", "GPU-one")
    calls = 0

    def one_gpu() -> TelemetrySample:
        """Record collection count to detect accidental per-worker device duplication."""
        nonlocal calls
        calls += 1
        return telemetry()

    monkeypatch.setattr(routed, "_gpu_sampler", SharedGpuSampler(one_gpu))
    try:
        await routed.refresh_snapshots()
        leases = [routed.router.reserve(RoutingRequest(model="model")) for _ in range(2)]
        assert [lease.decision.replica_id for lease in leases] == ["a", "b"]
        status = await routed.status()
        assert calls == 1 and len(status["physical_gpu_observation"]["devices"]) == 1
        for lease in leases:
            routed.router.release(lease)
        routed.set_enabled("a", False)
        survivor = routed.router.reserve(RoutingRequest(model="model"))
        assert survivor.decision.replica_id == "b"
        routed.record_selection(InferenceRequest(model="model", prompt="private"), survivor)
        evidence = (await routed.status())["recent_decisions"]
        assert len(evidence) == 1 and "private" not in str(evidence)
        routed.router.release(survivor)
        await routed.refresh_snapshots()
        assert not next(row for row in routed.router.snapshots if row.replica_id == "a").healthy
        routed.set_enabled("a", True)
        await routed.refresh_snapshots()
        assert next(row for row in routed.router.snapshots if row.replica_id == "a").healthy
        routed.shared_gpu_uuid = "wrong"
        await routed.refresh_snapshots()
        with pytest.raises(NoReplicaAvailable):
            routed.router.reserve(RoutingRequest(model="model"))
    finally:
        await routed.close()


def test_hardware_aware_configuration_requires_actual_engine_observations() -> None:
    """Explicit opt-in preserves existing CPU proxy deployments without claiming GPU readiness."""
    with pytest.raises(ValueError, match="requires backend"):
        BackendConfiguration(
            model="model", backends={"a": "http://engine/v1"}, shared_gpu_uuid="GPU-one"
        )
