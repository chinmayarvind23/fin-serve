"""Bounded vLLM gauges and one physical-device sampler for colocated engine routing.

KV occupancy is an engine resource gauge, never a substitute for physical VRAM.
The shared sampler reports a device once even when several processes use it.
"""

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, ConfigDict, Field

from finserve.benchmark.gpu import TelemetrySample, collect


class EngineObservation(BaseModel):
    """Keep physical occupancy and logical engine state distinct in routing evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    running: int = Field(ge=0, le=65536, strict=True)
    waiting: int = Field(ge=0, le=65536, strict=True)
    kv_cache_utilization: float = Field(ge=0, le=1)


def parse_engine_metrics(body: bytes, model: str) -> EngineObservation:
    """Require one finite matching-model gauge each; duplicates never silently sum workers."""
    if len(body) > 262144:
        raise ValueError("engine metrics exceed byte limit")
    wanted = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_cache_utilization",
    }
    values: dict[str, float | int] = {}
    for family in text_string_to_metric_families(body.decode("utf-8")):
        for sample in family.samples:
            field = wanted.get(sample.name)
            if field is None:
                continue
            if family.type != "gauge":
                raise ValueError("engine occupancy must be a gauge")
            if sample.labels.get("model_name") != model:
                raise ValueError("engine metrics model mismatch")
            if field in values or not math.isfinite(sample.value):
                raise ValueError("duplicate or nonfinite engine gauge")
            value = sample.value
            if field != "kv_cache_utilization":
                if not value.is_integer():
                    raise ValueError("fractional request count")
                values[field] = int(value)
            else:
                values[field] = value
    return EngineObservation.model_validate(values)


@dataclass(frozen=True)
class PhysicalObservation:
    """Router-local start time conservatively includes collector latency in freshness."""

    started_at: float
    sample: TelemetrySample

    def snapshot_fields(self, uuid: str) -> dict[str, object]:
        """Every colocated worker receives the same device value, or is quarantined together."""
        matching = [device for device in self.sample.devices if device.uuid == uuid]
        if self.sample.error or len(matching) != 1:
            return {"healthy": False}
        device = matching[0]
        if device.memory_used_mib > device.memory_total_mib:
            return {"healthy": False}
        return {
            "gpu_device_id": uuid,
            "gpu_type": device.name,
            "gpu_memory_utilization": device.memory_used_mib / device.memory_total_mib,
            "gpu_observed_at": self.started_at,
        }


class SharedGpuSampler:
    """Coalesce refreshes and retain native collection ownership on cancellation."""

    def __init__(self, collector: Callable[[], TelemetrySample] = collect) -> None:
        """Injection keeps CPU tests independent of device access; production uses nvidia-smi."""
        self._collector = collector
        self._pending: asyncio.Task[PhysicalObservation] | None = None
        self._closed = False

    async def _sample(self) -> PhysicalObservation:
        """The bounded native collector runs outside the event loop and has one explicit owner."""
        started = time.monotonic()
        try:
            sample = await asyncio.to_thread(self._collector)
        except Exception as exc:
            sample = TelemetrySample(
                epoch_s=time.time(),
                collection_seconds=time.monotonic() - started,
                devices=[],
                error=type(exc).__name__,
            )
        return PhysicalObservation(started, sample)

    async def get(self) -> PhysicalObservation:
        """Reuse a one-second observation; cancelled callers cannot launch duplicate collectors."""
        if self._closed:
            raise RuntimeError("GPU sampler closed")
        task = self._pending
        if task is None or (task.done() and time.monotonic() - task.result().started_at >= 1):
            task = self._pending = asyncio.create_task(self._sample())
        return await asyncio.shield(task)

    async def close(self) -> None:
        """Drain the native collection even when the shutdown caller is repeatedly cancelled."""
        self._closed = True
        if self._pending is None:
            return
        while not self._pending.done():
            try:
                await asyncio.shield(self._pending)
            except asyncio.CancelledError:
                continue
        self._pending.result()
