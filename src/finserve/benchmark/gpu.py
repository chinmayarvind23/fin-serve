"""Raw device telemetry preserves missing samples instead of inventing utilization."""

import argparse
import json
import math
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeviceSample(BaseModel):
    """UUID identifies a device across process restarts and visible-device remapping."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    uuid: str = Field(min_length=1)
    name: str
    utilization_percent: float = Field(ge=0, le=100)
    memory_used_mib: float = Field(ge=0)
    memory_total_mib: float = Field(gt=0)


class TelemetrySample(BaseModel):
    """A failed collection is raw evidence, with no assumed zero utilization."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    epoch_s: float = Field(ge=0)
    collection_seconds: float = Field(ge=0)
    devices: list[DeviceSample]
    error: str | None = None

    @model_validator(mode="after")
    def unique_devices(self) -> "TelemetrySample":
        """Loaded evidence must meet the same identity rules as command output."""
        if len({device.uuid for device in self.devices}) != len(self.devices):
            raise ValueError("duplicate GPU identity")
        if not self.devices and self.error is None:
            raise ValueError("empty telemetry must declare a collection failure")
        return self


def parse_devices(output: str) -> list[DeviceSample]:
    """Reject malformed/N/A fields; consumer must distinguish missing telemetry from idle GPU."""
    devices: list[DeviceSample] = []
    for line in output.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError("invalid nvidia-smi field count")
        devices.append(
            DeviceSample(
                uuid=fields[0],
                name=fields[1],
                utilization_percent=float(fields[2]),
                memory_used_mib=float(fields[3]),
                memory_total_mib=float(fields[4]),
            )
        )
    if not devices or len({device.uuid for device in devices}) != len(devices):
        raise ValueError("missing or duplicate GPU identity")
    return devices


def collect() -> TelemetrySample:
    """A bounded subprocess records collection latency, relevant for short benchmark windows."""
    start = time.perf_counter()
    devices: list[DeviceSample] = []
    error = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        devices = parse_devices(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        error = type(exc).__name__
    return TelemetrySample(
        epoch_s=time.time(),
        collection_seconds=time.perf_counter() - start,
        devices=devices,
        error=error,
    )


def aggregate(
    samples: Sequence[TelemetrySample], start: float, end: float, max_gap: float = 2.5
) -> dict[str, object]:
    """Integrate sample-held utilization across a declared window, capping stale observations.

    GPU inventory must stay fixed throughout the supplied recording. GPUs receive
    equal weight within a sample. Time receives interval weight across
    samples. Missing devices, collection failures, or long gaps lower coverage; no
    utilization claim is returned below 95% coverage. Epoch windows require clock
    synchronization when collection and load generation run on different hosts.
    """
    if any(not math.isfinite(value) for value in (start, end, max_gap)):
        raise ValueError("finite window required")
    if end <= start or max_gap <= 0:
        raise ValueError("positive window and freshness required")
    if any(
        right.epoch_s <= left.epoch_s for left, right in zip(samples, samples[1:], strict=False)
    ):
        raise ValueError("samples must be strictly ordered")
    expected = {device.uuid for sample in samples for device in sample.devices}
    area, covered = 0.0, 0.0
    for index, sample in enumerate(samples):
        stop = samples[index + 1].epoch_s if index + 1 < len(samples) else end
        lower = max(start, sample.epoch_s)
        upper = min(end, stop, sample.epoch_s + max_gap)
        if sample.error or {device.uuid for device in sample.devices} != expected or not expected:
            continue
        duration = max(0, upper - lower)
        mean = sum(device.utilization_percent for device in sample.devices) / len(sample.devices)
        area += duration * mean
        covered += duration
    coverage = covered / (end - start)
    return {
        "window_start_epoch_s": start,
        "window_end_epoch_s": end,
        "coverage": coverage,
        "device_ids": sorted(expected),
        "average_gpu_utilization_percent": area / covered if coverage >= 0.95 else None,
        "method": "time-weighted sample hold; equal-device mean; max-gap capped",
        "max_gap_seconds": max_gap,
        "minimum_coverage": 0.95,
    }


def main() -> None:
    """Write a unique raw JSONL file; callers retain warmup and select windows afterward."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (args.duration, args.interval)):
        parser.error("duration and interval must be finite and positive")
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        parser.error("telemetry must be outside the source repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    start, index = time.perf_counter(), 0
    with output.open("x", encoding="utf-8") as stream:
        while time.perf_counter() - start < args.duration:
            stream.write(json.dumps(collect().model_dump(), allow_nan=False) + "\n")
            stream.flush()
            index += 1
            time.sleep(max(0, start + index * args.interval - time.perf_counter()))


if __name__ == "__main__":
    main()
