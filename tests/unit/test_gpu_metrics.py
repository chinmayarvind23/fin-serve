"""Exercise coverage and aggregation boundaries with no physical GPU dependency."""

from unittest.mock import patch

import pytest

from finserve.benchmark.gpu import TelemetrySample, aggregate, collect, parse_devices


def sample(at: float, load: float, error: str | None = None) -> TelemetrySample:
    """Explicit times make sample weighting expectations independent of scheduling jitter."""
    return TelemetrySample(
        epoch_s=at,
        collection_seconds=0.01,
        error=error,
        devices=parse_devices(f"GPU-A, Example, {load}, 512, 8192"),
    )


def test_weighted_utilization_and_missing_samples() -> None:
    """Longer intervals carry proportionally more weight; failures cannot look like idle time."""
    result = aggregate([sample(10, 20), sample(11, 80)], 10, 13)
    assert result["average_gpu_utilization_percent"] == 60
    assert result["coverage"] == 1
    missing = aggregate([sample(10, 20), sample(11, 80, "Failed")], 10, 13)
    assert missing["average_gpu_utilization_percent"] is None
    assert missing["coverage"] == 1 / 3
    assert aggregate([], 10, 13)["average_gpu_utilization_percent"] is None
    assert aggregate([sample(10, 20)], 10, 20)["coverage"] == 0.25


@pytest.mark.parametrize("raw", ["", "GPU-A, x, N/A, 0, 1", "GPU-A,x,101,0,1", "x,y"])
def test_malformed_device_samples(raw: str) -> None:
    """Missing and invalid values are failures, never converted to plausible zero values."""
    with pytest.raises(ValueError):
        parse_devices(raw)


@pytest.mark.parametrize("start,end,gap", [(1, 1, 1), (2, 1, 1), (1, 2, 0), (1, float("nan"), 1)])
def test_invalid_window(start: float, end: float, gap: float) -> None:
    """Nonfinite/empty windows must not generate utilization claims."""
    with pytest.raises(ValueError):
        aggregate([], start, end, gap)


def test_order_and_duplicate_device_identity() -> None:
    """Duplicate samples cannot silently double counted GPU time."""
    with pytest.raises(ValueError):
        aggregate([sample(2, 50), sample(1, 50)], 1, 3)
    with pytest.raises(ValueError):
        parse_devices("GPU-A,x,50,0,1\nGPU-A,x,50,0,1")


def test_loaded_telemetry_and_collection_timestamp() -> None:
    """A slow query cannot establish coverage before its observation completes."""
    device = parse_devices("GPU-A,x,50,0,1")[0]
    with pytest.raises(ValueError):
        TelemetrySample(epoch_s=1, collection_seconds=0, devices=[device, device])
    with pytest.raises(ValueError):
        TelemetrySample(epoch_s=1, collection_seconds=0, devices=[])
    with patch("finserve.benchmark.gpu.subprocess.run") as run:
        run.return_value.stdout = "GPU-A,x,50,0,1"
        with patch("finserve.benchmark.gpu.time.perf_counter", side_effect=[10.0, 12.0]):
            with patch("finserve.benchmark.gpu.time.time", return_value=102.0):
                observed = collect()
    assert observed.epoch_s == 102
    assert observed.collection_seconds == 2
    assert aggregate([observed], 100, 102)["coverage"] == 0
