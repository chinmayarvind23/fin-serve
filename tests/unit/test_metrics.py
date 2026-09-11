"""Protect denominators, clock semantics and billing math against optimistic reporting."""

import math

import pytest
from pydantic import ValidationError

from finserve.benchmark.cost import CostInput, cost_reduction
from finserve.benchmark.metrics import RequestRecord, percentile, summarize
from finserve.benchmark.runner import RunConfig
from finserve.benchmark.workload import default_workload


def sample(index: int, success: bool = True, tokens: int | None = 20) -> RequestRecord:
    """Use independently chosen timings so the test detects formula/denominator changes."""
    return RequestRecord(
        logical_id=index,
        case_id="a",
        family="GENERAL",
        scheduled_s=1,
        send_s=2,
        first_content_s=2.25,
        complete_s=3,
        success=success,
        generated_tokens=tokens,
        error=None if success else "timeout",
    )


def test_metrics_preserve_failures_and_distinguish_server_ttft() -> None:
    """Two successful responses among three offered yield 2/3 reliability, not 100%."""
    result = summarize([sample(0), sample(1), sample(2, False)], 4)
    assert result["success_rate"] == pytest.approx(2 / 3)
    assert result["requests_per_second"] == 0.5
    assert result["tokens_per_second"] == 10
    assert result["client_ttft_median_s"] == 0.25
    assert result["server_ttft_median_s"] is None
    assert result["e2e_p95_s"] == 1
    assert result["scheduled_to_complete_p95_s"] == 2
    assert result["scheduled_to_send_p95_s"] == 1


def test_missing_usage_does_not_become_a_token_estimate() -> None:
    """A partial usage population cannot support a headline token throughput claim."""
    assert summarize([sample(0), sample(1, tokens=None)], 2)["tokens_per_second"] is None


def test_warmup_is_excluded() -> None:
    """Warmup records remain present but cannot inflate measured success or throughput."""
    warmup = sample(0).model_copy(update={"phase": "warmup"})
    assert summarize([warmup, sample(0)], 2)["offered_requests"] == 1


def test_percentile_interpolates_and_empty_population_is_unknown() -> None:
    """The declared interpolation rule is independently checkable on a two-point population."""
    assert percentile([1, 3], 0.95) == pytest.approx(2.9)
    assert percentile([], 0.5) is None
    assert percentile([3], 0.5) == 3


@pytest.mark.parametrize("value", [math.nan, math.inf, -1])
def test_nonfinite_or_negative_inputs_rejected(value: float) -> None:
    """Invalid numbers must fail before reaching JSON summaries or scheduling sleeps."""
    with pytest.raises(ValueError):
        summarize([], value)
    with pytest.raises(ValueError):
        percentile([value], 0.5)
    with pytest.raises(ValidationError):
        RunConfig(rate=value)


def test_clock_and_duplicate_id_invariants() -> None:
    """An impossible first-content clock and repeated logical ID both invalidate evidence."""
    with pytest.raises(ValidationError):
        RequestRecord.model_validate({**sample(0).model_dump(), "first_content_s": 1})
    with pytest.raises(ValueError):
        summarize([sample(0), sample(0)], 1)


@pytest.mark.parametrize(
    "updates",
    [
        {"complete_s": 0},
        {"send_s": 4},
        {"send_s": None, "first_content_s": None},
        {"error": "failed"},
    ],
)
def test_impossible_success_records_fail(updates: dict[str, object]) -> None:
    """Exercise each independent invariant so changing a guard cannot silently inflate metrics."""
    with pytest.raises(ValidationError):
        RequestRecord.model_validate({**sample(0).model_dump(), **updates})


@pytest.mark.parametrize("quantile", [float("nan"), -0.01, 1.01])
def test_invalid_quantiles_fail(quantile: float) -> None:
    """Percentiles outside their mathematical domain must not clamp to convenient values."""
    with pytest.raises(ValueError):
        percentile([1, 2], quantile)


def test_workload_hash_freezes_generation_budget() -> None:
    """Changing output length is changing the workload, even if prompt text is identical."""
    assert default_workload(8).digest() == default_workload(8).digest()
    assert default_workload(8).digest() != default_workload(9).digest()


def test_cost_requires_quality_and_real_token_denominator() -> None:
    """Use billing inputs with a simple independently computed 50% reduction."""
    baseline = CostInput(
        instance_hour_price_usd=2,
        billed_instance_hours=1,
        generated_tokens=1_000_000,
        pricing_source="test fixture",
        pricing_date="2026-09-11",
        hardware="test GPU",
    )
    candidate = baseline.model_copy(update={"billed_instance_hours": 0.5})
    assert baseline.per_million_tokens() == 2
    assert cost_reduction(baseline, candidate, True) == 0.5
    with pytest.raises(ValueError):
        cost_reduction(baseline, candidate, False)


@pytest.mark.parametrize("price,hours", [(1e308, 1e308), (1e-200, 1e-200)])
def test_cost_overflow_and_underflow_fail(price: float, hours: float) -> None:
    """Finite individual inputs can still produce unusable computed cost denominators."""
    value = CostInput(
        instance_hour_price_usd=price,
        billed_instance_hours=hours,
        generated_tokens=1,
        pricing_source="fixture",
        pricing_date="2026-09-11",
        hardware="fixture",
    )
    with pytest.raises(ValueError):
        value.per_million_tokens()


def test_cost_reduction_overflow_fails() -> None:
    """Two individually finite costs can still produce an infinite ratio."""
    baseline = CostInput(
        instance_hour_price_usd=1e-200,
        billed_instance_hours=1e-100,
        generated_tokens=1,
        pricing_source="fixture",
        pricing_date="2026-09-11",
        hardware="fixture",
    )
    candidate = baseline.model_copy(
        update={"instance_hour_price_usd": 1e200, "billed_instance_hours": 1e100}
    )
    with pytest.raises(ValueError):
        cost_reduction(baseline, candidate, True)


@pytest.mark.parametrize("updates", [{"status_code": 503}, {"first_content_s": None}])
def test_success_requires_http_and_token_consistency(updates: dict[str, object]) -> None:
    """Edited raw evidence cannot call HTTP failures or unobserved positive tokens successful."""
    with pytest.raises(ValidationError):
        RequestRecord.model_validate({**sample(0).model_dump(), **updates})
