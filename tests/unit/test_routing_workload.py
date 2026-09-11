"""Freeze population and context acceptance before the held-out model runs."""

from collections import Counter

import pytest

from finserve.benchmark.routing_workload import RoutingWorkload, frozen_workload, preflight


def test_frozen_population_balanced_and_roundtrip_identical() -> None:
    """Exact ordered prompt bytes survive persistence and remain independent of policy selection."""
    workload = frozen_workload()
    assert workload.digest() == frozen_workload().digest()
    assert RoutingWorkload.model_validate_json(workload.canonical_bytes()) == workload
    assert len({case.prompt for case in workload.cases}) == 64
    assert set(
        Counter((case.input_class, case.max_tokens) for case in workload.cases).values()
    ) == {16}
    assert workload.policies == ("least_load", "adaptive", "adaptive", "least_load")


def test_context_preflight_preserves_exact_input_and_refuses_overflow() -> None:
    """An infeasible population fails instead of silently becoming a shorter easier workload."""
    workload = frozen_workload()
    counts = preflight(workload, lambda prompt: len(prompt.split()))
    assert len(counts) == 64 and max(counts.values()) < 300
    with pytest.raises(ValueError, match="context-overflow"):
        preflight(workload, lambda _: 1024)
