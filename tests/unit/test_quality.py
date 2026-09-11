"""Counterexamples prove that parity alone cannot hide degraded task quality."""

import pytest
from pydantic import ValidationError

from finserve.evaluation.quality import (
    GoldenSuite,
    QualityConfig,
    default_suite,
    evaluate_output_parity,
    evaluate_quality,
)


def answers() -> dict[str, str]:
    """Return golden outputs independently of any serving engine's implementation."""
    return {"margin": "0.15", "table": '{"capex":41,"revenue":820}', "general": "5"}


def test_exact_correct_candidate_passes() -> None:
    """JSON field order is irrelevant but the expected fields and values remain exact."""
    report = evaluate_quality(default_suite(), answers(), answers())
    assert report["passed"] is True
    assert report["parity"] == 1
    assert report["candidate_accuracy"] == 1


def test_equal_wrong_models_have_parity_but_fail_quality() -> None:
    """A reference model's incorrect answer cannot launder candidate accuracy."""
    wrong = {**answers(), "margin": "0.20"}
    report = evaluate_quality(default_suite(), wrong, wrong)
    assert report["parity"] == 1
    assert report["passed"] is False


@pytest.mark.parametrize(
    "bad",
    [
        '{"revenue":NaN,"capex":41}',
        '{"revenue":"820","capex":41}',
        '{"revenue":820,"capex":41,"invented":1}',
        '{"revenue":123,"revenue":820,"capex":41}',
        '{"revenue":{"value":1,"value":820},"capex":41}',
        "[]",
    ],
)
def test_hard_failures_cannot_average_away(bad: str) -> None:
    """Even permissive fraction thresholds cannot override structured/numeric corruption."""
    report = evaluate_quality(
        default_suite(),
        answers(),
        {**answers(), "table": bad},
        QualityConfig(minimum_parity=0, minimum_accuracy=0),
    )
    assert report["passed"] is False
    assert report["hard_failures"]


def test_missing_extra_and_duplicate_cases_fail() -> None:
    """Require exact case identity, preventing convenient dropping or denominator inflation."""
    assert evaluate_quality(default_suite(), answers(), {})["passed"] is False
    assert (
        evaluate_quality(default_suite(), answers(), {**answers(), "extra": "5"})["passed"] is False
    )
    case = default_suite().cases[0]
    with pytest.raises(ValidationError):
        GoldenSuite(cases=(case, case))


def test_serving_parity_scope_and_length_guard() -> None:
    """Output equivalence is explicit and never represented as general model accuracy."""
    result = evaluate_output_parity(["wrong", "ok"], ["wrong", "different"])
    assert result["parity"] == 0.5
    assert "correctness unmeasured" in str(result["scope"])
    with pytest.raises(ValueError):
        evaluate_output_parity(["one"], [])
