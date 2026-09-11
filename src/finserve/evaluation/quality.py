"""Versioned deterministic quality rules that never let hard failures average away."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoldenCase(BaseModel):
    """Synthetic task expectations are distinct from reference-model output agreement."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    family: str
    prompt: str
    expected: str
    kind: Literal["exact", "json"] = "exact"


class GoldenSuite(BaseModel):
    """A content hash freezes cases and evaluator settings before model optimization."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    suite_id: str = "synthetic-task-correctness-v1"
    evaluator_version: str = "exact-or-typed-json-v1"
    cases: tuple[GoldenCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> "GoldenSuite":
        """Duplicate identifiers would silently reduce the evaluation denominator."""
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate golden case identifiers")
        return self

    def digest(self) -> str:
        """Include evaluator version and expectations in the immutable suite identity."""
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class QualityConfig(BaseModel):
    """Gate thresholds must be chosen before results, and finite fractions only."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    minimum_parity: float = Field(default=0.992, ge=0, le=1)
    minimum_accuracy: float = Field(default=1.0, ge=0, le=1)


def reject_nonfinite(value: str) -> None:
    """JSON NaN/Infinity are invalid structured answers despite Python's permissive parser."""
    raise ValueError(f"nonfinite JSON constant: {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Duplicate JSON keys are ambiguous, including nested objects; never accept last-key wins."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def canonical_answer(value: str, kind: str) -> str:
    """Normalize JSON key ordering only; wrong field types and extra fields remain incorrect."""
    if kind == "exact":
        return value.strip()
    parsed: object = json.loads(
        value, parse_constant=reject_nonfinite, object_pairs_hook=unique_object
    )
    if not isinstance(parsed, dict):
        raise ValueError("structured answer must be a JSON object")
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)


def evaluate_quality(
    suite: GoldenSuite,
    reference: dict[str, str],
    candidate: dict[str, str],
    config: QualityConfig | None = None,
) -> dict[str, object]:
    """Parity measures agreement; accuracy measures truth. Missing/invalid cases always fail."""
    config = config or QualityConfig()
    rows: list[dict[str, object]] = []
    hard_failures: list[str] = []
    ids = {case.case_id for case in suite.cases}
    for label, results in (("reference", reference), ("candidate", candidate)):
        if set(results) != ids:
            hard_failures.append(f"{label}_case_set_mismatch")
    for case in suite.cases:
        correct = False
        reference_correct = False
        parity = False
        try:
            expected = canonical_answer(case.expected, case.kind)
            ref = canonical_answer(reference[case.case_id], case.kind)
            cand = canonical_answer(candidate[case.case_id], case.kind)
            parity, correct, reference_correct = cand == ref, cand == expected, ref == expected
        except (KeyError, ValueError, TypeError):
            hard_failures.append(f"{case.case_id}:missing_or_invalid_output")
        if not correct:
            # These tiny deterministic finance cases require exact numeric/evidence fidelity.
            hard_failures.append(f"{case.case_id}:incorrect_answer")
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "parity": parity,
                "candidate_correct": correct,
                "reference_correct": reference_correct,
            }
        )
    count = len(rows)
    parity_score = sum(row["parity"] is True for row in rows) / count
    accuracy = sum(row["candidate_correct"] is True for row in rows) / count
    return {
        "suite_id": suite.suite_id,
        "suite_hash": suite.digest(),
        "evaluator_version": suite.evaluator_version,
        "configuration": config.model_dump(),
        "case_count": count,
        "parity": parity_score,
        "candidate_accuracy": accuracy,
        "reference_accuracy": sum(row["reference_correct"] is True for row in rows) / count,
        "hard_failures": hard_failures,
        "cases": rows,
        "passed": not hard_failures
        and parity_score >= config.minimum_parity
        and accuracy >= config.minimum_accuracy,
        "scope": "small synthetic deterministic correctness suite; not general model quality",
    }


def default_suite() -> GoldenSuite:
    """Keep finance and nonfinancial cases original, reproducible, and independent of engine."""
    return GoldenSuite(
        cases=(
            GoldenCase(
                case_id="margin",
                family="SEC_QA",
                prompt="Revenue 500; operating income 75. Return operating margin as a decimal.",
                expected="0.15",
            ),
            GoldenCase(
                case_id="table",
                family="FINANCIAL_TABLE_EXTRACTION",
                prompt="Return JSON with revenue 820 and capex 41, and no extra keys.",
                expected='{"revenue":820,"capex":41}',
                kind="json",
            ),
            GoldenCase(
                case_id="general",
                family="GENERAL",
                prompt="Return only the sum of 2 and 3.",
                expected="5",
            ),
        )
    )


def evaluate_output_parity(reference: list[str], candidate: list[str]) -> dict[str, object]:
    """Serving equivalence is reported separately; matching wrong outputs is not task accuracy."""
    if not reference or len(reference) != len(candidate):
        raise ValueError("parity requires nonempty equally sized output sequences")
    matches = sum(left == right for left, right in zip(reference, candidate, strict=True))
    return {
        "evaluator_version": "byte-exact-serving-parity-v1",
        "cases": len(reference),
        "matches": matches,
        "parity": matches / len(reference),
        "scope": "exact serving output agreement only; task correctness unmeasured",
    }
