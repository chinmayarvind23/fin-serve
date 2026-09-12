"""Bounded scientific and flat JSON shapes preserve answer diversity and legacy identity."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from finserve.contracts.inference import InferenceRequest
from finserve.contracts.output_constraint import ObjectField, OutputConstraint


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"decimal_places": 2},
        {"exponent_digits": 2},
        {"decimal_places": -1, "exponent_digits": 2},
        {"decimal_places": 13, "exponent_digits": 2},
        {"decimal_places": True, "exponent_digits": 2},
        {"decimal_places": 2, "exponent_digits": 0},
        {"decimal_places": 2, "exponent_digits": 5},
        {"decimal_places": 2, "exponent_digits": True},
        {"decimal_places": 2, "exponent_digits": "2"},
        {"decimal_places": 2, "exponent_digits": 2, "regex": "1.00e+02"},
        {"decimal_places": 2, "exponent_digits": 2, "expected": "1.00e+02"},
    ],
)
def test_scientific_requires_bounded_precision(options: dict[str, object]) -> None:
    """Every scientific width must be explicit, bounded and independent of task truth."""
    with pytest.raises(ValidationError):
        OutputConstraint.model_validate({"kind": "scientific", **options})


@pytest.mark.parametrize("kind", ["integer", "decimal", "yes_no", "lowercase_word", "json_object"])
def test_exponent_width_cannot_be_silently_ignored(kind: str) -> None:
    """Options for a different shape must fail instead of weakening caller intent."""
    value: dict[str, object] = {"kind": kind, "exponent_digits": 2}
    if kind == "json_object":
        value["fields"] = [{"name": "x", "type": "null"}]
    with pytest.raises(ValidationError):
        OutputConstraint.model_validate(value)


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("1.23e+04", True),
        ("-9.87e-02", True),
        ("0.00e+00", True),
        ("4.56e+99", True),
        ("2.00e-00", True),
        ("12.34e+04", False),
        ("01.23e+04", False),
        ("+1.23e+04", False),
        ("1.23E+04", False),
        ("1.23e04", False),
        ("1.2e+04", False),
        ("1.230e+04", False),
        ("1.23e+4", False),
        ("1.23e+004", False),
        ("1.23e+", False),
        (" 1.23e+04", False),
        ("1.23e+04\n", False),
        ("The answer is 1.23e+04", False),
        ("```1.23e+04```", False),
        ("NaN", False),
    ],
)
def test_scientific_checks_unrepaired_final_text(text: str, accepted: bool) -> None:
    """Exact lexical form still permits unrelated positive, negative and zero answers."""
    constraint = OutputConstraint.model_validate(
        {"kind": "scientific", "decimal_places": 2, "exponent_digits": 2}
    )
    assert constraint.accepts(text) is accepted


@pytest.mark.parametrize(("places", "width"), [(0, 1), (0, 4), (12, 1), (12, 4)])
def test_scientific_precision_boundaries(places: int, width: int) -> None:
    """Zero precision removes the decimal point; maximum widths remain usable."""
    constraint = OutputConstraint.model_validate(
        {"kind": "scientific", "decimal_places": places, "exponent_digits": width}
    )
    fraction = "." + "5" * places if places else ""
    for mantissa in ("0", "3", "-7"):
        for sign in ("+", "-"):
            assert constraint.accepts(f"{mantissa}{fraction}e{sign}{'1' * width}")
    assert not constraint.accepts(f"3{fraction}e+{'1' * (width + 1)}")


@pytest.mark.parametrize(
    "field",
    [
        {"type": "array"},
        {"type": "array", "item_type": None},
        {"type": "array", "item_type": "array"},
        {"type": "array", "item_type": "object"},
        {"type": "array", "item_type": "integer", "max_items": 1},
        {"type": "array", "item_type": "integer", "const": [42]},
        *[
            {"type": kind, "item_type": "string"}
            for kind in ("integer", "number", "string", "boolean", "null")
        ],
    ],
)
def test_array_schema_cannot_nest_or_fix_an_answer(field: dict[str, object]) -> None:
    """Only flat primitive arrays are expressible; callers cannot constrain answer content."""
    with pytest.raises(ValidationError):
        ObjectField.model_validate({"name": "values", **field})


@pytest.mark.parametrize(
    ("item_type", "items", "invalid_items"),
    [
        ("integer", [-9, 0, 42], [True, 1.0, None, "1"]),
        ("number", [-9, 0, 3.25], [True, None, "1", float("inf"), float("nan")]),
        ("string", ["", "alpha", "different answer"], [1, True, None, "x" * 4097]),
        ("boolean", [False, True], [0, 1, "true", None]),
        ("null", [None], [0, False, "null"]),
    ],
)
def test_primitive_arrays_enforce_types_and_resource_boundaries(
    item_type: str, items: list[object], invalid_items: list[object]
) -> None:
    """Empty through 64-item arrays are valid; mixed types and nested values are rejected."""
    constraint = OutputConstraint.model_validate(
        {
            "kind": "json_object",
            "fields": [{"name": "values", "type": "array", "item_type": item_type}],
        }
    )
    for values in ([], items, [items[0]] * 64):
        assert constraint.accepts(json.dumps({"values": values}, indent=2))
    rejected_arrays: list[list[object]] = [
        [items[0]] * 65,
        *[[value] for value in invalid_items],
        [[]],
        [{}],
    ]
    for values in rejected_arrays:
        assert not constraint.accepts(json.dumps({"values": values}))
    rejected_scalars: list[object] = [None, {}, 1, "array"]
    for value in rejected_scalars:
        assert not constraint.accepts(json.dumps({"values": value}))


def test_null_and_array_schema_matches_runtime_policy() -> None:
    """Engine schemas bound arrays and string items exactly as raw-final validation does."""
    constraint = OutputConstraint.model_validate(
        {
            "kind": "json_object",
            "fields": [
                {"name": "missing", "type": "null"},
                *[
                    {"name": kind, "type": "array", "item_type": kind}
                    for kind in ("integer", "number", "string", "boolean", "null")
                ],
            ],
        }
    )
    properties: dict[str, object] = {"missing": {"type": "null"}}
    for kind in ("integer", "number", "string", "boolean", "null"):
        properties[kind] = {
            "type": "array",
            "maxItems": 64,
            "items": {"type": kind, **({"maxLength": 4096} if kind == "string" else {})},
        }
    assert constraint.vllm_parameters() == {
        "json": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    }
    payload = {
        "missing": None,
        "integer": [1],
        "number": [2.5],
        "string": ["other"],
        "boolean": [False],
        "null": [None],
    }
    assert constraint.accepts(json.dumps(payload))
    assert not constraint.accepts(json.dumps({**payload, "missing": False}))
    assert not constraint.accepts(json.dumps({**payload, "extra": None}))
    assert not constraint.accepts(json.dumps(payload)[:-1] + ', "missing": null}')
    del payload["missing"]
    assert not constraint.accepts(json.dumps(payload))


def test_legacy_field_and_constraint_encoding_stays_identical() -> None:
    """Absent new knobs must not invalidate already frozen requests or workload hashes."""
    field = ObjectField(name="profit", type="number")
    assert field.model_dump() == {"name": "profit", "type": "number"}
    constraint = OutputConstraint(kind="json_object", fields=(field,))
    assert constraint.model_dump() == {
        "kind": "json_object",
        "fields": ({"name": "profit", "type": "number"},),
        "decimal_places": None,
    }
    assert hashlib.sha256(constraint.model_dump_json().encode()).hexdigest() == (
        "a568d0ed810144852251e2c2aa225ace20fc41f25150f8fde3971ad842dad099"
    )
    request = InferenceRequest(prompt="profit?", request_id="stable", output_constraint=constraint)
    assert "item_type" not in request.model_dump_json()
    assert "exponent_digits" not in request.model_dump_json()


def test_new_shape_options_survive_internal_request_roundtrip() -> None:
    """New options must survive relaying rather than disappear with legacy-compatible omission."""
    for value in (
        {"kind": "scientific", "decimal_places": 0, "exponent_digits": 4},
        {
            "kind": "json_object",
            "fields": [{"name": "values", "type": "array", "item_type": "null"}],
        },
    ):
        constraint = OutputConstraint.model_validate(value)
        request = InferenceRequest(prompt="answer", output_constraint=constraint)
        assert InferenceRequest.model_validate_json(request.model_dump_json()) == request
