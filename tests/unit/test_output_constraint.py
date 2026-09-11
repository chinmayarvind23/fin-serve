"""Output syntax permits multiple answers while rejecting ambiguous and unbounded contracts."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from finserve.contracts.inference import ChatMessage, ChatRequest, InferenceRequest
from finserve.contracts.output_constraint import ObjectField, OutputConstraint


@pytest.mark.parametrize(
    ("constraint", "accepted", "rejected"),
    [
        (OutputConstraint(kind="integer"), ["0", "17", "-2048"], ["-0", "+1", "01", "1.0", " 2"]),
        (
            OutputConstraint(kind="decimal"),
            ["0", "1", "-7", "0.2", "-0.003", "17.04"],
            ["-0", "0.20", "1.0", ".2", "2e3", "NaN", "1\n"],
        ),
        (
            OutputConstraint(kind="decimal", decimal_places=2),
            ["0.20", "-0.03", "14.00"],
            ["0.2", "1", "1.001"],
        ),
        (OutputConstraint(kind="decimal", decimal_places=0), ["0", "-3", "42"], ["2.0"]),
        (OutputConstraint(kind="yes_no"), ["yes", "no"], ["Yes", "No", "no.", "yes\n"]),
        (
            OutputConstraint(kind="lowercase_word"),
            ["blue", "green", "red", "zebra"],
            ["Blue", "light blue", "x" * 65],
        ),
    ],
)
def test_lexical_policy_preserves_alternative_answers(
    constraint: OutputConstraint, accepted: list[str], rejected: list[str]
) -> None:
    """Broad syntax is independent of task truth; exact precision overrides canonical decimals."""
    assert all(constraint.accepts(text) for text in accepted)
    assert not any(constraint.accepts(text) for text in rejected)


def test_json_types_whitespace_duplicates_and_resource_bounds() -> None:
    """Allow whitespace; reject duplicate fields, bool-as-number and nonfinite values."""
    constraint = OutputConstraint(
        fields=(ObjectField(name="profit", type="number"),), kind="json_object"
    )
    for value in (-14, 0, 32.25):
        assert constraint.accepts(json.dumps({"profit": value}, indent=2))
    for text in (
        '{"profit": true}',
        '{"profit": NaN}',
        '{"profit": 1e999}',
        '{"profit": 1, "profit": 2}',
        '{"profit": 1, "extra": 2}',
        '```json\n{"profit": 1}\n```',
        '{"profit":',
        "[]",
        "[" * 5000,
        '{"profit": 1}' + " " * 131072,
    ):
        assert not constraint.accepts(text)
    schema = constraint.vllm_parameters()["json"]
    assert schema == {
        "type": "object",
        "properties": {"profit": {"type": "number"}},
        "required": ["profit"],
        "additionalProperties": False,
    }
    mixed = OutputConstraint(
        kind="json_object",
        fields=(
            ObjectField(name="count", type="integer"),
            ObjectField(name="label", type="string"),
            ObjectField(name="active", type="boolean"),
        ),
    )
    assert mixed.accepts('{"count": 2, "label": "ok", "active": false}')
    assert not mixed.accepts('{"count": 2.0, "label": "ok", "active": false}')
    assert not mixed.accepts(json.dumps({"count": 2, "label": "a" * 4097, "active": False}))


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "yes_no", "choice": ["yes"]},
        {"kind": "decimal", "expected": "0.2"},
        {"kind": "integer", "regex": "42"},
        {"kind": "json_object", "fields": []},
        {"kind": "yes_no", "decimal_places": 1},
        {"kind": "decimal", "decimal_places": True},
        {"kind": "decimal", "decimal_places": 13},
        {"kind": "json_object", "fields": [{"name": "x", "type": "integer", "const": 1}]},
        {"kind": "json_object", "fields": [{"name": "x", "type": "object"}]},
        {"kind": "json_object", "fields": [{"name": "x", "type": "integer"}] * 2},
        {
            "kind": "json_object",
            "fields": [{"name": f"x{i}", "type": "integer"} for i in range(17)],
        },
    ],
)
def test_invalid_or_answer_bearing_contracts_are_rejected(value: dict[str, object]) -> None:
    """The public shape contract cannot smuggle a schema language, single answer or unused knob."""
    with pytest.raises(ValidationError):
        OutputConstraint.model_validate(value)


def test_chat_roundtrip_and_legacy_request_encoding() -> None:
    """Internal relays retain explicit constraints; legacy unconstrained serialization is stable."""
    constraint = OutputConstraint(kind="yes_no")
    chat = ChatRequest(
        messages=[ChatMessage(role="user", content="Is it raining?")], output_constraint=constraint
    )
    internal = chat.to_inference()
    assert (
        InferenceRequest.model_validate_json(internal.model_dump_json()).output_constraint
        == constraint
    )
    assert (
        "output_constraint"
        not in InferenceRequest(prompt="hello", request_id="stable").model_dump()
    )
    assert "output_constraint" not in ChatRequest(messages=chat.messages).model_dump()
    # Captured from source 65669c6 before the optional field was added.
    legacy = InferenceRequest(prompt="hello", request_id="stable").model_dump_json().encode()
    assert hashlib.sha256(legacy).hexdigest() == (
        "b54fd47ed2d8888c507e96cb46127e32c5195edfd8b443e0192a6105e11de6cc"
    )
    assert (
        chat.model_dump_json()
        != chat.model_copy(update={"output_constraint": None}).model_dump_json()
    )
    with pytest.raises(ValidationError):
        constraint.kind = "integer"
