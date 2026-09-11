"""Bound caller-selected output shape without accepting expected answers or arbitrary grammars."""

import json
import math
import re
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAXIMUM_CONSTRAINED_OUTPUT_BYTES = 131072


def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Duplicate JSON keys cannot hide an earlier value when checking the completed shape."""
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON property")
        result[name] = value
    return result


def matches_type(value: object, kind: str) -> bool:
    """JSON booleans are not numbers; reject nonfinite values and oversized string values."""
    if kind == "boolean":
        return type(value) is bool
    if kind == "string":
        return isinstance(value, str) and len(value) <= 4096
    if kind == "integer":
        return type(value) is int
    return type(value) is int or (type(value) is float and math.isfinite(value))


class ObjectField(BaseModel):
    """A flat property names a requested value type, with no defaults, constants or alternatives."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: Literal["integer", "number", "string", "boolean"]


class OutputConstraint(BaseModel):
    """Versioned lexical policy constrains syntax; correctness remains the evaluator's decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["integer", "decimal", "yes_no", "lowercase_word", "json_object"]
    fields: tuple[ObjectField, ...] = Field(default=(), max_length=16)
    decimal_places: int | None = Field(default=None, ge=0, le=12, strict=True)

    @model_validator(mode="after")
    def coherent_shape(self) -> Self:
        """Reject unused options and duplicate properties instead of hiding a conflicting schema."""
        if bool(self.fields) != (self.kind == "json_object"):
            raise ValueError("only json_object requires fields")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("JSON field names must be unique")
        if self.decimal_places is not None and self.kind != "decimal":
            raise ValueError("decimal_places requires decimal output")
        return self

    def vllm_parameters(self) -> dict[str, object]:
        """Emit vLLM 0.29 parameters from fixed grammar templates, never caller regex or answers.

        Numeric text permits up to 128 integral and fractional digits. Canonical decimals omit
        redundant zeros unless the caller explicitly requests precision. JSON whitespace remains
        permitted by the engine; typed JSON correctness does not require compact serialization.
        """
        if self.kind == "yes_no":
            return {"choice": ["yes", "no"]}
        if self.kind == "json_object":
            properties: dict[str, object] = {
                field.name: {
                    "type": field.type,
                    **({"maxLength": 4096} if field.type == "string" else {}),
                }
                for field in self.fields
            }
            return {
                "json": {
                    "type": "object",
                    "properties": properties,
                    "required": [field.name for field in self.fields],
                    "additionalProperties": False,
                }
            }
        integer = r"(0|-?[1-9][0-9]{0,127})"
        if self.kind == "integer" or self.decimal_places == 0:
            pattern = integer
        elif self.kind == "lowercase_word":
            pattern = r"[a-z]{1,64}"
        elif self.decimal_places is not None:
            pattern = rf"-?(0|[1-9][0-9]{{0,127}})\.[0-9]{{{self.decimal_places}}}"
        else:
            pattern = r"(0|-?([1-9][0-9]{0,127}(\.[0-9]{0,127}[1-9])?|0\.[0-9]{0,127}[1-9]))"
        return {"regex": pattern}

    def accepts(self, text: str) -> bool:
        """Validate final syntax without repairing text or comparing it to an expected answer.

        A length-limited stream may end inside a valid grammar prefix. Such output must retain
        its partial text and fail before the adapter publishes a successful terminal event.
        """
        if len(text.encode("utf-8")) > MAXIMUM_CONSTRAINED_OUTPUT_BYTES:
            return False
        if self.kind == "yes_no":
            return text in {"yes", "no"}
        if self.kind != "json_object":
            return re.fullmatch(str(self.vllm_parameters()["regex"]), text) is not None
        try:
            value: object = json.loads(text, object_pairs_hook=unique_fields)
        except (ValueError, RecursionError):
            return False
        if not isinstance(value, dict):
            return False
        fields = cast(dict[str, object], value)
        return set(fields) == {field.name for field in self.fields} and all(
            matches_type(fields[field.name], field.type) for field in self.fields
        )
