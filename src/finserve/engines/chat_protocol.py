"""Strict shared native-chat delta parsing for text and image engine adapters."""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finserve.engines.openai_adapter import (
    CompletionFrame,
    CompletionState,
    CompletionUsage,
    EngineProtocolError,
)


class ChatDelta(BaseModel):
    """Accept normal assistant role/content deltas; tool and reasoning protocols are separate."""

    model_config = ConfigDict(extra="ignore", strict=True)
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: None = None
    function_call: None = None
    refusal: None = None


class ChatChoice(BaseModel):
    """Only one generation is budgeted, and every finish reason must have defined semantics."""

    model_config = ConfigDict(extra="ignore", strict=True)
    index: Literal[0]
    delta: ChatDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatFrame(BaseModel):
    """The final usage-only frame is distinct from assistant text deltas."""

    model_config = ConfigDict(extra="ignore", strict=True)
    choices: list[ChatChoice] = Field(max_length=1)
    usage: CompletionUsage | None = None


@dataclass
class ChatState(CompletionState):
    """Reuse accounting invariants while parsing the actual chat delta wire shape."""

    def consume(self, data: str) -> str:
        """Translate validated chat fields to the shared count state without guessing tokens."""
        try:
            frame = ChatFrame.model_validate(json.loads(data, object_pairs_hook=unique_object))
        except (ValidationError, ValueError):
            raise EngineProtocolError("Engine returned an invalid chat event") from None
        choices = [
            {"index": 0, "text": choice.delta.content or "", "finish_reason": choice.finish_reason}
            for choice in frame.choices
        ]
        # Shared state enforces finish-before-usage, total bounds and no data after usage.
        translated = CompletionFrame.model_validate({"choices": choices, "usage": frame.usage})
        return super().consume(translated.model_dump_json())


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Ambiguous duplicate JSON fields cannot decide final usage or change media semantics."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
