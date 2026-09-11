"""Bound request work before it can consume engine memory or scheduling slots."""

import json
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def request_id() -> str:
    """Use opaque IDs so telemetry does not need prompt text or user identities."""
    return str(uuid4())


class ChatMessage(BaseModel):
    """Text-only chat is explicit; media requires a separate bounded modality contract."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32768)


class InferenceRequest(BaseModel):
    """Only supported generation options are accepted; unknown options fail explicitly."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    request_id: str = Field(
        default_factory=request_id, max_length=128, min_length=1, pattern=r"^[A-Za-z0-9_-]+$"
    )
    model: str = Field(default="reference", min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=32768)
    messages: list[ChatMessage] | None = Field(default=None, min_length=1, max_length=64)
    max_tokens: int = Field(default=32, ge=1, le=2048, strict=True)
    temperature: float = Field(default=0, ge=0, le=2)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    stream: bool = True

    @model_validator(mode="after")
    def coherent_chat(self) -> Self:
        """Keep reference scheduling length identical to the full role-preserving chat payload."""
        if self.messages is not None and self.prompt != reference_prompt(self.messages):
            raise ValueError("chat messages must match the bounded reference prompt")
        if self.messages is not None:
            text_fields = {
                "prompt": self.prompt,
                "messages": [item.model_dump() for item in self.messages],
            }
            encoded = json.dumps(text_fields, ensure_ascii=False, separators=(",", ":")).encode()
            # Ray carries both forms; reserve 8KiB of its 128KiB cap for bounded request metadata.
            if len(encoded) > 120 * 1024:
                raise ValueError("chat text exceeds the internal routing byte budget")
        return self


def reference_prompt(messages: list[ChatMessage]) -> str:
    """Reference engines consume deterministic labeled text; native engines retain the roles."""
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


class ChatStreamOptions(BaseModel):
    """Chat callers may explicitly request the authoritative usage already emitted by this API."""

    model_config = ConfigDict(extra="forbid")
    include_usage: Literal[True] = True

    @field_validator("include_usage", mode="before")
    @classmethod
    def exact_true(cls, value: object) -> object:
        """Reject numeric/string coercion and unsupported usage suppression."""
        if value is not True:
            raise ValueError("include_usage must be true")
        return value


class ChatRequest(BaseModel):
    """A small compatible chat surface avoids silently ignoring unsupported parameters."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = Field(default="reference", min_length=1, max_length=256)
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=32, ge=1, le=2048, strict=True)
    temperature: float = Field(default=0, ge=0, le=2)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    stream: bool = True
    stream_options: ChatStreamOptions | None = None

    def to_inference(self) -> InferenceRequest:
        """Reference chat uses labeled text; production adapters use model chat templates."""
        return InferenceRequest(
            model=self.model,
            prompt=reference_prompt(self.messages),
            messages=[message.model_copy(deep=True) for message in self.messages],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            stream=self.stream,
        )


class EngineToken(BaseModel):
    """Token counts come from the engine, never from the number of SSE frames."""

    text: str
    token_id: int | None = None
    generated_tokens: int = Field(default=1, ge=0)
    finish_reason: Literal["stop", "length"] | None = None
