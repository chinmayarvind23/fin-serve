"""Bound request work before it can consume engine memory or scheduling slots."""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def request_id() -> str:
    """Use opaque IDs so telemetry does not need prompt text or user identities."""
    return str(uuid4())


class InferenceRequest(BaseModel):
    """Only supported generation options are accepted; unknown options fail explicitly."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    request_id: str = Field(
        default_factory=request_id, max_length=128, min_length=1, pattern=r"^[A-Za-z0-9_-]+$"
    )
    model: str = Field(default="reference", min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=32768)
    max_tokens: int = Field(default=32, ge=1, le=2048, strict=True)
    temperature: float = Field(default=0, ge=0, le=2)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    stream: bool = True


class ChatMessage(BaseModel):
    """Text-only chat is explicit; media requires a separate bounded modality contract."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32768)


class ChatRequest(BaseModel):
    """A small compatible chat surface avoids silently ignoring unsupported parameters."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = Field(default="reference", min_length=1, max_length=256)
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=32, ge=1, le=2048, strict=True)
    temperature: float = Field(default=0, ge=0, le=2)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    stream: bool = True

    def to_inference(self) -> InferenceRequest:
        """Reference chat uses labeled text; production adapters use model chat templates."""
        return InferenceRequest(
            model=self.model,
            prompt="\n".join(f"{message.role}: {message.content}" for message in self.messages),
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
