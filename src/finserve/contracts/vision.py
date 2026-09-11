"""A single bounded inline image is a distinct capability from text or visual generation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finserve.contracts.inference import request_id

MAX_PNG_BYTES = 1_048_576
MAX_VISION_BODY_BYTES = 1_450_000
VISION_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
VISION_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"


class VisionRequest(BaseModel):
    """Exclude URLs, batches and arbitrary upstream options before engine admission."""

    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, validate_default=True
    )
    request_id: str = Field(default_factory=request_id, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    model: str = Field(default=VISION_MODEL, min_length=1, max_length=256)
    modality: Literal["image-text"] = "image-text"
    prompt: str = Field(min_length=1, max_length=4096)
    image_png_base64: str = Field(min_length=1, max_length=4 * ((MAX_PNG_BYTES + 2) // 3))
    max_tokens: int = Field(default=128, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    stream: bool = True
