"""Contract failures must happen before engine work or admission."""

import pytest
from pydantic import ValidationError

from finserve.contracts.inference import InferenceRequest


@pytest.mark.parametrize(
    "values",
    [
        {"prompt": ""},
        {"prompt": "x", "max_tokens": 0},
        {"prompt": "x", "max_tokens": True},
        {"prompt": "x", "temperature": float("nan")},
        {"prompt": "x", "timeout_seconds": float("inf")},
        {"prompt": "x", "request_id": "bad\r\nheader"},
        {"prompt": "x", "request_id": "\u03bb"},
        {"prompt": "x", "unknown": 1},
    ],
)
def test_invalid_requests(values: dict[str, object]) -> None:
    """Invalid work budgets and ignored options must never reach the scheduler."""
    with pytest.raises(ValidationError):
        InferenceRequest.model_validate(values)
