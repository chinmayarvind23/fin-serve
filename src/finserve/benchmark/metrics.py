"""Explicit client-side timing and complete logical-request accounting."""

import math
from collections.abc import Sequence
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from finserve.benchmark.routing import RequestRouting


class RequestRecord(BaseModel):
    """One offered logical request, including overload, protocol errors and timeouts."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    logical_id: int = Field(ge=0)
    case_id: str
    family: str
    phase: str = "measured"
    offered: bool = True
    intended_arrival_s: float | None = Field(default=None, ge=0)
    scheduled_s: float = Field(ge=0)
    send_s: float | None = Field(default=None, ge=0)
    first_content_s: float | None = Field(default=None, ge=0)
    complete_s: float = Field(ge=0)
    success: bool
    status_code: int | None = None
    error: str | None = None
    generated_tokens: int | None = Field(default=None, ge=0)
    output: str = ""
    server_ttft_s: float | None = Field(default=None, ge=0)
    routing: RequestRouting | None = None

    @model_serializer(mode="wrap")
    def preserve_historical_rows(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Missing attribution stays absent when old retained request records are reconstructed."""
        result: dict[str, Any] = handler(self)
        if self.routing is None:
            result.pop("routing", None)
        return result

    @model_validator(mode="after")
    def valid_timing(self) -> "RequestRecord":
        """Reject impossible clocks before they can produce a plausible-looking aggregate."""
        if self.complete_s < self.scheduled_s:
            raise ValueError("completion precedes scheduled arrival")
        if self.send_s is not None and not self.scheduled_s <= self.send_s <= self.complete_s:
            raise ValueError("send outside request lifetime")
        if self.first_content_s is not None:
            if self.send_s is None or not self.send_s <= self.first_content_s <= self.complete_s:
                raise ValueError("first content outside network lifetime")
        if self.success and (self.send_s is None or self.error is not None):
            raise ValueError("success must be sent and error-free")
        if self.success and self.status_code is not None and not 200 <= self.status_code < 300:
            raise ValueError("successful HTTP status must be 2xx")
        if self.success and self.generated_tokens and self.first_content_s is None:
            raise ValueError("positive generated tokens require observed content")
        return self


def percentile(values: Sequence[float], q: float) -> float | None:
    """Use linear interpolation at (n-1)*q, including singleton and empty populations."""
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("quantile must be finite and within [0, 1]")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("durations must be finite and nonnegative")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(records: Sequence[RequestRecord], measured_seconds: float) -> dict[str, object]:
    """Count all offered requests; latency covers successful responses and errors remain raw."""
    if not math.isfinite(measured_seconds) or measured_seconds <= 0:
        raise ValueError("measurement window must be finite and positive")
    measured = [row for row in records if row.phase == "measured" and row.offered]
    if len({row.logical_id for row in measured}) != len(measured):
        raise ValueError("duplicate logical request IDs")
    succeeded = [row for row in measured if row.success]
    sent = sum(row.send_s is not None for row in measured)
    known_tokens = all(row.generated_tokens is not None for row in succeeded)
    tokens = sum(row.generated_tokens or 0 for row in succeeded) if known_tokens else None
    e2e = [row.complete_s - row.send_s for row in succeeded if row.send_s is not None]
    offered_e2e = [row.complete_s - row.scheduled_s for row in succeeded]
    send_lag = [row.send_s - row.scheduled_s for row in measured if row.send_s is not None]
    ttft = [
        row.first_content_s - row.send_s
        for row in succeeded
        if row.first_content_s is not None and row.send_s is not None
    ]
    server_ttft = [row.server_ttft_s for row in succeeded if row.server_ttft_s is not None]
    return {
        "offered_requests": len(measured),
        "sent_requests": sent,
        "successful_requests": len(succeeded),
        "failed_requests": len(measured) - len(succeeded),
        "success_rate": len(succeeded) / len(measured) if measured else None,
        "success_denominator": "all offered measured logical requests, including local overload",
        "measured_seconds": measured_seconds,
        "requests_per_second": len(succeeded) / measured_seconds,
        "generated_tokens": tokens,
        "tokens_per_second": tokens / measured_seconds if tokens is not None else None,
        "client_ttft_median_s": percentile(ttft, 0.5),
        "client_ttft_samples": len(ttft),
        "server_ttft_median_s": percentile(server_ttft, 0.5),
        "server_ttft_samples": len(server_ttft),
        "e2e_p50_s": percentile(e2e, 0.5),
        "e2e_p95_s": percentile(e2e, 0.95),
        "scheduled_to_complete_p95_s": percentile(offered_e2e, 0.95),
        "scheduled_to_send_p95_s": percentile(send_lag, 0.95),
        "latency_population": "successful requests; missing content excluded from TTFT only",
        "percentile_method": "linear interpolation, index=(n-1)*q",
    }
