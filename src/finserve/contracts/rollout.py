"""Frozen local traffic endpoint and bounded probation settings for producer orchestration."""

from typing import Self

import httpx
from pydantic import Field, model_validator

from finserve.contracts.deployment import ImmutableModel


class RolloutSettings(ImmutableModel):
    """Credentials stay in worker memory; this immutable configuration contains no secrets."""

    traffic_url: str = Field(max_length=2048)
    maximum_probes: int = Field(default=60, ge=3, le=256, strict=True)
    interval_seconds: float = Field(default=5, ge=0.1, le=60)
    slow_probe_seconds: float = Field(default=2, gt=0, le=30)
    probe_timeout_seconds: float = Field(default=5, gt=0, le=30)

    @model_validator(mode="after")
    def bounded_destination(self) -> Self:
        """Reject credentials, query strings and unreachable timing thresholds before collection."""
        address = httpx.URL(self.traffic_url)
        if (
            address.scheme not in {"http", "https"}
            or not address.host
            or address.username
            or address.password
            or address.query
            or address.fragment
            or self.slow_probe_seconds > self.probe_timeout_seconds
        ):
            raise ValueError("invalid rollout endpoint or probe thresholds")
        return self
