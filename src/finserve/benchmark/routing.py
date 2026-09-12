"""Retain observed gateway attribution separately from immutable runtime attestation."""

from typing import Self

import httpx
from pydantic import Field, model_validator

from finserve.contracts.deployment import ImmutableModel


class RoutingRevision(ImmutableModel):
    """A revision digest and generation describe one observed logical or physical binding."""

    revision_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    revision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=0, le=2**63 - 1, strict=True)


class RequestRouting(ImmutableModel):
    """Pre-dispatch failures may name an anchor without claiming an executing model process."""

    physical: RoutingRevision | None = None
    anchor: RoutingRevision | None = None
    pool_generation: int | None = Field(default=None, ge=0, le=2**63 - 1, strict=True)

    @model_validator(mode="after")
    def complete_identity(self) -> Self:
        """Pool attribution requires both bindings; legacy physical-only headers remain valid."""
        if self.physical is None and self.anchor is None:
            raise ValueError("routing attribution requires an observed binding")
        if (self.pool_generation is not None) != (
            self.physical is not None and self.anchor is not None
        ):
            raise ValueError("pool generation requires complete physical and anchor bindings")
        return self


def _header(headers: httpx.Headers, name: str) -> str | None:
    """Reject duplicate or oversized identity values instead of silently selecting one."""
    values = headers.get_list(name)
    if not values:
        return None
    if len(values) != 1 or len(values[0]) > 128:
        raise ValueError("ambiguous routing header")
    return values[0]


def _generation(value: str) -> int:
    """Use canonical bounded decimal counters, without signs, padding or coercible booleans."""
    if not value.isascii() or not value.isdecimal() or len(value) > 19:
        raise ValueError("invalid routing generation")
    result = int(value)
    if str(result) != value or result > 2**63 - 1:
        raise ValueError("noncanonical routing generation")
    return result


def _binding(headers: httpx.Headers, names: tuple[str, str, str]) -> RoutingRevision | None:
    """A partial group cannot be interpreted as a complete process or anchor identity."""
    values = tuple(_header(headers, name) for name in names)
    if all(value is None for value in values):
        return None
    revision, digest, generation = values
    if revision is None or digest is None or generation is None:
        raise ValueError("incomplete routing binding")
    return RoutingRevision(
        revision_id=revision, revision_digest=digest, generation=_generation(generation)
    )


def observed_routing(response: httpx.Response) -> RequestRouting | None:
    """Capture claims from the actual response; receipt verification remains a separate step."""
    physical = _binding(
        response.headers,
        (
            "x-finserve-revision",
            "x-finserve-revision-digest",
            "x-finserve-route-generation",
        ),
    )
    anchor = _binding(
        response.headers,
        (
            "x-finserve-anchor-revision",
            "x-finserve-anchor-digest",
            "x-finserve-anchor-generation",
        ),
    )
    pool = _header(response.headers, "x-finserve-pool-generation")
    if physical is None and anchor is None and pool is None:
        return None
    if response.is_success and physical is None:
        raise ValueError("successful routed response lacks physical attribution")
    return RequestRouting(
        physical=physical,
        anchor=anchor,
        pool_generation=_generation(pool) if pool is not None else None,
    )
