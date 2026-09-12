"""Frozen local replication authorization never substitutes for canonical release approval."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.managed_runtime import RuntimeLaunchSpec
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.reliability.capacity_policy import CapacityPolicy

CAPACITY_PROTOCOL = "local-capacity-v1"


class CapacityPlan(ImmutableModel):
    """Every possible launch is frozen before load observations; only one extra may exist."""

    protocol: Literal["local-capacity-v1"] = CAPACITY_PROTOCOL
    plan_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    deployment_id: str = Field(min_length=1, max_length=128)
    expected_generation: int = Field(ge=0, strict=True)
    approval_job_id: str = Field(min_length=1, max_length=128)
    primary_launch_stage: str
    model_stage_id: str
    build_stage_id: str
    primary: RuntimeLaunchSpec
    replicas: tuple[RuntimeLaunchSpec, ...] = Field(min_length=1, max_length=8)
    workspace: Path
    policy: CapacityPolicy = Field(default_factory=CapacityPolicy)
    per_member_requests: int = Field(default=4, ge=1, le=128, strict=True)
    max_samples: int = Field(default=120, ge=1, le=4096, strict=True)

    @model_validator(mode="after")
    def equivalent_replicas(self) -> Self:
        """Authorize endpoint-only copies while preserving each full profile/revision digest."""
        if not self.workspace.is_absolute():
            raise ValueError("capacity workspace must be absolute")
        if (
            self.per_member_requests
            > VLLMParameters.model_validate_json(
                self.primary.profile.engine_parameters_json
            ).max_num_seqs
        ):
            raise ValueError("serving admission exceeds frozen engine sequence capacity")
        ids = {self.primary.revision.revision_id}
        endpoint = self.replicas[0].profile.base_url
        for replica in self.replicas:
            if (
                replica.revision.revision_id in ids
                or replica.profile.base_url == self.primary.profile.base_url
                or replica.profile.base_url != endpoint
            ):
                raise ValueError("replica identities and fixed extra endpoint must be distinct")
            ids.add(replica.revision.revision_id)
            # These are the only authorized physical-identity differences. Model/image,
            # canonical engine knobs and all managed resource limits remain exact.
            expected = self.primary.model_dump(mode="json")
            actual = replica.model_dump(mode="json")
            for value in (expected, actual):
                value["profile"].pop("base_url")
                value["revision"].pop("revision_id")
                value["revision"].pop("config_digest")
            if expected != actual:
                raise ValueError("capacity replica changes approved serving semantics")
        return self


class CapacityState(ImmutableModel):
    """A persisted phase retains unresolved ownership rather than recycling a slot by age."""

    plan_id: str
    phase: Literal["one", "warming", "two", "draining", "blocked", "closed"] = "one"
    cycle: int = Field(default=0, ge=0, le=8)
    samples: int = Field(default=0, ge=0)
    memory: str = "{}"
    error: str | None = None
