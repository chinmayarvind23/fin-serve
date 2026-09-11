"""GPU cost calculations require explicit, independently sourced billing inputs."""

import math

from pydantic import BaseModel, ConfigDict, Field


class CostInput(BaseModel):
    """Cost covers billed instance-hours, including declared idle and warmup allocation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    instance_hour_price_usd: float = Field(gt=0)
    billed_instance_hours: float = Field(gt=0)
    generated_tokens: int = Field(gt=0)
    pricing_source: str = Field(min_length=1)
    pricing_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    hardware: str = Field(min_length=1)

    def per_million_tokens(self) -> float:
        """Never substitute request count or client CPU time for generated tokens/GPU hours."""
        result = self.instance_hour_price_usd * self.billed_instance_hours * 1e6
        result /= self.generated_tokens
        if not math.isfinite(result) or result <= 0:
            raise ValueError("cost overflow or underflow")
        return result


def cost_reduction(baseline: CostInput, candidate: CostInput, quality_passed: bool) -> float:
    """Reject an economic improvement claim if the candidate failed its quality gate."""
    if not quality_passed:
        raise ValueError("cost comparison requires equivalent quality")
    reduction = 1 - candidate.per_million_tokens() / baseline.per_million_tokens()
    if not math.isfinite(reduction):
        raise ValueError("cost reduction overflow")
    return reduction
