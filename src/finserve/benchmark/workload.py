"""Frozen, synthetic workloads with an identity independent of serving configuration."""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class WorkItem(BaseModel):
    """An immutable logical input; family is reporting metadata, never routing policy."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    case_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=32, ge=1, le=4096)
    family: str = "GENERAL"


class Workload(BaseModel):
    """A versioned sequence replayed in exactly the same order for each configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    suite_id: str
    version: int = Field(ge=1)
    items: tuple[WorkItem, ...] = Field(min_length=1)

    def digest(self) -> str:
        """Hash canonical inputs so baseline/candidate changes cannot be hidden."""
        encoded = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def default_workload(max_tokens: int = 32) -> Workload:
    """Supply original synthetic text cases; no pending image fixture is represented as tested."""
    cases = (
        ("general_001", "GENERAL", "Complete the sentence: The quick brown fox"),
        (
            "sec_001",
            "SEC_QA",
            "Revenue is 500 and operating income is 75. What is operating margin?",
        ),
        ("table_001", "FINANCIAL_TABLE_EXTRACTION", "Return JSON: Revenue 820, Capex 41."),
        (
            "summary_001",
            "EARNINGS_SUMMARY",
            "Summarize: Revenue grew. Margin fell. Guidance unchanged.",
        ),
    )
    return Workload(
        suite_id="synthetic-text-serving-v1",
        version=1,
        items=tuple(
            WorkItem(case_id=key, family=family, prompt=prompt, max_tokens=max_tokens)
            for key, family, prompt in cases
        ),
    )
