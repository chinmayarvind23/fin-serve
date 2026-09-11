"""Original held-out routing inputs, frozen before any engine or policy measurement."""

import argparse
import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 20260911


class RoutingCase(BaseModel):
    """Length and SLO labels are report slices; the scheduler sees only ordinary inference input."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    prompt: str
    input_class: Literal["short", "long"]
    max_tokens: Literal[16, 128]
    ttft_slo_seconds: float = 1.0
    e2e_slo_seconds: float = 15.0


class RoutingWorkload(BaseModel):
    """All four cohorts share ordered bytes and budgets; failure never changes this population."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    suite_id: Literal["two-engine-heldout-routing-v1"] = "two-engine-heldout-routing-v1"
    seed: Literal[20260911] = SEED
    model: Literal["Qwen/Qwen2.5-0.5B-Instruct"] = MODEL
    revision: Literal["7ae557604adf67be50417f59c2c2f167def9a775"] = REVISION
    concurrency: Literal[8] = 8
    warmup_per_cohort: Literal[8] = 8
    policies: tuple[str, ...] = ("least_load", "adaptive", "adaptive", "least_load")
    cases: tuple[RoutingCase, ...] = Field(min_length=64, max_length=64)

    def canonical_bytes(self) -> bytes:
        """Stable serialization includes reporting thresholds as well as measured request input."""
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")).encode()

    def digest(self) -> str:
        """Hash prompts, order, budgets and reporting thresholds together."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def frozen_workload() -> RoutingWorkload:
    """Construct balanced original mechanical tasks without borrowing finance release/eval cases."""
    rng = random.Random(SEED)
    vocabulary = ("maple", "river", "copper", "garden", "window", "paper", "cloud", "stone")
    cases: list[RoutingCase] = []
    for index in range(64):
        kind: Literal["short", "long"] = "long" if index % 4 >= 2 else "short"
        budget: Literal[16, 128] = 16 if index % 2 == 0 else 128
        marker = f"record-{index:02d}-{rng.randrange(100000, 999999)}"
        words = [rng.choice(vocabulary) for _ in range(200 if kind == "long" else 12)]
        prompt = (
            f"Record identifier: {marker}.\nContext words: {' '.join(words)}.\n"
            "Task: Write a numbered list of simple objects found in a classroom. "
            "Use one object per line. Continue the list until item 40.\n1."
        )
        cases.append(
            RoutingCase(case_id=marker, prompt=prompt, input_class=kind, max_tokens=budget)
        )
    rng.shuffle(cases)
    return RoutingWorkload(cases=tuple(cases))


def preflight(workload: RoutingWorkload, token_count: Callable[[str], int]) -> dict[str, int]:
    """Reject context overflow before model calls instead of truncating frozen inputs."""
    counts = {case.case_id: token_count(case.prompt) for case in workload.cases}
    if len(counts) != 64 or any(
        type(counts[case.case_id]) is not int
        or counts[case.case_id] < 1
        or counts[case.case_id] + case.max_tokens > 1024
        for case in workload.cases
    ):
        raise ValueError("invalid, duplicate or context-overflow routing workload")
    return counts


def main() -> None:
    """Materialize one immutable workload outside source control before any engine is started."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        parser.error("workload evidence must be outside the source repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    workload = frozen_workload()
    with output.open("xb") as target:
        target.write(workload.canonical_bytes())
    print(json.dumps({"workload_sha256": workload.digest(), "cases": len(workload.cases)}))


if __name__ == "__main__":
    main()
