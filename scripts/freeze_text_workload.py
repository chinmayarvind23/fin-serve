"""Write original synthetic release inputs before running any serving optimization."""

import argparse
from pathlib import Path

from finserve.benchmark.workload import WorkItem, Workload


def chat_prompt(content: str) -> str:
    """Freeze Qwen's basic text chat delimiters explicitly in raw-completion prompt bytes."""
    return (
        "<|im_start|>system\nYou are a careful assistant. Use only the provided facts."
        "<|im_end|>\n<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
    )


def frozen_workload() -> Workload:
    """Vary synthetic entities, lengths and budgets; use the same sequence for every engine."""
    items: list[WorkItem] = []
    for index in range(16):
        revenue, capex = 500 + 25 * index, 20 + index
        context = " ".join(
            f"Division {part} reported stable customer retention and no acquisitions."
            for part in range(1 + (index % 4) * 8)
        )
        cases = (
            (
                "SEC_QA",
                f"Example {index} reported revenue {revenue} and operating income "
                f"{revenue // 5}. {context} Compute the operating margin. Explain in one sentence.",
                32,
            ),
            (
                "FINANCIAL_TABLE_EXTRACTION",
                f"Extract these facts as JSON only: company Example "
                f"{index}, revenue {revenue}, capex {capex}. Use keys company, revenue, capex.",
                48,
            ),
            (
                "EARNINGS_SUMMARY",
                f"Summarize the following in four sentences: Example {index} "
                f"reported revenue of {revenue}. Costs increased by {index + 1} percent. Guidance "
                f"remained unchanged. {context} Do not add unsupported claims.",
                96,
            ),
            (
                "GENERAL",
                f"A library has {100 + index} books and adds {10 + index}. "
                "Explain how to calculate the new total, then give the total in one sentence.",
                32,
            ),
        )
        for family, content, budget in cases:
            items.append(
                WorkItem(
                    case_id=f"{family.lower()}_{index:02d}",
                    family=family,
                    prompt=chat_prompt(content),
                    max_tokens=budget,
                )
            )
    return Workload(suite_id="original-synthetic-text-release-v1", version=1, items=tuple(items))


def main() -> None:
    """Exclusive creation prevents accidental edits after an unfavorable experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workload = frozen_workload()
    with args.output.open("x", encoding="utf-8") as output:
        output.write(workload.model_dump_json(indent=2) + "\n")
    print(workload.digest())


if __name__ == "__main__":
    main()
