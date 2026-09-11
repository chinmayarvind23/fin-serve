"""Freeze original numeric, extraction, factual and nonfinancial correctness cases."""

import argparse
import json
from pathlib import Path

from finserve.evaluation.quality import GoldenCase, GoldenSuite


def prompt(content: str) -> str:
    """Pin raw chat bytes because this suite is sent through the completions boundary."""
    return (
        "<|im_start|>system\nFollow the requested output format exactly. "
        "Use only the facts provided.<|im_end|>\n<|im_start|>user\n"
        + content
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def frozen_suite() -> GoldenSuite:
    """Separate objective correctness from byte agreement and freeze all cases before scoring."""
    cases: list[GoldenCase] = []
    for index in range(8):
        revenue, capex = 400 + 100 * index, 20 + index
        rows = (
            (
                "SEC_QA",
                f"Revenue is {revenue}, operating income is {revenue // 5}. "
                "What is operating margin as a decimal? Return only the decimal number.",
                "0.2",
                "exact",
            ),
            (
                "FINANCIAL_TABLE_EXTRACTION",
                f"Revenue is {revenue} and capital expenditure is {capex}. "
                "Return only a JSON object with numeric keys revenue and capex. No extra keys.",
                json.dumps({"revenue": revenue, "capex": capex}),
                "json",
            ),
            (
                "EARNINGS_SUMMARY",
                f"Example {index} reported revenue {revenue}. Costs rose. Guidance was unchanged. "
                "Did guidance increase? Return only yes or no.",
                "no",
                "exact",
            ),
            (
                "GENERAL",
                f"A library has {100 + index} books and adds {10 + index}. "
                "How many books are there now? Return only the integer.",
                str(110 + 2 * index),
                "exact",
            ),
        )
        for family, content, expected, kind in rows:
            cases.append(
                GoldenCase(
                    case_id=f"{family.lower()}_{index:02d}",
                    family=family,
                    prompt=prompt(content),
                    expected=expected,
                    kind="json" if kind == "json" else "exact",
                )
            )
    return GoldenSuite(suite_id="original-synthetic-correctness-32-v1", cases=tuple(cases))


def main() -> None:
    """Exclusive output creation keeps an unfavorable result from silently changing the suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = frozen_suite()
    with args.output.open("x", encoding="utf-8") as output:
        output.write(suite.model_dump_json(indent=2) + "\n")
    print(suite.digest())


if __name__ == "__main__":
    main()
