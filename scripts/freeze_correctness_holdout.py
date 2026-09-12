"""Freeze original synthetic verification cases without running or consulting any model.

The generated expectations are private evaluation inputs. Keep candidate configuration
frozen before inspecting them; after one evaluation the suite is consumed evidence.
"""

import argparse
import hashlib
import json
import random
import secrets
import stat
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from finserve.benchmark.constraint_mapping import (
    RequestConstraintBinding,
    RequestConstraintMap,
    prompt_digest,
)
from finserve.contracts.output_constraint import OutputConstraint
from finserve.evaluation.quality import GoldenCase, GoldenSuite


def exact_decimal(value: Fraction, places: int) -> str:
    """Cross-check finite decimal rendering with rational arithmetic, never binary floats."""
    result = Decimal(value.numerator) / Decimal(value.denominator)
    rendered = format(result, f".{places}f")
    if Fraction(Decimal(rendered)) != value:
        raise ValueError("case needs rounding or exceeds requested precision")
    return rendered


def numeric_rows(rng: random.Random, index: int) -> list[tuple[str, str, str, dict[str, object]]]:
    """Vary arithmetic operands and wording while deriving truth independently of syntax."""
    a, b = rng.randint(101, 987), rng.randint(13, 89)
    integer_prompts = (
        f"A ledger starts at {a} and receives {b}. What is its new total?",
        f"Compute the sum: {a} plus {b}.",
        f"Combine {a} units with another {b} units. Give the total.",
    )
    integer = integer_prompts[index % 3] + " Return only an integer."
    numerator, denominator = rng.randint(-799, 799), 1000
    scale = rng.randint(19, 97)
    income, revenue = numerator * scale, denominator * scale
    margin = (
        f"Revenue is {revenue}; operating income is {income}. "
        if index % 2
        else f"For this firm, operating income = {income} and revenue = {revenue}. "
    ) + "Compute operating income divided by revenue. Return only a decimal with 3 decimal places."
    cents_a, cents_b = rng.randint(-95000, 95000), rng.randint(101, 9000)
    amount_a = exact_decimal(Fraction(cents_a, 100), 2)
    amount_b = exact_decimal(Fraction(cents_b, 100), 2)
    decimal_prompt = (
        f"Subtract {amount_b} from {amount_a}. "
        if index % 2
        else f"Find the difference ({amount_a}) minus ({amount_b}). "
    ) + "Return only the result with exactly 2 decimal places."
    coefficient = rng.randint(100, 999) * (-1 if index % 2 else 1)
    exponent = rng.randint(-6, 6)
    quantity = Fraction(coefficient, 100) * Fraction(10) ** exponent
    original = exact_decimal(quantity, max(2 - exponent, 0))
    scientific_prompt = (
        f"Express {original} in scientific notation. "
        if index % 2
        else f"Rewrite the decimal value {original} using scientific notation. "
    ) + (
        "Return only a single-digit mantissa with exactly 2 decimal places, lowercase e, "
        "an explicit exponent sign, and exactly 2 exponent digits."
    )
    scientific = f"{exact_decimal(Fraction(coefficient, 100), 2)}e{exponent:+03d}"
    if Fraction(Decimal(scientific)) != quantity:
        raise ValueError("scientific expectation differs from original numeric value")
    return [
        ("integer_arithmetic", integer, str(a + b), {"kind": "integer"}),
        (
            "operating_margin",
            margin,
            exact_decimal(Fraction(income, revenue), 3),
            {"kind": "decimal", "decimal_places": 3},
        ),
        (
            "decimal_arithmetic",
            decimal_prompt,
            exact_decimal(Fraction(cents_a - cents_b, 100), 2),
            {"kind": "decimal", "decimal_places": 2},
        ),
        (
            "scientific",
            scientific_prompt,
            scientific,
            {"kind": "scientific", "decimal_places": 2, "exponent_digits": 2},
        ),
    ]


def structured_rows(
    rng: random.Random, index: int
) -> list[tuple[str, str, str, dict[str, object]]]:
    """Keep JSON schemas flat and broad; lengths and task values stay solely in the prompt."""
    amount = rng.randint(-9900, 9900)
    balance = exact_decimal(Fraction(amount, 100), 2)
    scalar_prompt = (
        f"A cash account has balance {balance}; its audit date is unavailable. "
        "Return only a JSON object with balance as a number, positive as a boolean "
        "indicating whether balance is greater than zero, and audit_date as null."
    )
    scalar_expected = (
        f'{{"balance":{balance},"positive":{str(amount > 0).lower()},"audit_date":null}}'
    )
    item_type = ("integer", "number", "string", "boolean", "null", "number")[index]
    size = rng.randint(2, 5)
    values: list[object]
    array_text = ""
    if item_type == "integer":
        values = [rng.randint(-400, 400) for _ in range(size)]
    elif item_type == "number":
        # Exact decimal tokens avoid a float conversion even during JSON serialization.
        tokens = [exact_decimal(Fraction(rng.randint(-999, 999), 100), 2) for _ in range(size)]
        array_text = "[" + ",".join(tokens) + "]"
        values = []
    elif item_type == "string":
        values = ["ledger" + str(rng.randint(1000, 9999)) for _ in range(size)]
    elif item_type == "boolean":
        values = [bool(rng.getrandbits(1)) for _ in range(size)]
    else:
        values = [None] * size
    if item_type != "number":
        array_text = json.dumps(values, separators=(",", ":"))
    array_prompt = (
        f"Preserve the order of this JSON list: {array_text}. "
        if index % 2
        else f"Put the list {array_text} into the requested object without reordering. "
    ) + f"Return only a JSON object with one field values, an array of {item_type} items."
    return [
        (
            "typed_json",
            scalar_prompt,
            scalar_expected,
            {
                "kind": "json_object",
                "fields": [
                    {"name": "balance", "type": "number"},
                    {"name": "positive", "type": "boolean"},
                    {"name": "audit_date", "type": "null"},
                ],
            },
        ),
        (
            "primitive_array",
            array_prompt,
            '{"values":' + array_text + "}",
            {
                "kind": "json_object",
                "fields": [{"name": "values", "type": "array", "item_type": item_type}],
            },
        ),
    ]


def lexical_rows(rng: random.Random, index: int) -> list[tuple[str, str, str, dict[str, object]]]:
    """Test numeric decisions and extraction with both answer alternatives available to decoding."""
    left, right = rng.sample(range(17, 987), 2)
    question = (
        f"Is {left} strictly greater than {right}?"
        if index % 2
        else f"Does the comparison {left} > {right} hold?"
    ) + " Return only lowercase yes or no."
    word = rng.choice(("copper", "willow", "quartz", "harbor", "silver", "maple"))
    ticket = rng.randint(10000, 99999)
    extraction = (
        f"Ticket {ticket} has category {word.upper()}. "
        if index % 2
        else f"Record {ticket}: CATEGORY={word.upper()}. "
    ) + "Return only the category as one lowercase word."
    return [
        ("yes_no", question, "yes" if left > right else "no", {"kind": "yes_no"}),
        ("lowercase_word", extraction, word, {"kind": "lowercase_word"}),
    ]


def build_suite(seed: int) -> tuple[GoldenSuite, RequestConstraintMap]:
    """Bind requested formats before expectation validation and require 48 unique prompt hashes."""
    rng = random.Random(seed)
    cases: list[GoldenCase] = []
    entries: list[RequestConstraintBinding] = []
    for index in range(6):
        rows = numeric_rows(rng, index) + structured_rows(rng, index) + lexical_rows(rng, index)
        for family, prompt, expected, requested_format in rows:
            constraint = OutputConstraint.model_validate(requested_format)
            entries.append(
                RequestConstraintBinding(prompt_sha256=prompt_digest(prompt), constraint=constraint)
            )
            if not constraint.accepts(expected):
                raise ValueError("independently computed expectation violates requested shape")
            cases.append(
                GoldenCase(
                    case_id=f"fresh-{family}-{index + 1:02d}",
                    family=family,
                    prompt=prompt,
                    expected=expected,
                    kind="json" if constraint.kind == "json_object" else "exact",
                )
            )
    suite = GoldenSuite(suite_id="synthetic-correctness-holdout-01", cases=tuple(cases))
    mapping = RequestConstraintMap(entries=tuple(entries))
    if len(cases) != 48:
        raise ValueError("unexpected verification denominator")
    return suite, mapping


def freeze(destination: Path, seed: int) -> dict[str, object]:
    """Create a new directory once, checksum every payload, and mark frozen files read-only."""
    destination = destination.resolve()
    repository = Path(__file__).resolve().parents[1]
    if destination == repository or repository in destination.parents:
        raise ValueError("private holdout evidence must stay outside the repository")
    suite, mapping = build_suite(seed)
    generator = Path(__file__).read_bytes()
    payloads = {
        "suite.json": suite.model_dump_json(indent=2).encode(),
        "constraint-map.json": mapping.model_dump_json(indent=2).encode(),
        "seed.txt": (str(seed) + "\n").encode(),
        "generator.py": generator,
    }
    provenance: dict[str, object] = {
        "schema_version": "synthetic-holdout-freeze-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "suite_hash": suite.digest(),
        "constraint_map_hash": mapping.digest(),
        "case_count": len(suite.cases),
        "family_counts": dict(Counter(case.family for case in suite.cases)),
        "scope": "48 original small synthetic deterministic cases; not a general benchmark",
        "source": "original generator; random fresh seed; no prior answers or model calls",
        "expectations": "integer and exact Fraction/Decimal arithmetic; no binary floats",
        "mapping": "prompt SHA256 and requested format only; no answer or length constraints",
        "freshness": "unqueried at freeze; consumed after evaluation; no global uniqueness claim",
        "candidate_blinding": "freeze candidate configuration before reading expectations",
        "file_protection": "write-once creation, read-only files, SHA256; not OS immutability",
        "evaluator_version": suite.evaluator_version,
    }
    payloads["provenance.json"] = json.dumps(provenance, indent=2, sort_keys=True).encode()
    checksums = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
    payloads["checksums.json"] = json.dumps(checksums, indent=2, sort_keys=True).encode()
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        path = destination / name
        with path.open("xb") as handle:
            handle.write(raw)
        if path.read_bytes() != raw:
            raise OSError("frozen payload verification failed")
        path.chmod(stat.S_IREAD)
    return {"path": str(destination), **provenance, "file_sha256": checksums}


def main() -> None:
    """Print only non-answer metadata so a coordinating agent can remain blinded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    seed = secrets.randbits(128) if args.seed is None else args.seed
    print(json.dumps(freeze(args.output, seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
