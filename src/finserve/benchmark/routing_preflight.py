"""CPU-only pinned-tokenizer preflight for the immutable twin-engine workload."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

from finserve.benchmark.routing_workload import MODEL, REVISION, frozen_workload, preflight


def inspect_tokenizer(snapshot: Path) -> dict[str, object]:
    """Load local tokenizer files only; weight loading and CUDA access are outside this process."""
    if snapshot.name != REVISION or not snapshot.is_dir():
        raise ValueError("exact pinned local model snapshot required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["USE_TORCH"] = "0"
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )

    def count(prompt: str) -> int:
        """Use default special-token semantics and conservatively retain the larger tokenization."""
        return max(
            len(tokenizer.encode(prompt)), len(tokenizer.encode(prompt, add_special_tokens=False))
        )

    workload = frozen_workload()
    counts = preflight(workload, count)
    hashes = {
        name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        if (snapshot / name).is_file()
    }
    return {
        "model": MODEL,
        "revision": REVISION,
        "snapshot": str(snapshot),
        "transformers_version": importlib.metadata.version("transformers"),
        "workload_sha256": workload.digest(),
        "prompt_tokens": counts,
        "max_prompt_plus_output_tokens": max(
            counts[case.case_id] + case.max_tokens for case in workload.cases
        ),
        "tokenizer_file_sha256": hashes,
        "scope": "CPU tokenizer only; no model inference",
    }


def main() -> None:
    """Persist unique preflight evidence outside the source tree before engine calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[3]
    if output == repository or repository in output.parents:
        parser.error("preflight evidence must be outside source repository")
    result = inspect_tokenizer(args.snapshot)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "max_prompt_plus_output_tokens": result["max_prompt_plus_output_tokens"],
                "workload_sha256": result["workload_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
