# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.0"]
# ///
"""Plot audited fixed-profile concurrency observations with their provenance and quality limits."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def public_data(raw: bytes) -> dict[str, Any]:
    """Publish aggregate identities while omitting prompts, URLs and physical device UUIDs."""
    report = json.loads(raw)
    if report["schema"] != "finserve-concurrency-frontier-v1":
        raise ValueError("audited concurrency data required")
    if [run["configuration"]["concurrency"] for run in report["runs"]] != [1, 4, 8, 16]:
        raise ValueError("this layout labels the four recorded development concurrency cells")
    if [run["run_id"] for run in report["runs"]] != [
        "37454aa0-a522-44a8-bc2d-1b64094d8280",
        "8beb35ef-c034-4c03-b81e-bb6c67cf719d",
        "a20137a9-0583-4cc8-a4d3-741a24379f34",
        "5c3fb499-04a6-46cf-9e67-c8cefbcbf348",
    ]:
        raise ValueError("figure labels are specific to the original development sweep")
    if report["quality_context"]["passed"]:
        raise ValueError("the recorded separate quality gate failed")
    return {
        "schema": report["schema"],
        "source_audit_sha256": hashlib.sha256(raw).hexdigest(),
        "audit_source_sha256": report["audit_source_sha256"],
        "frontiers": report["frontiers"],
        "runs": [
            {
                "name": run["name"],
                "run_id": run["run_id"],
                "created_at": run["created_at"],
                "configuration": run["configuration"],
                "workload_hash": run["workload_hash"],
                "summary": run["summary"],
                "warmup_records": run["warmup_records"],
                "dirty": bool(run["dirty_paths"]),
                "source_provenance": run["source_provenance"],
                "reviewed_untracked_runtime_exclusions": run[
                    "reviewed_untracked_runtime_exclusions"
                ],
                "untracked_runtime_bytes_attested": run["untracked_runtime_bytes_attested"],
                "raw_records_sha256": run["artifact_sha256"]["run/requests.jsonl"],
            }
            for run in report["runs"]
        ],
        "quality_context": {
            key: value
            for key, value in report["quality_context"].items()
            if key not in {"dirty_paths", "artifact_sha256"}
        },
        "limitations": report["limitations"],
    }


def render(data: dict[str, Any], output: Path) -> None:
    """Show two latency tradeoffs without smoothing or extrapolating unmeasured configurations."""
    ink, muted, paper = "#152b43", "#506174", "#f7f8fa"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": ink})
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 8.0), facecolor=paper)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.40, wspace=0.28)
    figure.text(0.08, 0.95, "FINSERVE / DEVELOPMENT CONCURRENCY SWEEP", color=muted, size=10)
    figure.text(
        0.08, 0.895, "Throughput and latency under fixed engine limits", size=21, weight="bold"
    )
    figure.text(0.08, 0.852, "Qwen2.5-0.5B / vLLM 0.29.0 / one RTX 4070 Laptop GPU", color=muted)
    figure.text(
        0.08,
        0.817,
        "256 measured + 16 warmups per cell; all measured requests completed",
        color=muted,
    )
    runs = data["runs"]
    for axis, metric, title in zip(
        axes,
        ("client_ttft_median_s", "e2e_p95_s"),
        ("Client median TTFT (ms)", "Successful-request E2E p95 (ms)"),
        strict=True,
    ):
        frontier = [
            run for run in runs if run["configuration"]["concurrency"] in data["frontiers"][metric]
        ]
        axis.plot(
            [run["summary"]["requests_per_second"] for run in frontier],
            [run["summary"][metric] * 1000 for run in frontier],
            color="#87a1b2",
            linestyle=":",
            linewidth=1.3,
            label="Observed frontier; guide only",
        )
        for run in runs:
            x, y = run["summary"]["requests_per_second"], run["summary"][metric] * 1000
            axis.scatter(x, y, s=65, color="#087f8c", zorder=3)
            axis.annotate(
                f"c{run['configuration']['concurrency']}",
                (x, y),
                xytext=(5, 9),
                textcoords="offset points",
                weight="bold",
                color=ink,
            )
        axis.set(
            title=title,
            xlabel="Successful requests / s",
            xlim=(0, 46),
            ylim=(0, max(run["summary"][metric] for run in runs) * 1220),
        )
        axis.set_facecolor(paper)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    table_axis = figure.add_axes((0.08, 0.205, 0.89, 0.14))
    table_axis.axis("off")
    rows = [
        [
            f"c{run['configuration']['concurrency']}",
            f"{run['summary']['measured_seconds']:.2f}",
            f"{run['summary']['requests_per_second']:.2f}",
            f"{run['summary']['tokens_per_second']:.2f}",
            str(run["summary"]["generated_tokens"]),
        ]
        for run in runs
    ]
    table = table_axis.table(
        cellText=rows,
        colLabels=["Concurrency", "Duration (s)", "Requests/s", "Tokens/s", "Output tokens"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.25)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e0e7")
        cell.set_facecolor("#e9eef3" if row == 0 else paper)
    quality = data["quality_context"]
    figure.text(
        0.08,
        0.15,
        f"Separate same-profile correctness: {quality['candidate_accuracy']:.2%}; gate FAILED.",
        weight="bold",
        color="#a33438",
    )
    figure.text(
        0.08,
        0.112,
        "One short sequential observation per cell; "
        "no uncertainty estimate or accepted release optimum.",
        size=9,
        color=muted,
    )
    figure.text(
        0.08,
        0.077,
        "Source labels differ; tracked runtime matches. "
        "Three reviewed untracked files lack byte attestation; digests undeclared.",
        size=8.5,
        color=muted,
    )
    figure.text(
        0.08,
        0.043,
        "Native sequence cap 16 / token budget 2,048 / prefix cache off. "
        "Internal batch shapes unmeasured.",
        size=9,
        color=muted,
    )
    figure.savefig(output / "concurrency-frontier.png", dpi=180, facecolor=paper)
    figure.savefig(output / "concurrency-frontier.svg", facecolor=paper)
    plt.close(figure)


def main() -> None:
    """Create a fresh external figure directory so existing evidence cannot be overwritten."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or repository in output.parents:
        raise ValueError("render outside the repository before copying reviewed public artifacts")
    data = public_data(args.audit.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    (output / "concurrency-frontier.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    render(data, output)


if __name__ == "__main__":
    main()
