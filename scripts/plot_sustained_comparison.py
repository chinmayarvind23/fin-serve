# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.0"]
# ///
"""Render an audited comparison with its rejected quality result beside performance metrics."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def public_data(raw: bytes) -> dict[str, Any]:
    """Project only reviewed aggregate fields; prompts, host paths and GPU UUIDs stay private."""
    report = json.loads(raw)
    if len(report["runs"]) != 2 or len(report["quality"]) != 2:
        raise ValueError("a paired audited comparison is required")
    if [run["run_id"] for run in report["runs"]] != [
        "914c1f2b-6dde-4d99-850c-ca099b6841de",
        "a4c55327-0e45-4506-9987-3b234327dcff",
    ]:
        raise ValueError("this figure labels the original native sustained pair only")
    return {
        "source_audit_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": report["scope"],
        "candidate_quality_gate_passed": report["candidate_quality_gate_passed"],
        "runs": [
            {
                "name": run["name"],
                "run_id": run["run_id"],
                "workload_hash": run["workload_hash"],
                "source_revision": run["configuration"]["revision"],
                "model_revision": run["configuration"]["model_revision"],
                "image_digest": run["configuration"]["image_digest"],
                "summary": run["summary"],
                "warmup_records": run["warmup_records"],
                "gpu_utilization_percent": run["gpu"]["average_gpu_utilization_percent"],
                "gpu_coverage": run["gpu"]["coverage"],
                "quality": {
                    key: quality["result"][key]
                    for key in (
                        "case_count",
                        "candidate_accuracy",
                        "parity",
                        "passed",
                        "suite_hash",
                    )
                },
            }
            for run, quality in zip(report["runs"], report["quality"], strict=True)
        ],
        "limitations": report["limitations"],
    }


def render(data: dict[str, Any], output: Path) -> None:
    """Use zero-based axes and explicit units; a speed gain never removes the quality warning."""
    runs = data["runs"]
    colors = ["#78899e", "#087f8c"]
    ink, muted, paper = "#152b43", "#506174", "#f7f8fa"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": ink})
    figure, axes = plt.subplots(2, 3, figsize=(13.8, 8.6), facecolor=paper)
    figure.subplots_adjust(left=0.07, right=0.97, top=0.77, bottom=0.23, wspace=0.38, hspace=0.65)
    figure.text(0.07, 0.94, "FINSERVE / MEASURED RESULTS", size=10, weight="bold", color=muted)
    figure.text(0.07, 0.89, "Native text serving: performance and quality", size=23, weight="bold")
    figure.text(
        0.07,
        0.848,
        "One RTX 4070 Laptop GPU · Qwen2.5-0.5B · vLLM 0.29.0 · concurrency 16",
        color=muted,
        size=11,
    )
    figure.legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in zip(colors, ("Eager baseline", "Compiled candidate"), strict=True)
        ],
        loc="upper right",
        bbox_to_anchor=(0.977, 0.962),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    metrics = [
        (
            "Successful requests / s",
            "Higher is better",
            [r["summary"]["requests_per_second"] for r in runs],
            2,
        ),
        (
            "Generated tokens / s",
            "Higher is better",
            [r["summary"]["tokens_per_second"] for r in runs],
            2,
        ),
        (
            "Median server TTFT · ms",
            "Lower is better",
            [r["summary"]["server_ttft_median_s"] * 1000 for r in runs],
            2,
        ),
        (
            "End-to-end p95 · seconds",
            "Lower is better",
            [r["summary"]["e2e_p95_s"] for r in runs],
            3,
        ),
        (
            "Physical GPU utilization · %",
            "Time-weighted observed activity",
            [r["gpu_utilization_percent"] for r in runs],
            2,
        ),
        (
            "Task correctness · %",
            "Separate frozen 32-case suite",
            [r["quality"]["candidate_accuracy"] * 100 for r in runs],
            2,
        ),
    ]
    for axis, (title, subtitle, values, decimals) in zip(axes.flat, metrics, strict=True):
        axis.set_facecolor(paper)
        axis.bar([0, 1], values, width=0.5, color=colors, zorder=3)
        axis.set_title(title, loc="left", pad=26, size=12, weight="bold")
        axis.text(0, 1.075, subtitle, transform=axis.transAxes, size=9, color=muted)
        axis.set_ylim(0, 108 if title.endswith("%") else max(values) * 1.25)
        axis.set_xticks([0, 1], ["Eager", "Compiled"])
        axis.tick_params(axis="both", length=0, labelsize=9, colors=muted, pad=7)
        axis.grid(axis="y", color="#e1e6eb", linewidth=0.8, zorder=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for position, value in enumerate(values):
            axis.annotate(
                f"{value:,.{decimals}f}",
                (position, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                weight="bold",
                size=11,
            )
        if title.startswith("Task"):
            axis.axhline(100, color="#a24433", linewidth=1, linestyle="--")
            axis.text(
                0.98,
                0.81,
                "Required: 100%",
                transform=axis.transAxes,
                ha="right",
                color="#a24433",
                size=9,
            )
    quality = runs[1]["quality"]
    verdict = "PASSED" if data["candidate_quality_gate_passed"] else "REJECTED"
    figure.text(
        0.07,
        0.137,
        f"{verdict} · {quality['candidate_accuracy']:.2%} correctness / "
        f"{quality['parity']:.2%} baseline parity on {quality['case_count']} separate cases",
        size=12,
        weight="bold",
        color="#973928",
        bbox={"facecolor": "#f9eae5", "edgecolor": "none", "pad": 12},
    )
    figure.text(
        0.07,
        0.084,
        "3,072 measured requests + 64 separate warmups per run. All measured requests completed.",
        size=10,
        color=muted,
    )
    figure.text(
        0.07,
        0.056,
        "One ordered pair; no randomized repeats, deployment image identity "
        "or measured cloud cost.",
        size=10,
        color=muted,
    )
    figure.savefig(output / "sustained-comparison.png", dpi=160, facecolor=paper)
    figure.savefig(output / "sustained-comparison.svg", facecolor=paper, metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    """Require a fresh artifact directory so a later plot cannot overwrite retained evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = public_data(args.audit.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sustained-comparison.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    render(data, args.output)


if __name__ == "__main__":
    main()
