"""A separate counterfactual diagnostic corpus; never replaces failed uniform-color workloads."""

import argparse
import hashlib
import importlib
import io
from pathlib import Path
from typing import Any

from finserve.benchmark.runner import prepare_output, write_json

PROMPT = "Which bar is tallest? Answer with its color: red, green, or blue."


def create(output: Path) -> None:
    """Freeze three counterfactual bar images and expected answers before any model call."""
    directory = prepare_output(output)
    image, draw = importlib.import_module("PIL.Image"), importlib.import_module("PIL.ImageDraw")
    colors, heights = ["red", "green", "blue"], [280, 120, 200]
    records: dict[str, Any] = {
        "prompt": PROMPT,
        "scope": "new diagnostic; not release quality evidence",
    }
    for shift in range(3):
        raster = image.new("RGB", (448, 448), "white")
        canvas = draw.Draw(raster)
        for index, color in enumerate(colors):
            height = heights[(index + shift) % 3]
            canvas.rectangle((40 + index * 135, 380 - height, 120 + index * 135, 380), fill=color)
        target = io.BytesIO()
        raster.save(target, format="PNG")
        raw = target.getvalue()
        expected = colors[(-shift) % 3]
        name = expected + "-tallest.png"
        (directory / name).write_bytes(raw)
        records[name] = {"expected": expected, "sha256": hashlib.sha256(raw).hexdigest()}
    write_json(directory / "manifest.json", records)


def main() -> None:
    """Write an immutable fixture directory outside source before model diagnosis begins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    create(parser.parse_args().output)


if __name__ == "__main__":
    main()
