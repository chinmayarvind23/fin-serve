"""Diagnose pretrained image input through direct vLLM and official Transformers execution."""

import argparse
import hashlib
import importlib
import importlib.metadata
import io
import time
from pathlib import Path
from typing import Any

import httpx

from finserve.benchmark.runner import prepare_output, write_json
from finserve.contracts.vision import VISION_MODEL, VISION_REVISION
from finserve.multimodal.images import prepare_png
from finserve.multimodal.vision_benchmark import PROMPT


def diagnose_vllm(raw: bytes, base_url: str, prompt_override: str | None = None) -> dict[str, Any]:
    """Compare the frozen prompt to a descriptive diagnostic without altering measured cohorts."""
    image = prepare_png(raw)
    records: list[dict[str, Any]] = []
    with httpx.Client(timeout=90, trust_env=False) as client:
        prompts = (
            [prompt_override]
            if prompt_override
            else [PROMPT, "Describe this image.", "What color is shown: red, green, or blue?"]
        )
        for prompt in prompts:
            content = [
                {"type": "image_url", "image_url": {"url": image.data_url()}},
                {"type": "text", "text": prompt},
            ]
            payload = {
                "model": VISION_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 32,
                "temperature": 0,
                "stream": False,
            }
            started = time.perf_counter()
            response = client.post(base_url.rstrip("/") + "/chat/completions", json=payload)
            response.raise_for_status()
            records.append(
                {
                    "prompt": prompt,
                    "response": response.json(),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    return {
        "canonical_sha256": image.sha256,
        "records": records,
        "note": "descriptive prompts are diagnostics only; frozen benchmark remains unchanged",
    }


def diagnose_transformers(raw: bytes, directory: Path, prompt: str = PROMPT) -> dict[str, Any]:
    """Run the official model/processor locally after all vLLM GPU owners have stopped."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    pil = importlib.import_module("PIL.Image")
    image = prepare_png(raw)
    processor = transformers.AutoProcessor.from_pretrained(
        VISION_MODEL,
        revision=VISION_REVISION,
        local_files_only=True,
        min_pixels=3136,
        max_pixels=200704,
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with pil.open(io.BytesIO(image.png)) as raster:
        inputs = processor(text=[text], images=[raster], return_tensors="pt")
    metadata = {
        "chat_text": text,
        "input_ids": inputs["input_ids"].tolist(),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "pixel_values_sha256": hashlib.sha256(inputs["pixel_values"].numpy().tobytes()).hexdigest(),
    }
    write_json(directory / "processor.json", metadata)
    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        VISION_MODEL,
        revision=VISION_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    inputs = inputs.to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()
    generated = output[:, inputs["input_ids"].shape[1] :]
    return {
        "canonical_sha256": image.sha256,
        "prompt": prompt,
        "output": processor.batch_decode(generated, skip_special_tokens=True),
        "generated_ids": generated.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
        "processor": metadata,
        "torch": torch.__version__,
        "note": "single diagnostic generation; not a performance benchmark",
    }


def main() -> None:
    """Require explicit mode and evidence location, preserving failed diagnostic attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("vllm", "transformers"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()
    directory = prepare_output(args.output)
    manifest: dict[str, Any] = {
        "status": "running",
        "mode": args.mode,
        "model": VISION_MODEL,
        "revision": VISION_REVISION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transformers": importlib.metadata.version("transformers"),
    }
    write_json(directory / "manifest.json", manifest)
    try:
        raw = args.image.read_bytes()
        (directory / "input.png").write_bytes(raw)
        result = (
            diagnose_vllm(raw, args.base_url, args.prompt)
            if args.mode == "vllm"
            else diagnose_transformers(raw, directory, args.prompt or PROMPT)
        )
        write_json(directory / "result.json", result)
        manifest["status"] = "succeeded"
    except BaseException as exc:
        manifest["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
