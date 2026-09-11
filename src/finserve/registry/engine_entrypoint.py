"""Verify baked model identity and explicit serving parameters before starting the pinned engine."""

import argparse
import importlib.metadata
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.model_assets import ModelManifest
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.registry.model_assets import verify_snapshot

BAKED_MANIFEST = Path("/opt/finserve/model-manifest.json")
MODEL_DIRECTORY = Path("/models")
RUNTIME_HOME = Path("/tmp/finserve")
ENV_PARAMETERS = {"use_v2_model_runner", "use_flashinfer_sampler"}


class VLLMParameters(ImmutableModel):
    """Versioned local experiment knobs exclude destinations, secrets and remote-code overrides."""

    dtype: Literal["float16", "bfloat16"] = "float16"
    max_model_len: int = Field(default=1024, ge=1, le=131072, strict=True)
    max_num_seqs: int = Field(default=8, ge=1, le=128, strict=True)
    max_num_batched_tokens: int = Field(default=2048, ge=1, le=131072, strict=True)
    gpu_memory_utilization: float = Field(default=0.4, gt=0, le=0.95, allow_inf_nan=False)
    enable_prefix_caching: bool = Field(default=False, strict=True)
    enforce_eager: bool = Field(default=True, strict=True)
    enable_chunked_prefill: bool = Field(default=True, strict=True)
    use_v2_model_runner: bool = Field(default=False, strict=True)
    use_flashinfer_sampler: bool = Field(default=False, strict=True)


def engine_arguments(
    profile: ServingProfileV1, manifest: ModelManifest, installed_version: str
) -> list[str]:
    """Translate only typed engine parameters after matching actual model and package identities."""
    if (
        profile.engine != "vllm"
        or profile.engine_version != installed_version
        or profile.model_manifest_sha256 != manifest.digest()
        or profile.tokenizer_manifest_sha256 != manifest.digest()
        or profile.model_revision != manifest.specification.revision
        or profile.tokenizer_revision != manifest.specification.revision
    ):
        raise ValueError("runtime package or model identity differs from serving profile")
    parameters = VLLMParameters.model_validate_json(profile.engine_parameters_json)
    names = {item.source.path for item in manifest.files}
    if (
        not {"config.json", "tokenizer_config.json"} <= names
        or not names & {"tokenizer.json", "tokenizer.model"}
        or not any(name.endswith(".safetensors") for name in names)
    ):
        raise ValueError("complete safetensors model and tokenizer snapshot required")
    arguments = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/models",
        "--tokenizer",
        "/models",
        "--served-model-name",
        profile.served_model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--generation-config",
        "vllm",
    ]
    for name, value in parameters.model_dump(exclude=ENV_PARAMETERS).items():
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            arguments.append("--" + ("" if value else "no-") + flag)
        else:
            arguments.extend(("--" + flag, str(value)))
    return arguments


def runtime_environment(profile: ServingProfileV1, inherited: Mapping[str, str]) -> dict[str, str]:
    """Own vLLM settings and resolve auth in memory, preventing inherited profile overrides."""
    parameters = VLLMParameters.model_validate_json(profile.engine_parameters_json)
    credential = (
        inherited.get(profile.credential_env) if profile.credential_env is not None else None
    )
    if profile.credential_env is not None and (
        not credential or len(credential) > 4096 or "\n" in credential or "\r" in credential
    ):
        raise ValueError("configured engine credential is unavailable")
    environment = {key: value for key, value in inherited.items() if not key.startswith("VLLM_")}
    environment.update(
        {
            "HOME": str(RUNTIME_HOME),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_CACHE_ROOT": str(RUNTIME_HOME / "vllm"),
            "VLLM_USE_V2_MODEL_RUNNER": str(int(parameters.use_v2_model_runner)),
            "VLLM_USE_FLASHINFER_SAMPLER": str(int(parameters.use_flashinfer_sampler)),
        }
    )
    if credential is not None:
        environment["VLLM_API_KEY"] = credential
    return environment


def read_bounded(path: Path, maximum_bytes: int) -> bytes:
    """Bound mounted configuration reads before deserializing runtime identity files."""
    with path.open("rb") as source:
        content = source.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise ValueError("runtime identity file exceeds byte limit")
    return content


def run(arguments: list[str] | None = None) -> None:
    """CPU verification checks image contents without acquiring a GPU or serving traffic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("/run/finserve/profile.json"))
    parser.add_argument("--verify-only", action="store_true")
    parsed = parser.parse_args(arguments)
    manifest = ModelManifest.model_validate_json(read_bounded(BAKED_MANIFEST, 2 * 1024**2))
    profile = ServingProfileV1.model_validate_json(read_bounded(parsed.profile, 128 * 1024))
    environment = runtime_environment(profile, os.environ)
    actual = verify_snapshot(MODEL_DIRECTORY, manifest.specification)
    if actual.digest() != manifest.digest():
        raise ValueError("mounted model differs from baked manifest")
    argv = engine_arguments(profile, manifest, importlib.metadata.version("vllm"))
    if parsed.verify_only:
        print(
            json.dumps(
                {
                    "model_manifest_sha256": actual.digest(),
                    "profile_sha256": profile.digest(),
                    "engine": "vllm",
                    "engine_version": profile.engine_version,
                    "scope": "CPU image/model verification; no GPU inference",
                }
            )
        )
        return
    RUNTIME_HOME.mkdir(parents=True, exist_ok=True)
    os.execve(argv[0], argv, environment)


if __name__ == "__main__":
    run()
