"""Refuse a local GPU launch when another compute workload owns the device."""

import json
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    """Check a single idle NVIDIA GPU; this observation is not an exclusive reservation."""
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        print("NVIDIA driver tools unavailable. Use the CPU explorer or configure GPU support.")
        return 1
    try:
        memory = (
            subprocess.run(
                [
                    executable,
                    "--query-gpu=memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            .stdout.strip()
            .splitlines()
        )
        applications = subprocess.run(
            [executable, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        if len(memory) != 1:
            raise ValueError("this local recipe expects exactly one visible GPU")
        total, used = (int(value.strip()) for value in memory[0].split(","))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"GPU availability could not be verified: {type(error).__name__}")
        return 1
    available = not applications and total >= 7500 and used <= 512
    print(
        json.dumps(
            {
                "total_mib": total,
                "used_mib": used,
                "compute_apps": applications,
                "ready_for_local_recipe": available,
            }
        )
    )
    if not available:
        print(
            "GPU busy or too small for this recipe. Arrange an idle window; no process was stopped."
        )
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
