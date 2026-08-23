# Reproducibility

Every run records git SHA, image digest, lock hashes, model/tokenizer revision, engine/version, PyTorch/JAX/CUDA/driver, GPU model/count, Ray/KubeRay/K8s config, engine arguments, routing policy, corpus hash, seed, warmup, request count, arrival model, concurrency, pricing snapshot, telemetry sampling.

GPU inference may not be bitwise deterministic across kernels/runtimes. Distinguish exact determinism, statistically equivalent performance, and task-quality parity.

A comparison is invalid if a material variable changes without declaration. Different GPU hardware can support a hardware comparison.

Release workload manifests are hashed before headline evidence runs.
