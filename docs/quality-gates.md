# Quality Gates

Python: Ruff, Pyright strict, pytest, coverage, mutation testing on critical scheduling/metric/rollback logic.

TypeScript: strict mode, Biome, unit tests, Playwright screenshot/visual tests.

Guidelines: cyclomatic complexity <= 10 unless documented; functions generally <= 50 logical lines; >=85% deterministic-core coverage and >=95% critical contracts/routing/metrics/rollback/auth coverage.

Coverage is a floor, not correctness proof.

No optimization is promoted on performance alone. Security gates include dependency/secret/container/Terraform/K8s checks.

Comments explain inference math, concurrency invariants, architecture choices, assumptions/tradeoffs/scaling/failure semantics, not obvious syntax.
