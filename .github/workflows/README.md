# Verification workflows

`quality.yml` runs Ruff, strict Pyright, Python coverage and semantic mutation checks;
Bun HTTP/DOM tests, lint/type checks and builds; actual Helm rendering against the
pinned RayService CRD; and pinned Prometheus configuration/alert tests. The Helm
binary archive is checksum verified. Infrastructure tests do not contact a cluster.
Python and Bun checks do not establish live GPU or browser behavior.

`performance-gate.yml` runs the release gate on explicitly supplied, independently
registered evidence through a configured self-hosted runner. It does not collect a
GPU benchmark on GitHub's standard runners or approve incomplete evidence. Cloud
deployment and runtime health require their own retained execution receipts.
