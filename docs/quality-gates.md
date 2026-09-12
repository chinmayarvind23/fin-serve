# Verification and release gates

Quality CI runs Ruff lint and formatting, strict Pyright, pytest with an 85% aggregate coverage floor and targeted semantic mutation checks. The Bun job runs Biome, strict TypeScript, real HTTP edge tests, DOM-emulated client tests and both builds. DOM emulation is not a browser visual review; use browser checks for rendered behavior.

Project guidance targets at least 85% deterministic-core coverage and 95% for critical contracts, routing, accounting, authentication and rollback. Coverage supports review but does not establish correctness. Tests exercise resource ownership, malformed input, stale identities and failure paths instead of counting only successful responses.

Runtime promotion is a different gate from source CI. It recomputes raw benchmark summaries and frozen task quality, checks the declared comparison envelope and requires matching immutable model/configuration/image identities. Canonical serving profiles bind the actual measured endpoint. A missing or invalid evidence input fails; no faster candidate bypasses the quality requirement.



Comments explain inference mechanics, concurrency ownership, identity rules, measurement definitions and material tradeoffs. Frozen workloads and graders are not edited after failures to obtain a desired score. Rejected runs stay available for audit.
