# Threat model

Assets include inference capacity, credentials, model/artifact bytes, durable job state, deployment authority and the integrity of benchmark claims. The local deployment trusts its operator, filesystem and configured engine/collector destinations. It does not claim isolation against a malicious administrator or a hostile process with write access to its artifact volume.

| Threat | Implemented boundary | Remaining limit |
| --- | --- | --- |
| Large or excessive requests | Actual byte counting, typed limits, admission and optional Redis quota | One shared service principal is not multi-user identity management |
| Inference client requesting deployment | No deployment action in inference or read-only GraphQL routes | CLI/filesystem/adapter access must be restricted by the deployment operator |
| Corrupt or substituted model/artifact | Pinned commits, per-file checksums, CAS verification and image labels | Trusted-volume concurrency and actual running-container identity still require operational controls |
| Misleading benchmark gain | Frozen workload/grader, raw failures, separate token accounting and recomputed gates | A shared-workstation ordered pair still has environmental confounds |
| Duplicate or stale control action | Immutable identities, transaction/CAS generations and action receipts | A local SQLite controller is not a distributed consensus service |
| Cancellation freeing capacity early | Explicit response/generator/worker ownership and acknowledged retirement | Native kernel interruption and node recovery need separate evidence |
| Trace data disclosure | Metadata sanitization, no baggage forwarding, bounded authenticated collector transport | Private raw evaluation artifacts intentionally contain inputs and outputs |
| Dependency or deployment compromise | Locked packages, pinned bases and validated infrastructure configuration | No blanket claim of completed SBOM/scanning or enforced cloud policy |
| Registry or collector unavailable | Gate failures block promotion; telemetry failures do not fail inference | Telemetry loss reduces evidence completeness and must remain visible |

Authentication failure rejects the relevant request. Integrity failure rejects the artifact. Missing or failed quality evidence prevents promotion. No optimizer may reinterpret unavailable checks as passing ones. [Security](security.md) and [failure behavior](failure-modes.md) state the current deployment scope.
