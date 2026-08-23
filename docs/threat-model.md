# Threat Model

## Assets

GPU capacity, model weights, provider credentials, AWS credentials, deployment authority, benchmark integrity, user data, registry state.

## Main threats

- resource exhaustion with huge/many requests,
- compromised model artifact or unsafe remote code,
- inference credential escalating to deployment privilege,
- benchmark manipulation by dropping slow failures or hidden cache,
- spoofed/incomplete metrics,
- dependency/container compromise,
- async queue abuse.

## Mitigation

Admission/quota/limits, artifact allowlist/pin/checksum, separate auth scopes, client-side accounting and immutable workload manifest, cross-check raw metrics, lockfiles/scans/SBOM, bounded async job policy.

## Fail closed

Artifact integrity failure -> do not load.

Authorization unavailable -> deny control action.

Quality gate unavailable -> do not promote.
