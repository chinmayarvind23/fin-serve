# Failure Modes

| Failure | Behavior | Recovery |
|---|---|---|
| replica crash | reroute eligible work | restart replica |
| GPU OOM | classify failure, stop blind retry | reduce load/config or larger pool |
| queue saturation | bounded overload rejection | scale or reduce admission |
| Redis unavailable | bypass optional cache / safe fallback | reconnect |
| RDS unavailable | keep already-running data plane where safe; block state-changing control ops | restore DB |
| S3 unavailable | do not start model needing missing artifact | retry/backoff |
| telemetry backend unavailable | serving continues with fallback/local telemetry | restore exporter |
| GPU node loss | replica unavailable | reschedule/provision |
| quality regression | do not promote / roll back | known-good revision |
| latency regression | stop promotion / roll back | analyze ablation |
| client disconnect | abort generation | cleanup KV/resources |
| speculation low acceptance | correct outputs continue | disable/tune speculation |
| JAX recompilation storm | latency spike | normalize shapes/precompile |
| Airflow duplicate retry | no duplicate release side effect | immutable idempotent task |

Retry transient 429/503/network artifact failures. Do not blindly retry invalid input, auth failure, checksum mismatch, incompatible model, or deterministic quality regression.

Chaos later injects replica/node failure, overload, bad release, and telemetry exporter failure with expected behavior defined first.
