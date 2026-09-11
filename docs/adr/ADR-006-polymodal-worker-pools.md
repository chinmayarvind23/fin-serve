# ADR 006: Separate workers where contracts and ownership differ

Status: accepted for implemented local boundaries; independent GPU stage pools unproven.

## Decision

Text engine proxies, pretrained VLM requests and JAX visual jobs use explicit contracts. JAX executes behind gRPC and a durable single-coordinator SQLite job path. Pretrained vision sends canonical PNG chat requests to an external vLLM process. CPU PNG preparation can run in a separate authenticated HTTP service.

## Evidence

Real CPU gRPC tests cover admission, artifact integrity, deadline and cancellation. Native JAX work drains before capacity is freed; instance barriers and job generations fence ambiguous or late attempts. The random visual model demonstrates image-conditioned autoregression and compiled parity without a semantic quality claim.

The pretrained preprocessing comparison retained 36 transport successes and 18 exact output pairs, but zero correct uniform-color answers. HTTP measurements cover CPU preparation and round-trip overhead. Encoder and decoder remain together in vLLM. Three separate bar-chart probes passed through the integrated route; their scope does not change the failed color cohort.

## Alternatives and consequences

Putting all numerical runtimes in the gateway would couple dependencies and native cancellation lifetimes. A separate worker makes ownership explicit, at the cost of transport and worker lifecycle management. In-band PNG output is bounded to the small reference artifact; large outputs need a separate storage and integrity design.

Independent encoder/decoder GPU pools could alter batching and memory behavior, but no such gain is measured here. The CPU-hop experiment cannot justify that architecture. Distributed job recovery, cloud scheduling and multiple coordinators remain outside the verified local implementation. See [multimodal serving](../multimodal-serving.md) for limits and restart semantics.
