# ADR 001: Build measurement before distributed infrastructure

## Decision

Implement raw request accounting and deterministic quality evaluation before using distributed throughput as a result. Start with a transport fixture and inspectable reference decoder, then measure actual pretrained engines with the same timing contract.

## Evidence and consequences

The completed native pair retained 6,144 measured requests, exact token totals, failed quality checks and physical GPU coverage. Compiled execution increased token throughput 2.64-fold but worsened median TTFT and failed the frozen task suite. The harness also retained a speculation regression and substantial admission failures in the first two-engine routing experiment.

Keeping these outcomes separate prevents a faster or more complicated configuration from becoming an accepted release without useful output. The additional evidence and audit code increases implementation work, but makes failed experiments diagnosable. See [methodology](../benchmark-methodology.md) and [results](../results.md).
