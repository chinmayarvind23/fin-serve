# ADR 003: Keep vLLM primary and SGLang as a measured alternative

## Decision

Use separately managed vLLM for the primary pretrained text and image experiments. Retain the SGLang HTTP adapter and a separately installed runtime for comparison instead of combining incompatible engine dependencies in the gateway.

## Evidence and tradeoffs

Both engines ran the same pinned Qwen2.5-0.5B model locally. The retained SGLang development run completed 128 requests at 25.64 requests/s and 734.20 generated tokens/s, with p95 0.916 seconds. Its deterministic quality result failed. This short run does not establish superiority over the later 3,072-request compiled vLLM cohort.

vLLM has the completed sustained pair and pretrained image-understanding integration, so it remains the primary measured path. Engine startup flags, token accounting and template behavior are version-specific; the shared adapter validates streaming and final usage rather than assuming identical wire behavior. Neither backend has passed the required optimized-release correctness gate. See [results](../results.md).
