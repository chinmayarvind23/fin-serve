# Recorded results

Raw artifacts remain in the private workspace `resources/fin_serve/evidence/`. This repository publishes the methods, workload and result summary. No optimized release has passed the required correctness gate.

## Sustained native text comparison

Both runs used clean source `241615dc04e3623e078500de7bded65f43e30ab6`, vLLM 0.29.0, Qwen2.5-0.5B-Instruct model/tokenizer revision `7ae557604adf67be50417f59c2c2f167def9a775`, one RTX 4070 Laptop GPU, concurrency 16, context 2,048, and disabled prefix caching/speculation. The workload hash is `ec5b6d6b04fff5ad77d8ad3205e0c9c5e9f90dcc0aed5380792d7d4754789e54`.

| Quantity | `vllm-sustained-eager-01` | `vllm-sustained-compiled-01` |
| --- | ---: | ---: |
| Measured requests | 3,072 | 3,072 |
| Separate warmup requests | 64 | 64 |
| Successful / failed | 3,072 / 0 | 3,072 / 0 |
| Measured interval | 236.4045561 s | 88.8175381 s |
| Authoritative generated tokens | 88,301 | 87,711 |
| Successful requests/s | 12.9946734 | 34.5877635 |
| Generated tokens/s | 373.5164899 | 987.5414459 |
| Median server TTFT | 105.3009 ms | 128.5691 ms |
| Median client TTFT | 112.9524 ms | 159.4531 ms |
| Successful-request end-to-end p95 | 2.13148195 s | 0.804339235 s |
| Mean physical GPU utilization | 29.105881% | 57.914404% |
| GPU sample coverage | 100% | 100% |

The gains are 2.66× successful-request throughput and 2.64× generated-token throughput. Exact paired output matches were 2,479/3,072 (80.6966%). A separate 32-case suite, hash `998b1f5dd448c2dffe247bc6fe89b5251699fb1b3fc170dc0fb427662706b463`, produced 31.25% candidate correctness and 75% baseline parity. The quality gate failed. Baseline self-parity also fails on invalid JSON; it is not an external-baseline comparison.

The raw audit recomputes populations, durations, token counts, percentiles and GPU integration. Its reusable command is `scripts/audit_sustained_comparison.py`; retained reports are `sustained-audit-01/comparison.json` and `.md`. Each native run records `image_digest=undeclared` and `config_digest=undeclared`. Built images from other runs cannot fill those fields afterward.

This is one baseline-then-candidate pair, not a randomized repeated study. Workstation activity, ordering and thermal effects remain possible confounds. All 6,144 measured requests completed, which is an observed transport outcome and not a 99.95% production availability guarantee. Local hardware produced no cloud billing evidence; no 37% GPU-cost claim is supported.

![Six-panel comparison of eager and compiled serving with the rejected quality gate](assets/sustained-comparison.png)

The figure is available as [SVG](assets/sustained-comparison.svg) with [aggregate source data](assets/sustained-comparison.json). Its data records the original audit SHA-256 and run IDs. Rebuild into a fresh directory with the pinned Matplotlib script:

```sh
uv run --script scripts/plot_sustained_comparison.py --audit "$FINSERVE_AUDIT_JSON" --output "$FINSERVE_NEW_FIGURE_DIRECTORY"
```

The plot consumes the recomputed audit; it does not replace raw-record verification.

## Rejected speculation experiment

At concurrency 16, short 256-request development runs measured 40.29 requests/s without speculation versus 20.59 with three-token n-gram proposals. Candidate acceptance was 30.04% across its two recordings including warmup. p95 rose from 0.533 to 1.387 seconds. The unchanged correctness suite failed. The preferred profile keeps speculation disabled; the failed records are retained in `ngram-load-c16-01` and `quality-ngram-01`.

## Rejected 1.5B image cohort

`image-chat-15b-cohort-01` used verified Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, model manifest `95334c7275d05bb8e71d1962d57e57a743c1b8aff95329cbb850b8ee5aaa370b`, and actual image `sha256:328f7cdadd2f494049560c12dc9f2a729473558807a1b3b49a971d6b5b11a568` built from source `d84c859cdda2caca616f1ca157d4a33e88d37f5f`. Both profiles used float16, eager execution, context/token limits 1,024, four engine sequences and memory fraction 0.45. Prefix caching alone changed from disabled to enabled. The native-chat mapping and generic format instruction were frozen before inference.

The second resident worker could not allocate KV cache on the 8 GB GPU. That failed startup remains retained. A sequential fallback was declared before any quality or performance requests; it does not establish two warm 1.5B replicas or warm rollback.

| Development quantity | Baseline | Prefix-cache candidate |
| --- | ---: | ---: |
| Measured successful requests | 128/128 | 128/128 |
| Separate warmup requests | 16 | 16 |
| Successful requests/s | 8.1765 | 11.2065 |
| Generated tokens/s | 75.6324 | 103.6602 |
| Median client TTFT | 536.62 ms | 375.30 ms |
| Successful-request end-to-end p95 | 1.8421 s | 1.4212 s |
| Mean physical GPU utilization | 33.78% | 42.17% |

The development workload hash is `927f69f403f87f50502126aa5f7f2f0600c521cc7e7200a22be7173fc026845d`; run IDs are `5c6c62f2-3297-4395-8d4f-5f49938b8e68` and `adaac19b-35f9-4381-bf54-2ebe82bbdbf4`. Client concurrency was eight. These short sequential intervals cannot replace the sustained native comparison or establish a cloud cost estimate.

Both models completed all 36 quality HTTP requests but scored 3/4 on the separately frozen format holdout and 10/32 on the unchanged release suite. Typed parity was 75%. Markdown-fenced JSON and exact numeric/yes-no formatting failures remained failures; outputs and graders were not normalized afterward. The actual canonical registration and lifecycle rejected the candidate for quality, with zero deployment callbacks. The planned sustained phase was not offered because its quality precondition failed. Raw quality responses, durable producer receipts, image/source identities and rejection state remain in the cohort directory.

## Multimodal evidence

Pinned Qwen2-VL-2B-Instruct revision `895c3a49bc3fa70a340399125c650a463535e71c` runs through the actual image/text adapter. The final uniform-color preprocessing comparison retained 36 successful requests and 18/18 exact local/HTTP output pairs, but 0/36 correct color answers. The rejected semantic result remains in `vision-stage-run-03-final`; earlier fp16 and bf16 cohorts also remain.

Final preprocessing medians were 9.25 ms local and 27.25 ms HTTP; end-to-end medians were 847.28 ms and 875.58 ms. The shared workstation was active, and the manifest records a dirty tree plus eight archived source files. These values measure CPU image normalization and actual HTTP transfer; vision encoding and language decoding remain together in vLLM.

Three separately frozen counterfactual bar charts produced correct red/green/blue answers through vLLM and through the integrated Bun/FastAPI path (`vision-integrated-edge-smoke-01.json`). That is a functional probe, not a release accuracy estimate. The JAX/Flax visual generator uses untrained reference weights and has a separate scope.

## Routing and rollback

A 64-request real Ray-to-GPU smoke completed 64/64 requests. It verifies the transport/lease bridge to one actual engine, not a multi-GPU routing gain. A later four-cohort comparison used two distinct vLLM processes on one physical GPU. It completed 21, 19, 24 and 24 of 64 requests in least-load/adaptive/adaptive/least-load order; all 168 failures preceded content, and Ray logs identify capacity rejection. The failed session provides no adaptive gain. Scale-down and a survivor request succeeded, then restart preflight failed. A separately reviewed bounded admission wait and Linux port-reuse correction require a new GPU session. See [scheduling](scheduling-and-batching.md).

A local warm-route fault drill restored the expected healthy revision in 0.680 seconds after detection. It retained one deliberately failed request and verified actual HTTP traffic after the route CAS. It switches between already-running fixture endpoints and does not include image pull, model load, cloud node replacement or GPU restart.

## Targets still requiring evidence

94 requests/s, improved median TTFT at the selected envelope, 99.2% quality parity, 81% mean GPU utilization, 37% lower equivalent-quality GPU cost and a 94-second cloud rollback remain unachieved or unmeasured. Successful local checks, configuration validation and mocked cloud tests do not establish deployed AWS/Hugging Face operation.
