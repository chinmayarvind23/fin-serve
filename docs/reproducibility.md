# Reproducibility

A run manifest records its declared source, hardware, model/tokenizer revisions, engine configuration, workload hash, warmup and measured population, arrival mode and timeout. Experiment wrappers add actual Git status, clock anchors, raw GPU samples and completion/failure status. Retain package inventories, startup arguments and logs alongside each runtime experiment.

For an explicitly selected offline chat experiment, `RunConfig.prompt_mapping="chatml_roles_v1"`
extracts native roles from the exact system/user/empty-assistant ChatML grammar. It preserves
role content and rejects malformed or nested controls. Plain prompts remain user content.
The configured `system_prompt` is appended to the source system content with two newlines.
`benchmark.request_mapping.FORMAT_INSTRUCTION_V1` provides the declared general formatting
instruction for this experiment; it is not applied automatically. The mapper receives no
case identifiers, expected outputs or evaluator metadata, and never processes model answers.

Use `--request-api chat --prompt-mapping chatml_roles_v1` with the actual tokenizer template
digest when invoking the benchmark CLI. Configuration, mapping digest and comparison checks
distinguish this arm. The default `literal` mapping preserves historical completion/chat bytes
and hashes. Both engine configurations must use the same mapping in a release comparison;
earlier measurements cannot be relabeled. Mapper tests establish protocol behavior, not model
quality. The gateway does not implicitly interpret ChatML in user text.

Some earlier native recordings explicitly declare image and configuration digests unknown. Do not substitute a later container build for that missing identity. The runtime producer creates a separate verified model manifest and image from an exact source archive. Its model bytes are checked again before startup. A new source build or prompt-to-chat mapping is a new cohort, even if it reuses the same workload cases.

Comparison validation checks the load envelope and pinned model identities before recomputing ratios. Request throughput and token throughput have separate numerators. Quality correctness and output agreement have separate references. Failed requests remain in the population; missing observations remain unknown. [Benchmark methodology](benchmark-methodology.md) gives the formulas and actual file layout.

GPU kernels need not be bitwise deterministic across execution modes. Report exact output agreement, task correctness and statistical performance separately. Repeat and randomize paired runs when possible. The recorded eager/compiled result is one ordered pair on a shared workstation, which limits the strength of a causal performance claim.

Keep evidence in a new external directory. Immutable CAS references include namespace, size and hash; registry reads verify the bytes. Historical serialization is preserved when optional cohort fields are added. An authenticated cloud deployment additionally needs actual image receipts, region/instance details, configuration, health checks and billed or explicitly modelled cost inputs. Those fields cannot be inferred from a successful local test.
