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

## Frozen output-constraint maps

`RunConfig.output_constraints` is an optional `RequestConstraintMap` with schema version
`prompt-output-contracts-v1`. Each entry contains `prompt_sha256` and a required `constraint`,
either a bounded `OutputConstraint` or explicit `null` for an unconstrained request. Hash the
exact original UTF-8 prompt before chat mapping, using `benchmark.constraint_mapping.prompt_digest`.
Do not trim whitespace or use case identifiers, reporting families, evaluator kinds or expected
answers to select constraints. A map records declared syntax; it cannot prove how its author
selected that syntax. Prepare and review it from request text before inference.

Maps contain at most 256 unique prompt bindings and 256 KiB of canonical configuration.
Entry order is normalized. Duplicate hashes, missing bindings and unsupported schema fields
fail. The CLI reads bounded sidecar bytes and also rejects duplicate JSON keys. Every quality
and performance prompt needs an entry, including requests intentionally left unconstrained.
Freeze the same complete map for both collectors and both release configurations. Its version
and full digest enter request-mapping identity; a changed unused binding still changes identity.
Existing workload and golden-suite bytes stay unchanged.

Pass `--output-constraints <external-map.json> --constraint-transport native_vllm` for a native
vLLM endpoint, or select `finserve` for the public gateway contract. The shared mapper emits
`structured_outputs` or `output_constraint` respectively. Executable configs must declare
`engine="vllm"` and an `engine_config` that pins `structured_output_backend="xgrammar"`.
Producer load templates may leave both identities undeclared until actual build receipts resolve
them; they cannot execute directly. Constrained producers require native transport and xgrammar
in both frozen engine parameter sets.

Collection retains invalid raw output as failed work. Syntax validation also runs during raw
evidence reconstruction, so relabeling an invalid shape as successful fails verification.
Completion and chat quality evidence both require the full mapping identity. A valid shape may
still be wrong under the unchanged evaluator. Collector and sequential producer integration
tests cover these boundaries with explicit synthetic HTTP/Docker fixtures. Actual constrained GPU quality collection is complete; the selected 7B model scored 55/56 on consumed regressions and 42/48 on an independently frozen evaluation. Runtime logs retain grammar compilation observations, but isolated compilation and first-use latency costs remain unmeasured.

Some earlier native recordings explicitly declare image and configuration digests unknown. Do not substitute a later container build for that missing identity. The runtime producer creates a separate verified model manifest and image from an exact source archive. Its model bytes are checked again before startup. A new source build or prompt-to-chat mapping is a new cohort, even if it reuses the same workload cases.

Comparison validation checks the load envelope and pinned model identities before recomputing ratios. Request throughput and token throughput have separate numerators. Quality correctness and output agreement have separate references. Failed requests remain in the population; missing observations remain unknown. [Benchmark methodology](benchmark-methodology.md) gives the formulas and actual file layout.

GPU kernels need not be bitwise deterministic across execution modes. Report exact output agreement, task correctness and statistical performance separately. Repeat and randomize paired runs when possible. The recorded eager/compiled result is one ordered pair on a shared workstation, which limits the strength of a causal performance claim.

Keep evidence in a new external directory. Immutable CAS references include namespace, size and hash; registry reads verify the bytes. Historical serialization is preserved when optional cohort fields are added. An authenticated cloud deployment additionally needs actual image receipts, region/instance details, configuration, health checks and billed or explicitly modelled cost inputs. Those fields cannot be inferred from a successful local test.
