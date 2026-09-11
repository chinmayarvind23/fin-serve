# Reproducibility

A run manifest records its declared source, hardware, model/tokenizer revisions, engine configuration, workload hash, warmup and measured population, arrival mode and timeout. Experiment wrappers add actual Git status, clock anchors, raw GPU samples and completion/failure status. Retain package inventories, startup arguments and logs alongside each runtime experiment.

Some earlier native recordings explicitly declare image and configuration digests unknown. Do not substitute a later container build for that missing identity. The runtime producer creates a separate verified model manifest and image from an exact source archive. Its model bytes are checked again before startup. A new source build or prompt-to-chat mapping is a new cohort, even if it reuses the same workload cases.

Comparison validation checks the load envelope and pinned model identities before recomputing ratios. Request throughput and token throughput have separate numerators. Quality correctness and output agreement have separate references. Failed requests remain in the population; missing observations remain unknown. [Benchmark methodology](benchmark-methodology.md) gives the formulas and actual file layout.

GPU kernels need not be bitwise deterministic across execution modes. Report exact output agreement, task correctness and statistical performance separately. Repeat and randomize paired runs when possible. The recorded eager/compiled result is one ordered pair on a shared workstation, which limits the strength of a causal performance claim.

Keep evidence in a new external directory. Immutable CAS references include namespace, size and hash; registry reads verify the bytes. Historical serialization is preserved when optional cohort fields are added. An authenticated cloud deployment additionally needs actual image receipts, region/instance details, configuration, health checks and billed or explicitly modelled cost inputs. Those fields cannot be inferred from a successful local test.
