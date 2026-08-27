# Airflow Model Lifecycle

Airflow coordinates slow dependency-heavy work, never token requests.

```text
verify_input_spec
 -> fetch_model
 -> checksum/license record
 -> optional optimization/quantization
 -> build runtime image
 -> push image
 -> quality eval
 -> performance smoke
 -> register candidate in MLflow
 -> infrastructure preflight
 -> deploy canary
 -> post-deploy smoke
 -> promote or rollback
```

Tasks use immutable model revision, optimization hash, image digest, and deployment revision so retries are idempotent.

Optional optimization branches can include vLLM tuning, quality-safe quantization, TensorRT-LLM comparison, speculation artifact, or JAX compile artifact. Keep only evidence-backed branches.
