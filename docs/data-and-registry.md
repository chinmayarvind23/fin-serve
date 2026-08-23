# Data and Registry

## Entities

ModelArtifact, EngineConfiguration, DeploymentRevision, BenchmarkDefinition, BenchmarkRun, EvaluationSuite, EvaluationRun, OptimizationRun, RollbackEvent, PricingSnapshot.

Benchmark definition describes what should run; benchmark run records one execution. This prevents silent definition changes after results exist.

## Storage

RDS: metadata/current promotion/audit.

S3: raw request records, Parquet metrics, traces, reports, model/image artifacts.

MLflow: experiment/eval comparison and promotion evidence.

Redis: ephemeral only.

Benchmark artifacts available for audit/reproduction.
