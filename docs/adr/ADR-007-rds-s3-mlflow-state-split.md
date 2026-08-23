# ADR 007 rds s3 mlflow state split

## Decision

RDS owns structured truth, S3 large artifacts, MLflow experiment/eval evidence; Redis is not durable truth.

## Evidence

Validate the decision with the benchmark, eval, telemetry, or failure test appropriate to the component.
