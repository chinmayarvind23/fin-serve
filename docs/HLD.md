# High-Level Design

```mermaid
flowchart LR
    C[Clients] --> G[TypeScript/Bun Gateway]
    G --> A[FastAPI/OpenAI Ingress]
    A --> R[Ray Serve]
    R --> V[vLLM Pool]
    R --> S[SGLang Pool]
    R --> J[JAX/Flax Multimodal Pool]
    R --> M[Engine Multimodal Pool]
    A --> REDIS[Redis]
    A --> PG[RDS PostgreSQL]
    PG --> MF[MLflow]
    MF --> S3[S3]
    AIR[Airflow] --> MF
    AIR --> EKS[EKS/KubeRay]
    EKS --> R
    A --> OT[OpenTelemetry]
    R --> OT
    OT --> OBS[Prometheus/Grafana/CloudWatch]
    LF[Langfuse] --> OBS
```

## Boundaries

Gateway: auth, API versioning, validation, coarse rate limits. Not GPU scheduling.

Ray Serve: distributed replica lifecycle, routing, autoscaling, placement.

Engine: KV cache, token scheduling, continuous batching, speculation, engine metrics.

Airflow: offline artifact lifecycle. Never blocks a token request.

## Data plane

`client -> ingress -> Ray Serve -> engine -> stream`

## Control/evidence plane

`registry -> Airflow -> deployment -> benchmark -> MLflow -> promote/rollback`

Already-running serving should not require MLflow/Airflow/RDS for every token.
