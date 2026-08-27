# Deployment

## Environments

Local -> staging GPU -> production-shaped benchmark.

Production-shaped means realistic infrastructure/operations for exercising the design, not real customer traffic.

## Local

Docker Compose, FastAPI, Redis, Postgres, one small/local engine, Prometheus/Grafana.

## AWS

Terraform provisions VPC/subnets/security, EKS, GPU node groups, RDS, S3, Redis/ElastiCache if used, IAM, observability, and ingress prerequisites.

KubeRay manages Ray clusters/services. Keep CPU/system and GPU node groups separate.

## Autoscaling

Tune Serve replicas, Ray worker Pods, and EKS GPU nodes independently. Maintain warm GPU capacity when cold start violates interactive SLOs.

## Rollout

`candidate -> separate revision/canary -> readiness -> smoke -> quality -> controlled load -> promote`

## Rollback

Restore last known-good immutable revision. Measure `healthy_at - regression_detected_at`to happen within 94 seconds.
