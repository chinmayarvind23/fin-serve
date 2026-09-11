# ADR 013: Count proxy, engine and GPU capacity separately

## Decision

Treat Ray Serve proxy replicas, Ray worker Pods and physical GPU nodes as separate capacity controls. A CPU routing proxy cannot create model memory or GPU execution capacity.

## Current implementation

Named proxies target distinct external engine endpoints. Routing uses bounded leases, observed native running/waiting/KV gauges and explicitly shared physical GPU samples. A single GPU UUID remains one device even when two engines report work. Least-load and adaptive policies use the same eligibility constraints.

The local owned-process experiment can drain, stop, restart and deliberately fail its own model processes. Linux PID descriptors fence signals against PID reuse. This is an operator-controlled scaling and failure experiment, not fleet autoscaling.

The Kubernetes staging chart has fixed CPU capacity and one separately managed GPU engine. Terraform bounds the GPU node group and defaults its desired count to zero. Cloud autoscaling, cold-node recovery and capacity/quality behavior under changing replicas have not been established by local tests. A measured capacity frontier and actual staging drills must precede those claims.
