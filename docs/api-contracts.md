# API Contracts

## Public inference

OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, and optional embeddings if implemented.

## Control REST

`POST /api/benchmarks`, `GET /api/benchmarks/{id}`, `POST /api/deployments`, `POST /api/deployments/{id}/rollback`, `GET /api/models`, `GET /api/deployments`.

Control routes require stronger auth than inference.

## GraphQL explorer

Read-oriented benchmark/deployment/eval schema. Resolvers are paginated and query-complexity limited.

## gRPC

Use selected internal services only where a typed binary/streaming boundary improves the design. Do not wrap every engine interaction if Ray Serve already provides transport.

## Error envelope

```json
{"error":{"code":"OVERLOADED","message":"Serving capacity is saturated","retryable":true,"request_id":"..."}}
```

Schema/API versions stay explicit so old benchmark evidence remains readable.
