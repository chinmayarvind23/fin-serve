# Redis quota and optional cache

`redis_state.py` implements shared quota decisions with bounded Lua operations and expiring keys, plus optional cached values. Quota errors fail closed; cache read errors are misses. HMAC-derived principal keys avoid putting raw principals into Redis keys. Instances share a quota only when the secret, namespace and policy agree.

Redis does not own engine batching or durable visual-job truth. The latter uses the job store and worker generation fence. See [security](../../../docs/security.md) and [local Redis setup](../../../infra/docker/README.md) for configuration and failure behavior.
