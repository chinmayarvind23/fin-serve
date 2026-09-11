# Frozen output grading

`quality.py` defines frozen cases, exact-string and typed-JSON grading, correctness against expected answers and parity against retained baseline answers. Identical invalid JSON does not pass typed parity. Numeric type, missing fields and exact-format failures remain visible; the evaluator does not strip Markdown fences or repair model output after observing a failure.

The recorded 32-case suite is separate from the load workload. Raw output equality and HTTP success have different meanings. See [evaluation methods](../../../docs/evaluation.md), [suite data](../../../evals/golden/correctness-32-v1.json) and [recorded failed gates](../../../docs/results.md).
