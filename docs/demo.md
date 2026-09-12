# Demo status and recording sequence

The recorded video and GIF are pending browser visual verification. The [free static results Space](https://huggingface.co/spaces/chinmayarvind/finserve) is deployed and passes HTTPS asset checks. The full explorer passes local HTTP and DOM checks; these are not a browser recording. Raw local evidence is retained outside the repository. Use the [local setup guide](run-free.md) for the running application and GPU inference.

The recording will show these implemented behaviors with visible run identities:

1. Open the evidence explorer and compare the 3,072-request eager and compiled runs. Show throughput, TTFT, p95 and physical GPU sample coverage.
2. Open the failed quality report and exact request outputs. Explain why the faster candidate was rejected.
3. Send an authenticated text request through Bun and FastAPI to an actual engine. Show authoritative final usage and the selected runtime identity.
4. Submit an image plus text request using the frozen bar-chart probe. Identify it as a functional example; show the separate failed uniform-color suite.
5. Submit a durable JAX/Flax visual job, retrieve its PNG and explain the reference model's scope.
6. Show a release decision and an actual warm-route rollback with revision-checked traffic. Separate the local fixture drill from any subsequent GPU or cloud lifecycle test.

Capture credentials only through masked input. Keep private prompts, environment values and unrelated desktop content out of the recording. Link the final video, GIF and reproducible commands here after capture; publish only the evidence actually shown.

The [results page](results.md), [architecture](HLD.md) and [commands](commands.md) remain usable when a GPU service is offline. No simulated cloud deployment or reconstructed benchmark screen substitutes for the live checks.
