---
title: FinServe
emoji: 🖥️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# FinServe explorer container

This CPU container runs the Bun explorer and Python GraphQL read service. Open `/explorer` from the product introduction. Inference engines run as separate services.

Set distinct `FINSERVE_API_KEY` and `FINSERVE_WEB_KEY` secrets. The first protects the internal service; the second grants browser access. Both must be printable ASCII strings between 16 and 4,096 characters. Startup rejects missing or equal keys.

The initial SQLite registry is empty. Import your own runs and mount persistent storage for durable use. Data under `/data` is ephemeral otherwise. The container contains no private requests, credentials, or model weights.

Bun listens on port 7860; Python listens on loopback 8050. The supervisor checks an authenticated registry read before starting the web listener and stops both owned children on termination. The read proxy accepts bounded GraphQL requests and exposes no deployment route.

```sh
python infra/huggingface/package.py --output /absolute/external/explorer-bundle
docker build -t finserve-explorer /absolute/external/explorer-bundle
```

Use `--static` to create a product introduction and setup guide without a backend. See [local setup](https://github.com/chinmayarvind23/fin-serve/blob/master/docs/run-free.md) and the official [Docker Spaces guide](https://huggingface.co/docs/hub/en/spaces-sdks-docker).
