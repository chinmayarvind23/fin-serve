---
title: FinServe Evidence Explorer
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

This CPU Docker Space serves FinServe's existing Bun evidence explorer and Python
GraphQL read service. It performs no GPU inference. The public landing page shows
committed aggregate figures and their measurement limitations; `/explorer` opens
the authenticated application.

If your account cannot host a Docker Space without a subscription, run this
container locally and publish the static aggregate page instead:
`python infra/huggingface/package.py --static --output /absolute/private/static-space`.
The static bundle has no backend or authentication secrets. See the
[free local setup](https://github.com/chinmayarvind23/fin-serve/blob/master/docs/run-free.md).

The initial SQLite registry is empty. No private raw request records, model
weights or credentials are included. Data under `/data` is ephemeral and is not a
durable hosted registry. A successful container start does not establish a cloud
inference deployment, semantic quality approval or any new performance result.

Set two different Space **secrets**, each 16–4,096 printable ASCII characters:

- `FINSERVE_API_KEY`: internal Python credential, never entered in the browser.
- `FINSERVE_WEB_KEY`: explorer access key supplied separately to authorized viewers.

The container fails closed if either secret is missing or they are equal. Secrets
are read only at runtime from environment variables; they are not build arguments,
command-line arguments or bundled JavaScript. The public page and aggregate figures
need no key. POST `/graphql` remains authenticated, read-only and bounded to a
16 KiB request, 2 MiB response and four active edge requests. No inference or admin
route is forwarded.

The container runs as UID1000. Bun listens on7860; Python listens only on loopback8050.
The launcher checks an authenticated registry read before starting Bun and stops
both owned processes on termination or unexpected child exit. Normal readiness
does not attest imported evidence; every registry artifact still undergoes the
existing content and checksum validation.

From the main repository, prepare a fresh external allowlisted bundle:

```sh
uv run --no-sync python infra/huggingface/package.py --output /absolute/private/space-bundle
docker build -t finserve-evidence-space /absolute/private/space-bundle
```

Only after local checks and a successful `hf auth whoami`, upload that exact bundle
to the authorized Space using `hf upload SPACE_ID /absolute/private/space-bundle --type space`.
The package manifest records every included file hash and actual repository status.
Uploading private source data requires a separate explicit decision; the default
bundle contains public source and six committed aggregate files only.

Follow the official [Docker Spaces setup](https://huggingface.co/docs/hub/en/spaces-sdks-docker)
for runtime secrets, port mapping and ephemeral storage. See the
[source project](https://github.com/chinmayarvind23/fin-serve) and
[measurement methods](https://github.com/chinmayarvind23/fin-serve/blob/master/docs/performance.md)
for the local GPU evidence represented by the public figures.
