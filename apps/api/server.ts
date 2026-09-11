/** Separate Bun edge process: internal upstream credentials never enter browser bundles. */
import { createProxy } from "./proxy";

const handler = createProxy({
  upstream: process.env.FINSERVE_UPSTREAM ?? "http://127.0.0.1:8000",
  apiKey: process.env.FINSERVE_EDGE_KEY ?? "",
  upstreamKey: process.env.FINSERVE_API_KEY ?? "",
});

Bun.serve({
  hostname: process.env.FINSERVE_EDGE_HOST ?? "127.0.0.1",
  port: Number(process.env.FINSERVE_EDGE_PORT ?? "8040"),
  maxRequestBodySize: 1450000,
  idleTimeout: 10,
  /** A request's explicit overall timer owns streaming lifetime after bounded HTTP header parsing. */
  fetch(request, server) {
    server.timeout(request, 0);
    return handler(request);
  },
  /** Expected request errors use typed envelopes; unexpected defects never expose stack traces. */
  error() {
    return Response.json({ error: { code: "EDGE_FAILURE" } }, { status: 500 });
  },
});
