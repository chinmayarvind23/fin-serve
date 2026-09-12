/** The browser receives an edge credential only; the registry credential stays in this process. */
import { createProxy } from "../api/proxy";
import home from "./index.html";

// Bun's production HTML manifest resolves emitted assets from the process directory.
// Anchor it to this entrypoint so launches from another directory find the same bundle.
process.chdir(import.meta.dir);

const proxy = createProxy({
  upstream: process.env.FINSERVE_EXPLORER_UPSTREAM ?? "http://127.0.0.1:8050",
  apiKey: process.env.FINSERVE_WEB_KEY ?? "",
  upstreamKey: process.env.FINSERVE_API_KEY ?? "",
  surface: "evidence",
  maxActive: 4,
  timeoutMs: 15000,
});

Bun.serve({
  hostname: process.env.FINSERVE_WEB_HOST ?? "127.0.0.1",
  port: Number(process.env.FINSERVE_WEB_PORT ?? "8051"),
  development: false,
  maxRequestBodySize: 16384,
  idleTimeout: 10,
  routes: { "/": home },
  /** Only the fixed GraphQL read endpoint reaches the configured evidence service. */
  fetch(request) {
    return proxy(request);
  },
  /** Unexpected exceptions never serialize internal endpoints or credentials. */
  error() {
    return Response.json({ error: { code: "EXPLORER_UNAVAILABLE" } }, { status: 500 });
  },
});
