/** CPU Space entrypoint reuses the actual explorer and its bounded authenticated proxy. */

import { createProxy } from "../../apps/api/proxy";
import explorer from "../../apps/web/index.html";

const proxy = createProxy({
  upstream: "http://127.0.0.1:8050",
  apiKey: process.env.FINSERVE_WEB_KEY ?? "",
  upstreamKey: process.env.FINSERVE_API_KEY ?? "",
  surface: "evidence",
  maxActive: 4,
  timeoutMs: 15000,
});
const publicFiles = new Map<string, string>([
  ["/public-assets/inference-demo.gif", "/app/public/inference-demo.gif"],
  ["/landing.css", "/app/landing.css"],
]);

/** Serve the fixed public introduction without asking the HTML bundler to import URL assets. */
function landing(): Response {
  return new Response(Bun.file("/app/landing.html"));
}

Bun.serve({
  hostname: "0.0.0.0",
  port: 7860,
  development: false,
  maxRequestBodySize: 16384,
  idleTimeout: 10,
  routes: { "/": landing, "/about": landing, "/explorer": explorer },
  /** Only fixed presentation assets are public; the existing proxy owns every data query. */
  fetch(request) {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/healthz") {
      return Response.json({ status: "alive", scope: "cpu-evidence-explorer" });
    }
    const path = publicFiles.get(url.pathname);
    if (request.method === "GET" && path && !url.search) {
      return new Response(Bun.file(path), {
        headers: { "X-Content-Type-Options": "nosniff", "Cache-Control": "public, max-age=3600" },
      });
    }
    return proxy(request);
  },
  /** Unexpected errors never expose process configuration or internal credentials. */
  error() {
    return Response.json({ error: { code: "EXPLORER_UNAVAILABLE" } }, { status: 500 });
  },
});
