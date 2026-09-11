/** Live loopback tests exercise Bun's real request/response cancellation behavior. */
import { afterEach, expect, test } from "bun:test";
import { createProxy } from "./proxy";

const key = "development-edge-key-32-characters";
const upstreamKey = "development-upstream-key-32-characters";
const servers: Bun.Server<undefined>[] = [];

/** Dispose only ephemeral servers created by this test module. */
afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true);
});

/** Bind a random loopback port so tests cannot collide with user applications. */
function serve(handler: (request: Request) => Response | Promise<Response>): Bun.Server<undefined> {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, idleTimeout: 0, fetch: handler });
  servers.push(server);
  return server;
}

/** Requests always carry a separate edge key; the upstream service key is injected internally. */
function request(body: BodyInit = "{}", signal?: AbortSignal): Request {
  return new Request("http://edge/v1/completions", {
    method: "POST",
    body,
    headers: { Authorization: `Bearer ${key}` },
    ...(signal ? { signal } : {}),
  });
}

/** Bounded polling waits for real socket cancellation without assuming one scheduler ordering. */
async function until(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100 && !predicate(); attempt++) await Bun.sleep(10);
  expect(predicate()).toBe(true);
}

test("allowlist and authentication reject before upstream; safe headers and SSE bytes survive", async () => {
  let calls = 0;
  const frames = 'data: {"text":"one"}\n\ndata: [DONE]\n\n';
  const upstream = serve((incoming) => {
    // The caller's identity and forwarding headers are never trusted by the internal boundary.
    calls++;
    expect(incoming.headers.get("authorization")).toBe(`Bearer ${upstreamKey}`);
    expect(incoming.headers.get("x-forwarded-user")).toBeNull();
    return new Response(frames, {
      headers: { "Content-Type": "text/event-stream", "X-Secret": "private" },
    });
  });
  const proxy = createProxy({ upstream: upstream.url.toString(), apiKey: key, upstreamKey });
  expect((await proxy(new Request("http://edge/v1/completions", { method: "POST" }))).status).toBe(
    401,
  );
  expect(
    (await proxy(new Request("http://edge/admin", { headers: { Authorization: `Bearer ${key}` } })))
      .status,
  ).toBe(404);
  expect(calls).toBe(0);
  const response = await proxy(request());
  expect(await response.text()).toBe(frames);
  expect(response.headers.get("x-secret")).toBeNull();
});

test("oversized chunked upload is rejected before any upstream call", async () => {
  let calls = 0;
  const upstream = serve(() => {
    calls++;
    return new Response("unexpected");
  });
  const proxy = createProxy({ upstream: upstream.url.toString(), apiKey: key, upstreamKey });
  const body = new ReadableStream({
    start(controller) {
      // Actual received bytes, not Content-Length, are the upload authority.
      controller.enqueue(new Uint8Array(70000));
      controller.enqueue(new Uint8Array(70000));
      controller.close();
    },
  });
  expect((await proxy(request(body))).status).toBe(413);
  expect(calls).toBe(0);
});

test("a trickling upload is bounded by the overall deadline", async () => {
  let closed = false;
  const upstream = serve(() => new Response("unexpected"));
  const proxy = createProxy({
    upstream: upstream.url.toString(),
    apiKey: key,
    upstreamKey,
    timeoutMs: 20,
  });
  const body = new ReadableStream({
    cancel() {
      closed = true;
    },
  });
  expect((await proxy(request(body))).status).toBe(504);
  expect(closed).toBe(true);
});

test("upstream redirects cannot escape the configured origin", async () => {
  const upstream = serve(
    () => new Response(null, { status: 302, headers: { Location: "http://127.0.0.1:1/private" } }),
  );
  const proxy = createProxy({ upstream: upstream.url.toString(), apiKey: key, upstreamKey });
  expect((await proxy(request())).status).toBe(502);
});

test("response cancellation closes upstream and returns edge capacity", async () => {
  let closed = false;
  const upstream = serve((incoming) => {
    // A real upstream socket abort is stronger evidence than local stream cancellation alone.
    incoming.signal.addEventListener("abort", () => {
      closed = true;
    });
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("data: first\n\n"));
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );
  });
  const proxy = createProxy({
    upstream: upstream.url.toString(),
    apiKey: key,
    upstreamKey,
    maxActive: 1,
  });
  const response = await proxy(request());
  expect((await proxy(request())).status).toBe(429);
  const reader = response.body?.getReader();
  expect((await reader?.read())?.value).toBeDefined();
  await reader?.cancel();
  await until(() => closed);
  const next = await proxy(request());
  expect(next.status).toBe(200);
  await next.body?.cancel();
});

test("real downstream socket abort reaches the upstream request", async () => {
  let closed = false;
  const upstream = serve((incoming) => {
    incoming.signal.addEventListener("abort", () => {
      closed = true;
    });
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("data: first\n\n"));
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );
  });
  const edge = serve(createProxy({ upstream: upstream.url.toString(), apiKey: key, upstreamKey }));
  const controller = new AbortController();
  const response = await fetch(new URL("/v1/completions", edge.url), {
    method: "POST",
    body: "{}",
    headers: { Authorization: `Bearer ${key}` },
    signal: controller.signal,
  });
  await response.body?.getReader().read();
  controller.abort();
  await until(() => closed);
});

test("upstream deadline closes a stream that never completes", async () => {
  let closed = false;
  const upstream = serve((incoming) => {
    incoming.signal.addEventListener("abort", () => {
      closed = true;
    });
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("data: first\n\n"));
        },
      }),
    );
  });
  const proxy = createProxy({
    upstream: upstream.url.toString(),
    apiKey: key,
    upstreamKey,
    timeoutMs: 30,
  });
  const response = await proxy(request());
  await expect(response.text()).rejects.toThrow("UPSTREAM_STREAM_FAILED");
  await until(() => closed);
});
