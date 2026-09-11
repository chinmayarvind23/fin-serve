/** Independent edge regressions cover consumers that stop pulling before a deadline. */
import { afterEach, expect, test } from "bun:test";
import { createProxy } from "./proxy";

const key = "review-edge-credential-length-32";
const upstreamKey = "review-upstream-credential-length-32";
const servers: Bun.Server<undefined>[] = [];

/** Each test owns only its ephemeral loopback servers. */
afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true);
});

/** Use an actual HTTP origin without touching running application services. */
function origin(handler: (request: Request) => Response): Bun.Server<undefined> {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, idleTimeout: 0, fetch: handler });
  servers.push(server);
  return server;
}

/** Keep test credentials and request destinations deterministic and non-sensitive. */
function request(): Request {
  return new Request("http://edge/v1/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}` },
    body: "{}",
  });
}

test("deadline returns capacity even when response is never pulled", async () => {
  const upstream = origin(() => new Response("data: token\n\n"));
  const proxy = createProxy({
    upstream: upstream.url.toString(),
    apiKey: key,
    upstreamKey,
    maxActive: 1,
    timeoutMs: 30,
  });
  const unconsumed = await proxy(request());
  expect(unconsumed.status).toBe(200);
  expect((await proxy(request())).status).toBe(429);
  await Bun.sleep(75);
  const afterDeadline = await proxy(request());
  try {
    expect(afterDeadline.status).toBe(200);
  } finally {
    await unconsumed.body?.cancel().catch(() => {});
    await afterDeadline.body?.cancel().catch(() => {});
  }
});

test("cancel before first response pull releases capacity", async () => {
  const upstream = origin(() => new Response("unread"));
  const proxy = createProxy({
    upstream: upstream.url.toString(),
    apiKey: key,
    upstreamKey,
    maxActive: 1,
  });
  const first = await proxy(request());
  await first.body?.cancel();
  const second = await proxy(request());
  expect(second.status).toBe(200);
  expect(await second.text()).toBe("unread");
});

test("oversized upstream bytes fail the stream and release capacity", async () => {
  let calls = 0;
  const upstream = origin(() => {
    calls++;
    return new Response(calls === 1 ? new Uint8Array(8 * 1024 * 1024 + 1) : "small");
  });
  const proxy = createProxy({
    upstream: upstream.url.toString(),
    apiKey: key,
    upstreamKey,
    maxActive: 1,
  });
  const first = await proxy(request());
  await expect(first.arrayBuffer()).rejects.toThrow("UPSTREAM_STREAM_FAILED");
  const second = await proxy(request());
  expect(second.status).toBe(200);
  expect(await second.text()).toBe("small");
});
