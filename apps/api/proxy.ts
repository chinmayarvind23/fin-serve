/** Bounded HTTP edge: authentication and byte forwarding, with no second token scheduler. */
import { createHash, timingSafeEqual } from "node:crypto";

export interface ProxyConfig {
  upstream: string;
  apiKey: string;
  upstreamKey: string;
  timeoutMs?: number;
  maxActive?: number;
  surface?: "inference" | "evidence";
}

/** Fixed envelopes prevent upstream connection addresses and exception text leaking to callers. */
function failure(code: string, status: number): Response {
  return Response.json({ error: { code } }, { status });
}

/** Compare fixed-size digests so authentication does not expose prefix equality through timing. */
function authenticated(header: string | null, key: string): boolean {
  const digest = (value: string) => createHash("sha256").update(value).digest();
  return timingSafeEqual(digest(header ?? ""), digest(`Bearer ${key}`));
}

/** Only versioned inference/job endpoints are exposed; admin and caller-selected destinations stay private. */
function permitted(method: string, path: string, surface: "inference" | "evidence"): boolean {
  if (surface === "evidence") return method === "POST" && path === "/graphql";
  return (
    (method === "GET" && path === "/v1/visual/models") ||
    (method === "POST" &&
      [
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/visual/jobs",
        "/v1/vision/completions",
      ].includes(path)) ||
    (["GET", "DELETE"].includes(method) &&
      /^\/v1\/visual\/jobs\/[A-Za-z0-9_-]{1,128}$/.test(path)) ||
    (method === "GET" && /^\/v1\/visual\/jobs\/[A-Za-z0-9_-]{1,128}\/artifact$/.test(path))
  );
}

/** Upload size is checked while reading, and an overall deadline also closes a trickling body. */
async function boundedBody(
  request: Request,
  signal: AbortSignal,
  maximum: number,
): Promise<Uint8Array<ArrayBuffer> | undefined> {
  if (!request.body) return undefined;
  const reader = request.body.getReader();
  const cancel = () => void reader.cancel().catch(() => {});
  signal.addEventListener("abort", cancel, { once: true });
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      signal.throwIfAborted();
      const part = await reader.read();
      signal.throwIfAborted();
      if (part.done) break;
      size += part.value.byteLength;
      if (size > maximum) throw new RangeError("BODY_TOO_LARGE");
      chunks.push(part.value);
    }
    const result = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return result;
  } finally {
    signal.removeEventListener("abort", cancel);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

/** Each accepted request owns one capacity slot until its response drains or downstream disconnects. */
export function createProxy(config: ProxyConfig): (request: Request) => Promise<Response> {
  const upstream = new URL(config.upstream);
  const timeoutMs = config.timeoutMs ?? 300000;
  const maxActive = config.maxActive ?? 64;
  const surface = config.surface ?? "inference";
  if (
    !["http:", "https:"].includes(upstream.protocol) ||
    upstream.username ||
    upstream.password ||
    upstream.search ||
    upstream.hash ||
    upstream.pathname !== "/" ||
    config.apiKey.length < 16 ||
    config.upstreamKey.length < 16 ||
    !Number.isInteger(timeoutMs) ||
    timeoutMs < 1 ||
    timeoutMs > 300000 ||
    !Number.isInteger(maxActive) ||
    maxActive < 1 ||
    maxActive > 1024 ||
    !["inference", "evidence"].includes(surface)
  )
    throw new Error("Invalid edge configuration");
  let active = 0;

  /** Abort propagation stays alive after headers because model work can continue throughout SSE delivery. */
  return async (request: Request): Promise<Response> => {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET")
      return Response.json({ status: "ok", scope: "edge liveness" });
    if (!authenticated(request.headers.get("authorization"), config.apiKey))
      return failure("UNAUTHORIZED", 401);
    if (url.search || !permitted(request.method, url.pathname, surface))
      return failure("NOT_FOUND", 404);
    if (active >= maxActive) return failure("EDGE_OVERLOADED", 429);
    active++;
    const controller = new AbortController();
    let released = false;
    let timedOut = false;
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    let downstream: ReadableStreamDefaultController<Uint8Array> | undefined;
    const timer = setTimeout(() => {
      timedOut = true;
      downstream?.error(new Error("UPSTREAM_STREAM_FAILED"));
      release();
    }, timeoutMs);
    /** Cleanup is shared by stream completion, downstream abort and setup failure, so it must be idempotent. */
    const release = () => {
      if (released) return;
      released = true;
      clearTimeout(timer);
      request.signal.removeEventListener("abort", aborted);
      controller.abort();
      void reader?.cancel().catch(() => {});
      active--;
    };
    /** A closed client socket cancels the actual upstream fetch instead of only dropping local output. */
    const aborted = () => release();
    request.signal.addEventListener("abort", aborted, { once: true });
    if (request.signal.aborted) release();
    try {
      const body = await boundedBody(
        request,
        controller.signal,
        surface === "evidence"
          ? 16384
          : url.pathname === "/v1/vision/completions"
            ? 1450000
            : 131072,
      );
      const headers = new Headers({ Authorization: `Bearer ${config.upstreamKey}` });
      headers.set("Content-Type", "application/json");
      const idempotency = request.headers.get("idempotency-key");
      if (idempotency && /^[A-Za-z0-9_-]{1,128}$/.test(idempotency))
        headers.set("Idempotency-Key", idempotency);
      const response = await fetch(new URL(url.pathname, upstream), {
        method: request.method,
        headers,
        ...(body ? { body } : {}),
        signal: controller.signal,
        redirect: "manual",
      });
      if (response.status >= 300 && response.status < 400) {
        await response.body?.cancel();
        release();
        return failure("UPSTREAM_REDIRECT", 502);
      }
      const safeHeaders = new Headers();
      for (const name of ["content-type", "cache-control", "retry-after", "x-request-id"]) {
        const value = response.headers.get(name);
        if (value) safeHeaders.set(name, value);
      }
      if (!response.body) {
        release();
        return new Response(null, { status: response.status, headers: safeHeaders });
      }
      reader = response.body.getReader();
      let size = 0;
      const stream = new ReadableStream<Uint8Array>(
        {
          /** Retain only the controller so a deadline can close even an unread response. */
          start(output) {
            downstream = output;
          },
          /** One upstream read per downstream pull preserves bytes and bounds read-ahead. */
          async pull(output) {
            try {
              const chunk = await reader?.read();
              controller.signal.throwIfAborted();
              if (!chunk || chunk.done) {
                output.close();
                release();
                return;
              }
              size += chunk.value.byteLength;
              if (size > (surface === "evidence" ? 2 : 8) * 1024 * 1024)
                throw new Error("UPSTREAM_TOO_LARGE");
              output.enqueue(chunk.value);
            } catch {
              output.error(new Error("UPSTREAM_STREAM_FAILED"));
              release();
            }
          },
          /** Cancellation may happen before the first pull and still owns upstream/capacity cleanup. */
          cancel() {
            release();
          },
        },
        { highWaterMark: 0 },
      );
      return new Response(stream, { status: response.status, headers: safeHeaders });
    } catch (error) {
      release();
      if (error instanceof RangeError) return failure("BODY_TOO_LARGE", 413);
      return failure(
        timedOut ? "EDGE_DEADLINE_EXCEEDED" : "UPSTREAM_UNAVAILABLE",
        timedOut ? 504 : 502,
      );
    }
  };
}
