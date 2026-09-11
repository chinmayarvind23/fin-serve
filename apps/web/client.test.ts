/** DOM contract tests cover privacy and stale results; they do not substitute for visual review. */
import { afterAll, beforeAll, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";

let serial = 0;
let fetchMock: ReturnType<typeof spyOn<typeof globalThis, "fetch">>;

/** Bun adds preconnect to fetch; preserve that callable shape without opening a real connection. */
function fakeFetch(
  handler: (
    input: Parameters<typeof fetch>[0],
    init?: Parameters<typeof fetch>[1],
  ) => Promise<Response>,
): typeof fetch {
  return Object.assign(handler, { preconnect() {} });
}
const metric = {
  offered: 40,
  succeeded: 40,
  failed: 0,
  generatedTokens: 80,
  requestsPerSecond: 4,
  tokensPerSecond: 8,
  successRate: 1,
  seconds: 10,
  clientTtft: 0.02,
  serverTtft: 0.01,
  p95: 0.05,
};
const configuration = {
  engine: "fixture",
  engineConfig: "eager",
  model: "fixture",
  modelRevision: "model-a",
  tokenizerRevision: "tokenizer-a",
  sourceRevision: "source-a",
  imageDigest: "undeclared",
  hardware: "CPU",
  workloadHash: "fixed",
  concurrency: 1,
  warmup: 0,
  cachePolicy: "off",
  arrivalMode: "closed",
  arrivalRate: 10,
  timeoutSeconds: 30,
};
const catalog = [
  { id: "baseline-id", configuration, metrics: metric, slices: [], gpu: null, quality: null },
  {
    id: "candidate-id",
    configuration: { ...configuration, engineConfig: "compiled", modelRevision: "model-b" },
    metrics: metric,
    slices: [],
    gpu: null,
    quality: {
      accuracy: 0,
      parity: 1,
      caseCount: 1,
      passed: false,
      scope: "UI fixture only",
      referenceScope: "external baseline",
      hardFailures: ["incorrect"],
    },
  },
];

/** Tests need only document/event primitives, with no external scripts or network navigation. */
beforeAll(() => GlobalRegistrator.register({ url: "http://127.0.0.1:8051" }));
/** Restore all host APIs once these emulated DOM tests finish. */
afterAll(async () => {
  mock.restore();
  await GlobalRegistrator.unregister();
});

/** Each import gets a fresh client module and complete page structure, avoiding shared session state. */
beforeEach(async () => {
  mock.restore();
  const html = await Bun.file(new URL("./index.html", import.meta.url)).text();
  document.body.innerHTML = html.split("<body>")[1]?.split("</body>")[0] ?? "";
  fetchMock = spyOn(globalThis, "fetch").mockImplementation(
    fakeFetch(async (_input, init) => {
      const payload = JSON.parse(String(init?.body));
      if (payload.query.includes("runs(first")) return Response.json({ data: { runs: catalog } });
      if (payload.query.includes("decisions"))
        return Response.json({
          data: {
            decisions: [],
            lifecycle: [{ id: "job", status: "evaluated", version: 2, applyAttempts: 0 }],
            events: [],
          },
        });
      return Response.json({
        data: {
          run: {
            requests: [
              {
                id: 1,
                caseId: payload.variables.id,
                family: "GENERAL",
                phase: "measured",
                success: false,
                error: "fixture",
                generatedTokens: null,
                output: "<script>literal output</script>",
                outputTruncated: false,
                ttft: null,
                latency: 0.01,
              },
            ],
          },
        },
      });
    }),
  );
  await import(`./client.ts?test=${++serial}`);
});

/** Finite event-loop polling observes completed promise handlers without tying behavior to a delay. */
async function until(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100 && !predicate(); attempt++) await Bun.sleep(5);
  expect(predicate()).toBe(true);
}

/** A synthetic key is submitted only to the mocked same-origin evidence endpoint. */
async function connect(): Promise<void> {
  (document.getElementById("credential") as HTMLInputElement).value = "test-browser-credential";
  document.getElementById("connect")?.dispatchEvent(new Event("submit", { cancelable: true }));
  await until(() => Boolean(document.querySelector("#request-table table")));
}

test("quality rejection, comparison differences and literal model output remain visible", async () => {
  // Matching workload hashes cannot conceal changed models, and state versions are not route generations.
  await connect();
  expect(document.getElementById("provenance")?.textContent).toContain("modelRevision");
  expect(document.getElementById("quality-content")?.textContent).toContain("Quality gate failed");
  expect(document.querySelector("#request-table script")).toBeNull();
  expect(document.querySelector("#request-table pre")?.textContent).toBe(
    "<script>literal output</script>",
  );
  await until(() =>
    Boolean(document.getElementById("history-content")?.textContent?.includes("state version 2")),
  );
  expect((document.getElementById("credential") as HTMLInputElement).value).toBe("");
});

test("a changed candidate clears old records immediately and stale failures cannot win", async () => {
  // A delayed rejected response must not replace a newer candidate page or its status.
  await connect();
  let rejectOld: (reason: Error) => void = () => {
    throw new Error("request was not started");
  };
  fetchMock.mockImplementationOnce(
    fakeFetch(
      () =>
        new Promise<Response>((_resolve, reject) => {
          rejectOld = reject;
        }),
    ),
  );
  const select = document.getElementById("candidate") as HTMLSelectElement;
  select.value = "baseline-id";
  select.dispatchEvent(new Event("change"));
  expect(document.querySelector("#request-table table")).toBeNull();
  expect(document.getElementById("request-table")?.textContent).toContain("baseline-id");
  select.value = "candidate-id";
  select.dispatchEvent(new Event("change"));
  await until(() => Boolean(document.querySelector("#request-table table")));
  rejectOld(new Error("stale failure"));
  await Bun.sleep(10);
  expect(document.getElementById("status")?.textContent).not.toContain("stale failure");
  expect(document.getElementById("request-table")?.textContent).toContain("candidate-id");
});

test("replacement credentials clear private state even when the new key is rejected", async () => {
  // Authentication failure cannot leave the prior user's run labels or records in the page DOM.
  await connect();
  fetchMock.mockImplementationOnce(
    fakeFetch(async () => new Response("unauthorized", { status: 401 })),
  );
  (document.getElementById("credential") as HTMLInputElement).value = "rejected-browser-credential";
  document.getElementById("connect")?.dispatchEvent(new Event("submit", { cancelable: true }));
  expect((document.getElementById("workspace") as HTMLElement).hidden).toBe(true);
  expect(document.getElementById("candidate")?.textContent).toBe("");
  await until(() => document.getElementById("status")?.textContent === "Access key was rejected.");
  expect(document.getElementById("connection")?.textContent).toBe("Disconnected");
  for (const id of ["request-table", "baseline", "candidate", "provenance", "page"])
    expect(document.getElementById(id)?.textContent).toBe("");
});
