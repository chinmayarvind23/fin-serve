/** A read-only client renders registered evidence as text and keeps its edge key in memory. */
export {};
type Metrics = {
  offered: number;
  succeeded: number;
  failed: number;
  generatedTokens: number;
  requestsPerSecond: number;
  tokensPerSecond: number;
  successRate: number;
  seconds: number;
  clientTtft: number | null;
  serverTtft: number | null;
  p95: number | null;
};
type Quality = {
  accuracy: number;
  parity: number;
  caseCount: number;
  passed: boolean;
  scope: string;
  referenceScope: string;
  hardFailures: string[];
};
type Run = {
  id: string;
  configuration: {
    engine: string;
    engineConfig: string;
    model: string;
    modelRevision: string;
    tokenizerRevision: string;
    sourceRevision: string;
    imageDigest: string;
    hardware: string;
    workloadHash: string;
    concurrency: number;
    warmup: number;
    cachePolicy: string;
    arrivalMode: string;
    arrivalRate: number;
    timeoutSeconds: number;
  };
  metrics: Metrics;
  slices: { name: string; metrics: Metrics }[];
  gpu: {
    averageUtilization: number | null;
    coverage: number;
    deviceIds: string[];
    method: string;
  } | null;
  quality: Quality | null;
};
type RequestRow = {
  id: number;
  caseId: string;
  family: string;
  phase: string;
  success: boolean;
  error: string | null;
  generatedTokens: number | null;
  output: string;
  outputTruncated: boolean;
  ttft: number | null;
  latency: number | null;
};
type History = {
  decisions: { id: string; runId: string; approved: boolean; reasons: string[] }[];
  lifecycle: { id: string; status: string; version: number; applyAttempts: number }[];
  events: { id: string; jobId: string; observedAt: number; status: string; version: number }[];
};

const metricsFields =
  "offered succeeded failed generatedTokens requestsPerSecond tokensPerSecond successRate seconds clientTtft serverTtft p95";
const catalogQuery = `{ runs(first: 20) {
  id configuration { engine engineConfig model modelRevision tokenizerRevision sourceRevision imageDigest hardware workloadHash concurrency warmup cachePolicy arrivalMode arrivalRate timeoutSeconds }
  metrics { ${metricsFields} } slices { name metrics { ${metricsFields} } }
  gpu { averageUtilization coverage deviceIds method }
  quality { accuracy parity caseCount passed scope referenceScope hardFailures }
} }`;
let credential = "";
let runs: Run[] = [];
let offset = 0;
let generation = 0;
let requestGeneration = 0;
let active: AbortController | undefined;

/** Missing document anchors are programming errors rather than silently empty UI regions. */
function element<T extends HTMLElement = HTMLElement>(id: string): T {
  const result = document.getElementById(id);
  if (!result) throw new Error(`Missing view: ${id}`);
  return result as T;
}

/** All evidence content uses textContent; generated HTML, script tags and links remain inert. */
function node<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text = "",
  className = "",
): HTMLElementTagNameMap[K] {
  const result = document.createElement(tag);
  result.textContent = text;
  result.className = className;
  return result;
}

/** Unknown/nonfinite measurements are shown explicitly and never coerced into successful zeros. */
function number(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      })
    : "Unknown";
}

/** Seconds remain authoritative in the API; milliseconds are a presentation conversion only. */
function milliseconds(value: number | null | undefined): string {
  return value == null ? "Unknown" : `${number(value * 1000, 1)} ms`;
}

/** A single same-origin endpoint prevents browser input from selecting internal registry destinations. */
async function query<T>(text: string, variables: Record<string, unknown> = {}): Promise<T> {
  const version = generation;
  const response = await fetch("/graphql", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${credential}` },
    body: JSON.stringify({ query: text, variables }),
    signal: AbortSignal.any([AbortSignal.timeout(15000), ...(active ? [active.signal] : [])]),
    cache: "no-store",
    credentials: "omit",
    redirect: "error",
  });
  if (response.status === 401 && version === generation) resetSession("Access key was rejected.");
  if (!response.ok)
    throw new Error(
      response.status === 401
        ? "Access key was rejected."
        : `Evidence request failed (${response.status}).`,
    );
  const payload = await response.json();
  if (payload.errors || !payload.data)
    throw new Error("The registry could not verify this evidence query.");
  return payload.data as T;
}

/** Configuration labels distinguish eager, compiled and other runs without inventing experiment names. */
function label(run: Run): string {
  return `${run.configuration.engineConfig} · ${run.id.slice(0, 8)}`;
}

/** Selection values refer only to runs returned by the authenticated catalog query. */
function selected(id: string): Run | undefined {
  return runs.find((run) => run.id === element<HTMLSelectElement>(id).value);
}

/** Missing slice metrics remain absent instead of silently falling back to the aggregate. */
function metrics(run: Run): Metrics | undefined {
  const slice = element<HTMLSelectElement>("slice").value;
  return slice === "all" ? run.metrics : run.slices.find((item) => item.name === slice)?.metrics;
}

/** Cards report candidate values alongside the directly corresponding baseline population. */
function card(title: string, value: string, detail: string): HTMLElement {
  const result = node("article", "", "metric");
  result.append(
    node("div", title, "metric-label"),
    node("div", value, "metric-value"),
    node("div", detail, "metric-detail"),
  );
  return result;
}

/** Native meter elements expose values accessibly while preserving one scale for each comparison. */
function bars(
  title: string,
  left: number | null | undefined,
  right: number | null | undefined,
  unit: string,
): HTMLElement {
  const group = node("div", "", "bar-group");
  group.append(node("div", title, "bar-label"));
  const maximum = Math.max(left ?? 0, right ?? 0, 0.00001);
  for (const [name, value] of [
    ["Baseline", left],
    ["Candidate", right],
  ] as const) {
    const row = node("div", "", "bar-row");
    const meter = node("meter", "", name.toLowerCase());
    meter.min = 0;
    meter.max = maximum;
    meter.value = value ?? 0;
    meter.setAttribute("aria-label", `${title}, ${name}: ${number(value)} ${unit}`);
    row.append(node("span", name), meter, node("span", `${number(value)} ${unit}`));
    group.append(row);
  }
  return group;
}

/** Identity details show declared native image gaps and the exact model/source hashes. */
function configuration(left: Run, right: Run): void {
  const list = node("dl");
  const pair = (a: string | number, b: string | number) =>
    a === b ? String(a) : `Baseline: ${a}\nCandidate: ${b}`;
  // A closed-loop driver's configured rate is unused and must not look like an offered load claim.
  const offeredRate = (run: Run) =>
    run.configuration.arrivalMode === "open"
      ? `${number(run.configuration.arrivalRate)} req/s`
      : "Not applicable (closed loop)";
  const rows: [string, string][] = [
    ["Engine", pair(left.configuration.engine, right.configuration.engine)],
    ["Engine options", pair(left.configuration.engineConfig, right.configuration.engineConfig)],
    ["Model", pair(left.configuration.model, right.configuration.model)],
    ["Model revision", pair(left.configuration.modelRevision, right.configuration.modelRevision)],
    [
      "Tokenizer",
      pair(left.configuration.tokenizerRevision, right.configuration.tokenizerRevision),
    ],
    [
      "Source revision",
      pair(left.configuration.sourceRevision, right.configuration.sourceRevision),
    ],
    ["Image", pair(left.configuration.imageDigest, right.configuration.imageDigest)],
    ["Hardware", pair(left.configuration.hardware, right.configuration.hardware)],
    ["Concurrency", `${left.configuration.concurrency} → ${right.configuration.concurrency}`],
    ["Cache policy", pair(left.configuration.cachePolicy, right.configuration.cachePolicy)],
    ["Workload SHA", pair(left.configuration.workloadHash, right.configuration.workloadHash)],
    ["Measured count", pair(left.metrics.offered, right.metrics.offered)],
    ["Warmup count", pair(left.configuration.warmup, right.configuration.warmup)],
    ["Arrival mode", pair(left.configuration.arrivalMode, right.configuration.arrivalMode)],
    ["Offered rate", pair(offeredRate(left), offeredRate(right))],
    ["Timeout (s)", pair(left.configuration.timeoutSeconds, right.configuration.timeoutSeconds)],
    [
      "GPU utilization",
      `${number(left.gpu?.averageUtilization)}% → ${number(right.gpu?.averageUtilization)}%`,
    ],
    [
      "GPU coverage",
      right.gpu ? `${number(right.gpu.coverage * 100)}% · ${right.gpu.method}` : "Not recorded",
    ],
  ];
  for (const [name, value] of rows) list.append(node("dt", name), node("dd", value));
  element("configuration").replaceChildren(list);
}

/** A quality rejection remains prominent even when the associated throughput improves. */
function qualityPanel(run: Run, title: string): HTMLElement {
  const panel = node("article", "", "panel quality-panel");
  panel.append(node("h3", `${title} · ${run.id.slice(0, 8)}`));
  const quality = run.quality;
  if (!quality) {
    panel.append(node("p", "No registered correctness evidence.", "muted"));
    return panel;
  }
  panel.append(
    node(
      "span",
      quality.passed ? "Quality gate passed" : "Quality gate failed",
      `pill ${quality.passed ? "connected" : "danger"}`,
    ),
  );
  panel.append(
    node("div", `${number(quality.accuracy * 100)}%`, "quality-score"),
    node(
      "p",
      `Correctness across ${quality.caseCount} cases · ${number(quality.parity * 100)}% output parity (${quality.referenceScope}).`,
    ),
  );
  panel.append(node("p", quality.scope, "muted"));
  if (quality.hardFailures.length) {
    const details = node("details");
    details.append(node("summary", `${quality.hardFailures.length} retained failure reasons`));
    const list = node("ul");
    for (const reason of quality.hardFailures) list.append(node("li", reason));
    details.append(list);
    panel.append(details);
  }
  return panel;
}

/** Changing either run recomputes visible comparisons without mutating evidence or issuing a gate. */
function renderComparison(): void {
  const left = selected("baseline"),
    right = selected("candidate");
  if (!left || !right) return;
  const baseline = metrics(left),
    candidate = metrics(right);
  const fields = [
    "model",
    "arrivalMode",
    "arrivalRate",
    "timeoutSeconds",
    "workloadHash",
    "modelRevision",
    "tokenizerRevision",
    "hardware",
    "concurrency",
    "cachePolicy",
    "warmup",
  ] as const;
  const differences: string[] = fields.filter(
    (key) => left.configuration[key] !== right.configuration[key],
  );
  if (left.metrics.offered !== right.metrics.offered) differences.push("measured request count");
  const compatible = differences.length === 0;
  element("provenance").textContent =
    `${compatible ? "Matched workload, model, hardware and load envelope." : `Comparison differences: ${differences.join(", ")}. These values do not establish an optimization gain.`} ${left.configuration.imageDigest === "undeclared" || right.configuration.imageDigest === "undeclared" ? "Native execution; deployment image identity is undeclared." : "Inspect both image identities before promotion."} Performance bars describe observations, not release approval.`;
  element("metrics").replaceChildren(
    card(
      "Successful requests / second",
      number(candidate?.requestsPerSecond),
      `Baseline ${number(baseline?.requestsPerSecond)} req/s`,
    ),
    card(
      "Generated tokens / second",
      number(candidate?.tokensPerSecond),
      `Baseline ${number(baseline?.tokensPerSecond)} tokens/s`,
    ),
    card(
      "Median server TTFT",
      milliseconds(candidate?.serverTtft),
      `Baseline ${milliseconds(baseline?.serverTtft)}`,
    ),
    card("End-to-end p95", milliseconds(candidate?.p95), `Baseline ${milliseconds(baseline?.p95)}`),
  );
  element("bars").replaceChildren(
    bars(
      "Successful requests / second",
      baseline?.requestsPerSecond,
      candidate?.requestsPerSecond,
      "req/s",
    ),
    bars(
      "Generated tokens / second",
      baseline?.tokensPerSecond,
      candidate?.tokensPerSecond,
      "tok/s",
    ),
    bars("End-to-end p95", baseline?.p95, candidate?.p95, "s"),
    node(
      "p",
      `Baseline: ${number(baseline?.offered, 0)} offered, ${number(baseline?.failed, 0)} failed. Candidate: ${number(candidate?.offered, 0)} offered, ${number(candidate?.failed, 0)} failed. Warmup excluded from these metrics.`,
      "muted",
    ),
  );
  configuration(left, right);
  element("quality-content").replaceChildren(
    qualityPanel(left, "Baseline"),
    qualityPanel(right, "Candidate"),
  );
}

/** Request pages include failures and warmup, with stale asynchronous pages prevented from winning. */
async function renderRequests(): Promise<void> {
  const run = selected("candidate");
  if (!run) return;
  const version = ++requestGeneration;
  const session = generation;
  element("request-table").replaceChildren(
    node("p", `Loading run ${run.id}, records from ${offset + 1}…`, "muted"),
  );
  element("page").textContent = "Loading";
  element<HTMLButtonElement>("previous").disabled = true;
  element<HTMLButtonElement>("next").disabled = true;
  const data = await query<{ run: { requests: RequestRow[] } }>(
    `query($id: ID!, $offset: Int!) { run(id: $id) { requests(first: 20, offset: $offset) { id caseId family phase success error generatedTokens output outputTruncated ttft latency } } }`,
    { id: run.id, offset },
  );
  if (version !== requestGeneration || session !== generation || !credential) return;
  const table = node("table"),
    head = node("thead"),
    titles = node("tr"),
    body = node("tbody");
  for (const title of ["Request / case", "Phase", "Result", "Tokens", "TTFT", "Latency", "Output"])
    titles.append(node("th", title));
  head.append(titles);
  for (const record of data.run.requests) {
    const row = node("tr");
    for (const value of [
      `${record.id} · ${record.caseId}`,
      record.phase,
      record.success ? "Success" : `Failed: ${record.error ?? "unknown"}`,
      number(record.generatedTokens, 0),
      milliseconds(record.ttft),
      milliseconds(record.latency),
    ])
      row.append(node("td", value));
    const output = node("td");
    output.append(
      node("pre", record.output + (record.outputTruncated ? "\n[Output truncated by API]" : "")),
    );
    row.append(output);
    body.append(row);
  }
  table.append(head, body);
  element("request-table").replaceChildren(table);
  element("page").textContent = `${offset + 1}–${offset + data.run.requests.length}`;
  element<HTMLButtonElement>("previous").disabled = offset === 0;
  element<HTMLButtonElement>("next").disabled =
    data.run.requests.length < 20 ||
    offset + 20 >= run.metrics.offered + run.configuration.warmup ||
    offset >= 10000;
}

/** History is loaded separately so its query budget is independent of detailed run comparisons. */
async function renderHistory(version: number): Promise<void> {
  const data = await query<History>(
    "{ decisions(first: 20) { id runId approved reasons } lifecycle(first: 20) { id status version applyAttempts } events(first: 20) { id jobId observedAt status version } }",
  );
  if (version !== generation || !credential) return;
  const view = element("history-content");
  view.replaceChildren();
  if (!data.decisions.length && !data.lifecycle.length && !data.events.length)
    view.append(
      node(
        "p",
        "No release decisions or lifecycle events are registered. Measurements alone do not establish a deployment.",
        "muted",
      ),
    );
  for (const decision of data.decisions)
    view.append(
      node(
        "p",
        `${decision.approved ? "Approved" : "Rejected"} · ${decision.runId} · ${decision.reasons.join(", ")}`,
      ),
    );
  for (const job of data.lifecycle)
    view.append(
      node(
        "p",
        `${job.id} · ${job.status} · state version ${job.version} · ${job.applyAttempts} apply attempts`,
      ),
    );
  for (const event of data.events)
    view.append(
      node(
        "p",
        `${new Date(event.observedAt * 1000).toLocaleString()} · ${event.jobId} · ${event.status} · state version ${event.version}`,
        "muted",
      ),
    );
}

/** Refresh replaces catalog state atomically and aborts superseded network requests. */
async function refresh(): Promise<void> {
  const version = ++generation;
  active?.abort();
  active = new AbortController();
  element("status").textContent = "Reading checksum-verified evidence…";
  const data = await query<{ runs: Run[] }>(catalogQuery);
  if (version !== generation) return;
  runs = data.runs;
  for (const id of ["baseline", "candidate"]) {
    const select = element<HTMLSelectElement>(id),
      previous = select.value;
    select.replaceChildren(
      ...runs.map((run) => {
        const option = node("option", label(run));
        option.value = run.id;
        return option;
      }),
    );
    select.value = runs.some((run) => run.id === previous)
      ? previous
      : ((id === "baseline"
          ? runs.find((run) => run.configuration.engineConfig.startsWith("eager"))
          : runs.find((run) => run.configuration.engineConfig.startsWith("compiled"))
        )?.id ??
        runs[id === "baseline" ? 0 : Math.min(1, runs.length - 1)]?.id ??
        "");
  }
  const slice = element<HTMLSelectElement>("slice");
  slice.replaceChildren(node("option", "All measured requests"));
  if (slice.options[0]) slice.options[0].value = "all";
  for (const name of new Set(runs.flatMap((run) => run.slices.map((item) => item.name)))) {
    const option = node("option", name);
    option.value = name;
    slice.append(option);
  }
  element("workspace").hidden = !runs.length;
  element("connection").textContent = "Registry connected";
  element("connection").className = "pill connected";
  element("status").textContent = runs.length
    ? `${runs.length} registered runs loaded. Showing at most 20 runs per catalog page.`
    : "No runs have been registered.";
  offset = 0;
  renderComparison();
  await Promise.all([renderRequests(), renderHistory(version)]);
}

/** UI handlers report errors without exposing response bodies, endpoints or internal stack traces. */
function report(task: Promise<void>, version = generation, pageVersion?: number): void {
  void task.catch((error: unknown) => {
    if (
      credential &&
      version === generation &&
      (pageVersion === undefined || pageVersion === requestGeneration)
    ) {
      element("status").textContent =
        error instanceof Error ? error.message : "Evidence request failed.";
      if (pageVersion !== undefined) {
        element("request-table").replaceChildren(
          node(
            "p",
            "This request page could not be loaded. Refresh or choose a run to retry.",
            "muted",
          ),
        );
        element("page").textContent = "Unavailable";
      }
    }
  });
}

/** Session replacement clears every private view and fences requests from the previous credential. */
function resetSession(message: string): void {
  credential = "";
  generation++;
  requestGeneration++;
  active?.abort();
  runs = [];
  element<HTMLInputElement>("credential").value = "";
  element("workspace").hidden = true;
  element("connection").textContent = "Disconnected";
  element("connection").className = "pill";
  element("status").textContent = message;
  for (const id of [
    "baseline",
    "candidate",
    "slice",
    "provenance",
    "page",
    "metrics",
    "bars",
    "configuration",
    "quality-content",
    "request-table",
    "history-content",
  ])
    element(id).replaceChildren();
}

element<HTMLFormElement>("connect").addEventListener("submit", (event) => {
  // Clear the password field immediately so the access key does not remain in DOM form state.
  event.preventDefault();
  const key = element<HTMLInputElement>("credential").value;
  resetSession("Connecting to registry…");
  credential = key;
  report(refresh());
});
element("disconnect").addEventListener("click", () => {
  // Disconnect fences in-flight responses and removes retained output as well as credentials.
  resetSession("Disconnected. Access key cleared.");
});
element("refresh").addEventListener("click", () => report(refresh()));
for (const id of ["baseline", "candidate", "slice"])
  element(id).addEventListener("change", () => {
    // A candidate change resets paging; slice changes leave raw request order untouched.
    renderComparison();
    if (id === "candidate") {
      offset = 0;
      report(renderRequests(), generation, requestGeneration);
    }
  });
element("previous").addEventListener("click", () => {
  offset = Math.max(0, offset - 20);
  report(renderRequests(), generation, requestGeneration);
});
element("next").addEventListener("click", () => {
  offset = Math.min(10000, offset + 20);
  report(renderRequests(), generation, requestGeneration);
});
