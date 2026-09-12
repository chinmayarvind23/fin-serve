# Recorded FinServe walkthrough

The [49.48-second walkthrough](../../resources/fin_serve/evidence/final-walkthrough-01/finserve-walkthrough.mp4) combines the evidence explorer, CPU JAX/gRPC job, warm rollback, and actual pretrained text/image requests. The private MP4 preserves the original playback speed and each component's scope. Full decode passed; SHA256 is `94a3d610e8384b7d3e1b4c159412e063039e7091c33def184ff92f024dfb2ee9`.

The evidence explorer now has actual Chromium recordings at 1440×1000 and 390×1000. The short, silent clips show the retained eager/compiled comparison, workload-slice selection, failed quality gate and expanded failure reasons. Browser checks passed for navigation, horizontal overflow and page errors; all four screenshots and a decoded video frame were visually reviewed. Authentication used a masked, ephemeral key and both owned CPU services stopped after capture.

Private review artifacts are outside the repository at `resources/fin_serve/evidence/explorer-demo-02`: `walkthrough-1440.webm` (8.48 seconds), `walkthrough-390.webm` (7.84 seconds), four comparison/quality PNGs, and `recording-manifest.json`. The manifest binds the exact source snapshot, SQLite backup, run IDs and capture SHA256 hashes. Source HEAD was `27bd10ccf4b48a4e3a9d35d6d9a42c08cf95f368`; per-file hashes identify the captured worktree. The earlier namespace-check failure remains in `explorer-demo-01`.

The capture shows historical runs `914c1f2b-6dde-4d99-850c-ca099b6841de` and `a4c55327-0e45-4506-9987-3b234327dcff`, each with 3,072 measured requests. It retains the faster candidate's failed quality qualification. These clips demonstrate reading existing evidence, not a new inference run or the latest correctness candidate. The original videos remain private. Use the [local setup guide](run-free.md) to run the application.

![Actual explorer comparison and failed quality detail](assets/explorer-demo.gif)

The [repository GIF](assets/explorer-demo.gif) samples the reviewed desktop recording at 2.5 frames/second, retaining its full 1440×1000 dimensions. Its 22 frames play for 8.8 seconds; palette and frame-difference compression reduce it to 2.88 MB. Decoded comparison and quality frames were reviewed for readable metrics, explicit failure status and absence of secrets. No frames were invented. GIF SHA256: `ecd332db926446956415230607ef5f1927a6f701bd21d2c9c3078992792bfdc4`.

Workspace-only links, outside a clean repository clone: [desktop video](../../resources/fin_serve/evidence/explorer-demo-02/walkthrough-1440.webm), [mobile video](../../resources/fin_serve/evidence/explorer-demo-02/walkthrough-390.webm), [recording manifest](../../resources/fin_serve/evidence/explorer-demo-02/recording-manifest.json), and [GIF conversion manifest](../../resources/fin_serve/evidence/explorer-demo-02/gif-manifest.json). The full private GIF master is retained beside these files. The GIF has not been added to the Hugging Face publication bundle.

Two additional private recordings now cover durable visual execution and warm-route recovery:

| Capture | Verified behavior | Scope |
| --- | --- | --- |
| [CPU visual job](../../resources/fin_serve/evidence/visual-cpu-demo-04/actual-cpu-job.webm), 10.28 seconds | Actual HTTP submission, queued/running/succeeded states, real gRPC JAX worker, SHA256-verified PNG, idempotent replay and durable SQLite state after shutdown | Private visual-API host using the unchanged coordinator/routes; CPU-only untrained reference, outside the full Bun application |
| [Warm rollback](../../resources/fin_serve/evidence/rollback-demo-03/actual-rollback.webm), 17.40 seconds | Actual HTTP requests, retained deliberate failure, candidate-to-baseline recovery, route generations 0/1/2 and verified healthy output | Real control/HTTP execution with synthetic backend responses and promotion identities; no GPU or cloud rollout |

The visual capture binds source `5f217fa` and job `2ef71650-4a69-4ab5-8f9f-6698071231c7`. Its worker finished with zero active or retained tasks. The rollback capture executes the unchanged integration drill from an archived `12e4906` source: five of six offered requests completed, with the deliberate failed request retained. Detection-to-healthy time was 0.265 seconds for this local fixture run. That observation does not replace the historical warm drill or measure model startup. The owned test process exited successfully and the private viewer stopped.

Both capture directories include manifests, original observations and reviewed screenshots. The rollback WebM decoded without errors. Earlier unsuccessful capture attempts and the second rollback capture's missing phase labels remain retained. Neither recording contains credentials, and neither has been uploaded to the static Space.

The [36.16-second component walkthrough](../../resources/fin_serve/evidence/component-walkthrough-01/finserve-components.mp4) joins the original desktop explorer, CPU visual-job and rollback clips in that order. It preserves their playback speed and visible scope statements. The adjacent manifest binds all three originals and the encoded MP4 by SHA256; the complete MP4 decoded successfully. These component recordings use their separately documented source versions and do not yet include pretrained text or vision inference.

The [13.32-second pretrained API capture](../../resources/fin_serve/evidence/text-vision-edge-demo-03/actual-vision-edge.webm) adds one text request and three image requests through production Bun and FastAPI to the pinned Qwen2-VL-2B model. All four returned HTTP 200 and complete SSE; the retained chart probes returned red, green and blue correctly. This is reused diagnostic evidence, not a new quality benchmark. The separate uniform-color suite remains failed. Source `66b5501`, exact API process identity, raw responses, image hashes, screenshots and cleanup receipts accompany the clip.

The recorder uses a real API client and displays completed SSE responses in the browser; it does not demonstrate a live browser streaming interface. The original browser-fetch attempt failed before sending requests and remains retained. Both CPU services stopped, and the owned GPU process group was verified empty. The 2B vision model used for this recording is distinct from the 3B/7B correctness candidates.

Keep credentials masked and private prompts, environment values and unrelated desktop content out of recordings. Publish only reviewed evidence actually captured.

The [results page](results.md), [architecture](HLD.md) and [commands](commands.md) remain usable when a GPU service is offline. No simulated cloud deployment or reconstructed benchmark screen substitutes for the live checks.
