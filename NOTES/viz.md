# Visualization and render performance

Area: `src/viz/**`, `src/ui/**`, `src/main.ts`, `src/eval/fps.ts`, `tests/e2e/**`, `tests/viz/**`, `tests/ui/**`.
Measured with `npm run eval -- --only=fps` (M3: 64 ghost cars, the training worker running).

## What the numbers say

The test box has no GPU. Chromium 141 headless_shell renders with SwiftShader (software Vulkan), on the
same 4 cores as the learner and everything else that was running (load average 4 to 6 during the runs).
Wall-clock fps there says little about a real machine, so every fps row is reported with three more things:

- CPU-ms per frame summed over the whole Chromium process tree (from `/proc`). It moves much less with load than fps.
- JS time per frame (update, render submit, UI), measured around the rAF callback. Independent of the GPU.
- `renderer.info` draw calls and triangles per frame.

### Before and after, Low preset, 64 ghosts

| | before | after |
| --- | --- | --- |
| fps (software GL) | 2.1 .. 3.6 (median 2.8) | 40.1 .. 40.8 at load ~4; 12 .. 29 at load ~5 to 6 |
| CPU-ms per frame (browser tree) | 274 .. 284 | 48 (59 under load) |
| main-thread long tasks > 50 ms | 8 .. 15 per 4 s | 0 at load ~4; 149 in 24 s at load ~5 to 6 |

"Before" is the build at HEAD (`9264d68`) served from a copy, measured with the same harness style (`scratch/ab*.mjs`).

### After, High preset, 64 ghosts

| | measured |
| --- | --- |
| fps (software GL) | 7.4, 7.0 (quick run); 6.9 median, 3.7 .. 7.5 (full run, loaded) |
| CPU-ms per frame | 250 .. 296 with the learner running (about 150 without it) |
| long tasks > 50 ms | 44 in 6 s; 146 in 24 s |

There is no usable "before" for High. The old build sent High through `WebGPURenderer`, which in headless
Chromium falls back to a WebGL2 backend that fails to initialise ("Initialization of backend webgl failed"),
so the old High page is not drawing a comparable frame (it reads 53 fps and 22 CPU-ms because it is nearly empty).
WebGPU is now opt-in (`?webgpu=1`), and High is the WebGL renderer with 4x MSAA, shadows, tone mapping and fog.

### Hardware-independent (identical on every run)

| | Low | High |
| --- | --- | --- |
| JS ms per frame, mean / p95 | 1.1 .. 1.4 / 4.8 | 2.9 / 6.6 .. 6.9 |
| draw calls / triangles | 11 .. 13 / 6.6k | 13 .. 14 / 6.7k .. 9.1k |

The 64 ghosts are one `InstancedMesh` draw. Draw calls and triangles are far below the 50 / 100000 budget.
The JS rows use a derived budget of 8 ms (half a 60 Hz frame), which is not a SPEC number.

### Where the bar is missed

- Low >= 60 fps: **missed under software GL** (about 40 fps on a quiet box, 12 .. 29 when the cores are contended).
  A real GPU is not the constraint here: JS is ~1.2 ms and the scene is 11 draw calls. What is left is
  SwiftShader rasterising 1280x720 plus the compositor readback, which is CPU work on shared cores.
- High >= 50 fps: **missed under software GL by a wide margin** (about 7 fps). The cost split measured with
  `?fx=` flags is roughly: 4x MSAA about 69 CPU-ms, shadow map about 30, lit shading with AgX about 16 (per frame).
  I did not lower High to reach the number; High keeps its MSAA, shadows and tone mapping.
- Long tasks > 50 ms: 0 at Low on a quiet box, failing whenever a frame takes more than 50 ms to reach the
  screen. Under software GL the main thread blocks in the compositor's ReadPixels until SwiftShader has
  finished the frame, so a slow raster shows up as a long task although the JS took ~1 ms.

I could not verify High or Low on a GPU. The lead should re-measure on an idle machine, serially, and with a GPU if one exists.

## What was changed, and what each change was worth

Profiling showed JS was never the problem (1.7 ms per frame before). The cost was in raster and compositing.

- `backdrop-filter` removed from the HUD panels: about 14 ms of compositor time per frame.
- Low is an unlit look: Basic materials, no tone mapping, linear output (no sRGB conversion pass), no MSAA.
  High keeps lit materials, AgX, MSAA and a shadow map. `src/viz/three/look.ts` swaps the materials; `setQuality`
  switches at runtime and only recreates the context when the antialias setting changes.
- The ground is a clear colour instead of a full-screen mesh in the top-down view; fog is dropped there.
- The shadow map is updated by hand every second frame with a frustum fitted to the pack.
- A pixel budget caps the drawing buffer; it does not reduce the car count.
- The hero car's geometry is merged; ghosts are one instanced mesh with a zero-allocation matrix write.
- Render is decoupled from messages: rAF draws every frame; the worker's `ghosts` messages only feed `src/viz/interp.ts`,
  which blends poses at 30 Hz over the observed message interval. Frames are not throttled.
- UI is throttled to 10 Hz; charts and heatmaps coalesce their redraws; the hidden PIP is placed visibly when shown.
- `src/viz/replay.ts` plays a pose tape at 25 Hz (`TAPE_HZ`) and exposes a deterministic probe, so one leaderboard row replays the same poses twice.
- `src/viz/frameStats.ts` records per-frame JS time, draw calls and triangles; `window.__drift` exposes it to the eval and e2e tests.
- `?fx=` URL flags (`noaa`, `basic`, `sm512`, `every4`, ...) exist only for profiling and default to off.

## The eval (`src/eval/fps.ts`)

- Builds the app with Vite and serves the production build with `vite preview` (cached under `/tmp/drift-school-eval`).
- Launches a fresh Chromium per preset with the real learner worker at 64 envs, waits for N PPO updates
  (2 in `--quick`, 3 otherwise), then measures several windows: median and min .. max fps, p95 frame, long tasks, CPU-ms per frame.
- Rows: fps at Low and at High, long tasks at Low and High, JS time per frame at both, draw calls and triangles,
  leaderboard rows equal to metrics updates (counted in the DOM and in the store, gens contiguous),
  and a row click replays the same poses twice (two probes identical, error against the tape reported).
- If Chromium cannot be found or cannot create a WebGL context, every row is `pass: null` with the reason.
- Chromium lookup: `src/eval/chromium.ts` uses the pinned playwright revision, then the highest revision installed under
  `PLAYWRIGHT_BROWSERS_PATH`, or `DRIFT_CHROMIUM=/path`. `playwright.config.ts` uses the same lookup and `locale: 'en-US'`
  (the container locale `en-US@posix` is an invalid language tag for Chromium). Nothing is downloaded.
- Thresholds are the SPEC's (60 / 50 fps, zero long tasks) and were not changed.

## Things for other owners

See `NOTES/viz-requests.md`: the tape is a boxed array and covers 0.64 s rather than a best lap, `ghosts` is published
once per PPO update rather than at 30 Hz, the pose buffer is copied twice, and the 64-env PPO update takes about 3 s
(a leaderboard row appears per update, so the leaderboard check waits on it). `src/workers/learner.worker.ts` was not edited.
