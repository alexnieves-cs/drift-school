/**
 * M3 render acceptance: 64 ghost cars, the real app, the learner worker training in the background.
 *
 * What it does
 *   1. Builds the app (vite build, cached by source mtime) and serves it with vite preview. No network.
 *   2. Launches Chromium through playwright. If it cannot launch, every row is `skip` with the
 *      reason; nothing is estimated.
 *   3. For the Low and High presets loads `/?cars=64&quality=<preset>` (64 training envs, 64 ghosts),
 *      waits for the learner to publish PPO updates, then measures several fixed windows of
 *      real requestAnimationFrame cadence. The render loop is never throttled and nothing is hidden.
 *   4. Counts main-thread long tasks (> 50 ms) with PerformanceObserver over the same windows.
 *   5. Checks the leaderboard has one row per PPO update and that replaying a row twice draws the
 *      same path (compared to the tape the worker sent).
 *
 * Reading the numbers
 *   The frame-rate rows are what this machine measured. Under software GL (SwiftShader, no GPU) the
 *   rasterizer shares the CPU cores with the training workers and with whatever else runs here, so
 *   they are pessimistic and noisy. The windows are reported as median (min..max). The rows after
 *   them are hardware independent: JS time per frame, draw calls and triangles from renderer.info.
 *   Those are the numbers to judge a real laptop by.
 */
import { existsSync, mkdtempSync, readdirSync, rmSync, statSync } from 'node:fs';
import { createServer } from 'node:net';
import { cpus, loadavg, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build, preview } from 'vite';
import { chromium, type BrowserContext, type Page } from '@playwright/test';
import { CHROMIUM_GL_ARGS, findBrowserPid, findChromium, treeCpuMs } from './chromium';
import { TAPE_HZ, poseAt } from '../viz/replay';
import type { EvalRow, Section } from './types';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const LONG_TASK_MS = 50;
const CARS = 64;
const FPS_TARGET = { low: 60, high: 50 } as const;
/** Derived budgets for the hardware-independent rows. They are not in SPEC.md. */
const JS_FRAME_BUDGET_MS = 8;
const DRAW_CALL_BUDGET = 50;
const TRIANGLE_BUDGET = 100_000;

type Preset = 'low' | 'high';

interface Window {
  fps: number;
  p95Ms: number;
  maxMs: number;
  frames: number;
  longTasks: number[];
  /** CPU time of the whole Chromium process tree per presented frame, or null off Linux. */
  cpuMsPerFrame: number | null;
}

interface Dist {
  mean: number;
  p50: number;
  p95: number;
  max: number;
}
interface FrameSummary {
  frames: number;
  update: Dist;
  render: Dist;
  ui: Dist;
  total: Dist;
  calls: number;
  triangles: number;
  scene: { drawingBufferWidth: number; drawingBufferHeight: number; pixelRatio: number; webgpu: boolean };
}

interface PresetRun {
  preset: Preset;
  windows: Window[];
  frame: FrameSummary;
  glRenderer: string;
  updatesBefore: number;
  page: Page;
}

function newestMtime(path: string): number {
  if (!existsSync(path)) return 0;
  const st = statSync(path);
  if (!st.isDirectory()) return st.mtimeMs;
  let newest = st.mtimeMs;
  for (const name of readdirSync(path)) newest = Math.max(newest, newestMtime(join(path, name)));
  return newest;
}

async function freePort(): Promise<number> {
  return new Promise((res, rej) => {
    const srv = createServer();
    srv.once('error', rej);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      srv.close(() => res(port));
    });
  });
}

async function buildApp(log: (s: string) => void): Promise<string> {
  const stamp = Math.round(
    Math.max(
      newestMtime(join(ROOT, 'src')),
      newestMtime(join(ROOT, 'index.html')),
      newestMtime(join(ROOT, 'dev')),
      newestMtime(join(ROOT, 'public')),
      newestMtime(join(ROOT, 'vite.config.ts')),
      newestMtime(join(ROOT, 'package-lock.json')),
    ),
  );
  const outDir = join(tmpdir(), 'drift-school-eval', String(stamp));
  if (existsSync(join(outDir, 'index.html'))) {
    log(`fps: reusing build ${outDir}`);
    return outDir;
  }
  log('fps: building the app (vite build)');
  await build({ root: ROOT, logLevel: 'error', build: { outDir, emptyOutDir: true, reportCompressedSize: false } });
  return outDir;
}

const PROBE = `
  window.__probe = { long: [] };
  try {
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) window.__probe.long.push([e.startTime, e.duration]);
    }).observe({ entryTypes: ['longtask'] });
  } catch (e) { window.__probe.unsupported = true; }
`;

async function measureWindow(page: Page, ms: number, pid: number): Promise<Window> {
  const cpu0 = treeCpuMs(pid);
  const w = await measureFrames(page, ms);
  const cpu1 = treeCpuMs(pid);
  const frames = w.frames + 1;
  return { ...w, cpuMsPerFrame: cpu0 !== null && cpu1 !== null ? (cpu1 - cpu0) / frames : null };
}

async function measureFrames(page: Page, ms: number): Promise<Omit<Window, 'cpuMsPerFrame'>> {
  return page.evaluate(
    async (span: number) => {
      const probe = (window as unknown as { __probe: { long: [number, number][] } }).__probe;
      const dts: number[] = [];
      const t0 = performance.now();
      let first = 0;
      let last = 0;
      await new Promise<void>((done) => {
        const tick = (t: number) => {
          if (first === 0) first = t;
          else dts.push(t - last);
          last = t;
          if (performance.now() - t0 >= span) done();
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
      const t1 = performance.now();
      const sorted = [...dts].sort((a, b) => a - b);
      const longTasks = probe.long.filter(([s]) => s >= t0 && s <= t1).map(([, d]) => d);
      return {
        fps: dts.length > 0 ? (dts.length * 1000) / (last - first) : 0,
        p95Ms: sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))]! : 0,
        maxMs: sorted.length ? sorted[sorted.length - 1]! : 0,
        frames: dts.length,
        longTasks,
      };
    },
    ms,
  );
}

async function runPreset(
  ctx: BrowserContext,
  pid: number,
  url: string,
  preset: Preset,
  opts: { windows: number; windowMs: number; updateTimeoutMs: number; minUpdates: number },
  log: (s: string) => void,
): Promise<PresetRun> {
  const page = await ctx.newPage();
  await page.addInitScript(PROBE);
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${url}/?cars=${CARS}&quality=${preset}`, { waitUntil: 'load' });
  await page.waitForFunction(() => Boolean((window as unknown as { __drift?: unknown }).__drift), null, { timeout: 60_000 });
  // The learner publishes one metrics message per PPO update. Wait for a few so the leaderboard,
  // charts and worker traffic are all live while the windows run.
  const t0 = Date.now();
  log(`fps: ${preset}: waiting for ${opts.minUpdates} PPO update(s) from the 64-env learner`);
  let updates = 0;
  while (Date.now() - t0 < opts.updateTimeoutMs) {
    updates = await page.evaluate(() => (window as unknown as { __drift: { metricsCount: () => number } }).__drift.metricsCount());
    if (updates >= opts.minUpdates) break;
    await page.waitForTimeout(500);
  }
  log(`fps: ${preset}: ${updates} update(s) after ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  await page.waitForTimeout(1500);
  const windows: Window[] = [];
  await page.evaluate(() => (window as unknown as { __drift: { resetFrameStats: () => void } }).__drift.resetFrameStats());
  for (let i = 0; i < opts.windows; i++) {
    const w = await measureWindow(page, opts.windowMs, pid);
    windows.push(w);
    log(
      `fps: ${preset}: window ${i + 1}/${opts.windows}: ${w.fps.toFixed(1)} fps, p95 ${w.p95Ms.toFixed(0)} ms, ${w.longTasks.length} long task(s)` +
        (w.cpuMsPerFrame === null ? '' : `, ${w.cpuMsPerFrame.toFixed(0)} cpu-ms/frame`),
    );
  }
  const frame = await page.evaluate(() => (window as unknown as { __drift: { frameStats: () => unknown } }).__drift.frameStats());
  const glRenderer = await page.evaluate(() => {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2');
    const ext = gl?.getExtension('WEBGL_debug_renderer_info');
    return gl && ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : 'unknown';
  });
  if (errors.length) log(`fps: ${preset}: page errors: ${errors.slice(0, 3).join(' | ')}`);
  return { preset, windows, frame: frame as FrameSummary, glRenderer, updatesBefore: updates, page };
}

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)]! : 0;
};
const f1 = (x: number) => x.toFixed(1);

function skipAll(reason: string): EvalRow[] {
  const mk = (test: string, target: string): EvalRow => ({ milestone: 'M3', test, target, measured: 'not run', pass: null, note: reason });
  return [
    mk('64 ghost cars frame rate, Low preset', `>= ${FPS_TARGET.low} fps`),
    mk('64 ghost cars frame rate, High preset', `>= ${FPS_TARGET.high} fps`),
    mk('main-thread long tasks > 50 ms, Low preset', '0'),
    mk('main-thread long tasks > 50 ms, High preset', '0'),
    mk('JS time per frame (hardware independent), Low', `<= ${JS_FRAME_BUDGET_MS} ms`),
    mk('JS time per frame (hardware independent), High', `<= ${JS_FRAME_BUDGET_MS} ms`),
    mk('draw calls / triangles per frame', `<= ${DRAW_CALL_BUDGET} / ${TRIANGLE_BUDGET}`),
    mk('leaderboard rows == PPO updates', 'equal'),
    mk('click row replays a deterministic ghost lap', 'identical poses'),
  ];
}

export const fps: Section = async ({ quick, log }) => {
  const exe = findChromium();
  if (!exe) return skipAll('no Chromium found (set DRIFT_CHROMIUM or install a browser into PLAYWRIGHT_BROWSERS_PATH)');
  // A fresh browser process per preset (own profile dir) so one preset cannot warm or poison the
  // other, and so the process tree's CPU time can be attributed to it.
  const launch = async (): Promise<{ ctx: BrowserContext; pid: number; dir: string }> => {
    const dir = mkdtempSync(join(tmpdir(), 'drift-eval-profile-'));
    const ctx = await chromium.launchPersistentContext(dir, {
      executablePath: exe,
      headless: true,
      args: CHROMIUM_GL_ARGS,
      viewport: { width: 1280, height: 720 },
      locale: 'en-US',
    });
    return { ctx, pid: findBrowserPid(dir), dir };
  };
  try {
    const probe = await launch();
    await probe.ctx.close();
    rmSync(probe.dir, { recursive: true, force: true });
  } catch (err) {
    return skipAll(`Chromium could not launch: ${err instanceof Error ? err.message.split('\n')[0] : String(err)}`);
  }

  let server: Awaited<ReturnType<typeof preview>> | null = null;
  try {
    const outDir = await buildApp(log);
    const port = await freePort();
    server = await preview({
      root: ROOT,
      logLevel: 'error',
      build: { outDir },
      preview: { host: '127.0.0.1', port, strictPort: true, open: false },
    });
    const url = `http://127.0.0.1:${port}`;
    const opts = quick
      ? { windows: 2, windowMs: 3_000, updateTimeoutMs: 150_000, minUpdates: 2 }
      : { windows: 3, windowMs: 8_000, updateTimeoutMs: 300_000, minUpdates: 3 };
    const runs = {} as Record<Preset, PresetRun>;
    let checks: EvalRow[] = [];
    const load = loadavg()[0]!;
    for (const preset of ['low', 'high'] as const) {
      const { ctx, pid, dir } = await launch();
      try {
        runs[preset] = await runPreset(ctx, pid, url, preset, opts, log);
        // Correctness checks run on the Low page, which has had the most PPO updates by now.
        if (preset === 'low') checks = await correctness(runs.low.page, log);
      } finally {
        await ctx.close().catch(() => undefined);
        rmSync(dir, { recursive: true, force: true });
      }
    }
    const cores = cpus().length;
    const hw = `${runs.low.glRenderer}; ${cores} cores, load ${load.toFixed(1)} at start; 1280x720; learner n=${CARS}`;
    const rows: EvalRow[] = [];
    for (const preset of ['low', 'high'] as const) {
      const r = runs[preset];
      const label = preset === 'low' ? 'Low' : 'High';
      const fpsList = r.windows.map((w) => w.fps);
      const med = median(fpsList);
      const target = FPS_TARGET[preset];
      const cpu = r.windows.map((w) => w.cpuMsPerFrame).filter((x): x is number => x !== null);
      rows.push({
        milestone: 'M3',
        test: `64 ghost cars frame rate, ${label} preset`,
        target: `>= ${target} fps`,
        measured: `${f1(med)} fps median (${f1(Math.min(...fpsList))}..${f1(Math.max(...fpsList))} over ${r.windows.length} windows), p95 frame ${f1(median(r.windows.map((w) => w.p95Ms)))} ms${cpu.length ? `, ${median(cpu).toFixed(0)} CPU-ms/frame in software GL` : ''}`,
        pass: med >= target,
        note: `${hw}. Software GL rasterizes on the same CPU cores as the learner and everything else running, so this is pessimistic and noisy; CPU-ms/frame (all Chromium threads) moves less with load. See the hardware independent rows for a real GPU.`,
      });
    }
    for (const preset of ['low', 'high'] as const) {
      const r = runs[preset];
      const all = r.windows.flatMap((w) => w.longTasks);
      const worst = all.length ? Math.max(...all) : 0;
      const over = all.filter((d) => d > LONG_TASK_MS).length;
      rows.push({
        milestone: 'M3',
        test: `main-thread long tasks > 50 ms, ${preset === 'low' ? 'Low' : 'High'} preset`,
        target: '0',
        measured: `${over} in ${r.windows.length} x ${opts.windowMs / 1000} s, worst ${f1(worst)} ms`,
        pass: over === 0,
        note:
          'PerformanceObserver longtask. Under software GL the compositor blocks the main thread in ReadPixels until SwiftShader finishes the frame, so a frame slower than 50 ms shows up here even though the JS took ~1 ms.',
      });
    }
    for (const preset of ['low', 'high'] as const) {
      const f = runs[preset].frame;
      rows.push({
        milestone: 'M3',
        test: `JS time per frame (hardware independent), ${preset === 'low' ? 'Low' : 'High'}`,
        target: `<= ${JS_FRAME_BUDGET_MS} ms mean and p95`,
        measured: `mean ${f.total.mean.toFixed(2)} ms, p95 ${f.total.p95.toFixed(2)}, max ${f.total.max.toFixed(1)} (update ${f.update.mean.toFixed(2)}, render submit ${f.render.mean.toFixed(2)}, ui ${f.ui.mean.toFixed(2)}) over ${f.frames} frames`,
        pass: f.total.mean <= JS_FRAME_BUDGET_MS && f.total.p95 <= JS_FRAME_BUDGET_MS,
        note: `Derived budget (half of a 60 Hz frame), not a SPEC number. Drawing buffer ${f.scene.drawingBufferWidth}x${f.scene.drawingBufferHeight}, pixel ratio ${f.scene.pixelRatio.toFixed(2)}.`,
      });
    }
    const lo = runs.low.frame;
    const hi = runs.high.frame;
    rows.push({
      milestone: 'M3',
      test: 'draw calls / triangles per frame',
      target: `<= ${DRAW_CALL_BUDGET} / ${TRIANGLE_BUDGET}`,
      measured: `Low ${lo.calls} / ${lo.triangles}, High ${hi.calls} / ${hi.triangles}`,
      pass: Math.max(lo.calls, hi.calls) <= DRAW_CALL_BUDGET && Math.max(lo.triangles, hi.triangles) <= TRIANGLE_BUDGET,
      note: 'renderer.info.render summed over the main view and the POV picture-in-picture pass. 64 ghosts are one instanced draw.',
    });
    rows.push(...checks);
    return rows;
  } finally {
    await server?.close().catch(() => undefined);
  }
};

async function correctness(page: Page, log: (s: string) => void): Promise<EvalRow[]> {
  const rows: EvalRow[] = [];
  const board = await page.evaluate(() => {
    const d = (window as unknown as { __drift: { metricsCount: () => number; boardRows: () => number; boardGens: () => number[] } }).__drift;
    const dom = document.querySelectorAll('table.board tbody tr').length;
    return { updates: d.metricsCount(), rows: d.boardRows(), gens: d.boardGens(), dom };
  });
  const contiguous = board.gens.every((g, i) => g === i + 1);
  rows.push({
    milestone: 'M3',
    test: 'leaderboard rows == PPO updates',
    target: 'equal',
    measured: `${board.rows} rows (${board.dom} in the DOM) vs ${board.updates} metrics updates, gens ${contiguous ? '1..N contiguous' : 'NOT contiguous'}`,
    pass: board.updates > 0 && board.rows === board.updates && board.dom === board.updates && contiguous,
    note: board.updates === 0 ? 'the learner published no PPO update in the time allowed' : undefined,
  });

  if (board.updates === 0) {
    rows.push({ milestone: 'M3', test: 'click row replays a deterministic ghost lap', target: 'identical poses', measured: 'no rows to click', pass: false });
    return rows;
  }
  const gen = board.updates;
  const sel = `table.board tbody tr[data-gen="${gen}"]`;
  const clicked: boolean[] = [];
  for (let k = 0; k < 2; k++) {
    await page.click(sel);
    const started = await page
      .waitForFunction((g: number) => document.body.dataset.replay === String(g), gen, { timeout: 3_000 })
      .then(() => true)
      .catch(() => false);
    const ended = started
      ? await page
          .waitForFunction(() => document.body.dataset.replay === '', null, { timeout: 15_000 })
          .then(() => true)
          .catch(() => false)
      : false;
    clicked.push(started && ended);
  }
  const probe = await page.evaluate((g: number) => {
    const d = (window as unknown as { __drift: { probeReplay: (g: number) => number[] | null; tape: (g: number) => number[] | null } }).__drift;
    return { a: d.probeReplay(g), b: d.probeReplay(g), tape: d.tape(g) };
  }, gen);
  const same = !!probe.a && !!probe.b && probe.a.length > 0 && probe.a.length === probe.b.length && probe.a.every((v, i) => v === probe.b![i]);
  // Independent check: the drawn path must follow the tape the worker sent.
  let maxErr = Infinity;
  if (probe.a && probe.tape) {
    maxErr = 0;
    const tape = Float32Array.from(probe.tape);
    const pose = { x: 0, y: 0, z: 0, yaw: 0 };
    for (let i = 0; i * 4 < probe.a.length; i++) {
      poseAt(tape, (i + 1) / 30, pose);
      maxErr = Math.max(
        maxErr,
        Math.abs(pose.x - probe.a[i * 4]!),
        Math.abs(pose.y - probe.a[i * 4 + 1]!),
        Math.abs(pose.z - probe.a[i * 4 + 2]!),
        Math.abs(pose.yaw - probe.a[i * 4 + 3]!),
      );
    }
  }
  log(`fps: replay of gen ${gen}: ${probe.a ? probe.a.length / 4 : 0} steps at 30 Hz, tape ${TAPE_HZ} Hz`);
  rows.push({
    milestone: 'M3',
    test: 'click row replays a deterministic ghost lap',
    target: 'identical poses',
    measured: `click -> replay -> end x2: ${clicked.every(Boolean) ? 'ok' : 'FAILED'}; two probes ${same ? 'identical' : 'DIFFER'} (${probe.a ? probe.a.length / 4 : 0} steps); max error vs tape ${Number.isFinite(maxErr) ? maxErr.toExponential(1) : 'n/a'}`,
    pass: clicked.every(Boolean) && same && maxErr < 1e-6,
    note: 'The tape is the worker\'s pose recording of car 0 over one PPO rollout, not a full best lap; see NOTES/viz-requests.md.',
  });
  return rows;
}
