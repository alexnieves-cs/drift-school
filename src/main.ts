import * as THREE from 'three';
import { QUALITY, ThreeRenderer, type HeroState, type SceneCounts } from './viz/renderer';
import { FrameStats, type FrameSummary } from './viz/frameStats';
import { ReplayPlayer } from './viz/replay';
import { Hud } from './viz/cameras/hud';
import { mountControls } from './ui/controls';
import { MetricCharts } from './ui/charts';
import { Leaderboard } from './ui/leaderboard';
import { TelemetryPanel } from './ui/telemetry';
import { Heatmap } from './ui/heatmaps';
import { generateTrack } from './track/generate';
import { sampleAtS } from './track/sample';
import { buildTrackFromPoints } from './track/build';
import { encodeTrackHash, decodeTrackHash, quantizePoints } from './editor/share';
import { toFloat } from './editor/stroke';
import { TrackEditor, type EditorState } from './editor/editor';
import { AdaptationPanel } from './ui/adaptation';
import { FineTune, type FineTuneReport } from './rl/finetune';
import { openEnv, openLearner } from './workers/pool';
import { saveCheckpoint, loadCheckpoint } from './rl/checkpoint';
import type { CameraMode, GenStat, TrackSpec } from './shared/types';
import type { CarSnap } from './sim/vecEnv';

const params = new URLSearchParams(location.search);
const view = params.get('view') ?? 'race';

const app = document.querySelector('#app') as HTMLElement;
app.innerHTML = `
  <canvas id="view"></canvas>
  <aside id="left"></aside>
  <aside id="right">
    <div id="board"></div>
    <div id="charts"></div>
  </aside>
  <footer id="bottom">
    <div id="telemetry"></div>
    <div id="heat"></div>
  </footer>
  <div id="hud-host"></div>
  <div id="editor" hidden><canvas id="stroke"></canvas></div>
`;

const canvas = document.querySelector('#view') as HTMLCanvasElement;
const left = document.querySelector('#left') as HTMLElement;
const renderer = new ThreeRenderer(canvas);
const hud = new Hud(document.querySelector('#hud-host') as HTMLElement);
const charts = new MetricCharts(document.querySelector('#charts') as HTMLElement);
const heat = new Heatmap(document.querySelector('#heat') as HTMLElement);
const telemetry = new TelemetryPanel(document.querySelector('#telemetry') as HTMLElement);
const adapt = document.createElement('p');
adapt.className = 'adapt';
adapt.dataset.adapt = 'idle';

const tapes = new Map<number, Float32Array>();
const replayer = new ReplayPlayer();
const frameStats = new FrameStats();
let track: TrackSpec = generateTrack({ seed: 1, stage: 0 });
let bestLap: number | null = null;
let lastHero: HeroState | null = null;
let lastSnap: CarSnap | null = null;
let crashLatch = false;
let metricsCount = 0;
let ghostCount = 0;
let ghostHighlight = -1;
let mapCache: { id: string; pts: { x: number; z: number }[] } | null = null;

/** Hooks for the fps eval and the e2e tests. Read-only except where noted. */
interface DriftDebug {
  viz: ThreeRenderer;
  frameStats: () => FrameSummary & { scene: SceneCounts };
  resetFrameStats: () => void;
  metricsCount: () => number;
  boardRows: () => number;
  boardGens: () => number[];
  tape: (gen: number) => number[] | null;
  /** Steps a fresh ReplayPlayer through the renderer at a fixed 30 Hz and returns x,y,z,yaw per step. */
  probeReplay: (gen: number) => number[] | null;
  replayGen: () => number;
}
declare global {
  interface Window {
    __drift?: DriftDebug;
  }
}

function startReplay(gen: number): void {
  const tape = tapes.get(gen);
  if (!tape) {
    if (worker) worker.postMessage({ type: 'replayReq', gen });
    return;
  }
  if (!replayer.start(gen, tape)) return;
  document.body.dataset.replay = String(gen);
  board.mark(gen);
  controls.setStatus(`replay gen ${gen}`, 'replay');
}

function endReplay(): void {
  renderer.setReplay(null);
  board.mark(null);
  document.body.dataset.replay = '';
  controls.setStatus('racing', 'racing');
}

const board = new Leaderboard(document.querySelector('#board') as HTMLElement, startReplay);

const controls = mountControls(left, (state, reason) => {
  if (reason === 'camera') renderer.setCamera(state.camera);
  if (reason === 'quality') renderer.setQuality(state.quality);
  if (reason === 'pause') send({ type: 'pause', paused: state.paused });
  if (reason === 'baseline') send({ type: 'baseline', on: state.baseline });
  if (reason === 'stage') send({ type: 'setStage', stage: state.stage });
  if (reason === 'grip' || reason === 'mass' || reason === 'dr') {
    send({ type: 'setDR', grip: state.grip, mass: state.mass, level: state.dr });
  }
  if (reason === 'editor') toggleEditor();
  if (reason === 'train') void trainOnDrawn();
  if (reason === 'save') void persist('manual');
  if (reason === 'load') void restore('manual');
  if (reason === 'cars') controls.setStatus(`${state.cars} cars on next load`, 'note');
});
left.appendChild(adapt);
const adaptation = new AdaptationPanel(left);
/** Set by `trainOnDrawn`: the drawn track's id and when the button was pressed. */
let swap: { id: string; t0: number } | null = null;

let worker: Worker | null = null;
let inline: import('./rl/trainer').Trainer | null = null;

function send(msg: Record<string, unknown>): void {
  if (worker) worker.postMessage(msg);
  else applyInline(msg);
}

function mapPts(t: TrackSpec): { x: number; z: number }[] {
  if (mapCache?.id === t.id) return mapCache.pts;
  const step = Math.max(1, Math.floor(t.samples.length / 80));
  const pts = [];
  for (let i = 0; i < t.samples.length; i += step) pts.push(t.samples[i]!);
  mapCache = { id: t.id, pts };
  return pts;
}

/** Message-rate work: build the hero state and record what the panels will show at the next tick. */
function heroFrom(snap: CarSnap | null, fallback: HeroState): HeroState {
  if (!snap) return fallback;
  lastSnap = snap;
  const slipRear = (Math.abs(snap.slip[2] ?? 0) + Math.abs(snap.slip[3] ?? 0)) / 2;
  if (slipRear > 0.4) crashLatch = true;
  heat.note(track, snap.s, snap.speed, crashLatch, snap.x, snap.z);
  crashLatch = false;
  return {
    x: snap.x,
    y: snap.y,
    z: snap.z,
    yaw: snap.yaw,
    pitch: snap.pitch,
    roll: snap.roll,
    speed: snap.speed,
    steer: snap.steer / 0.48,
    throttle: snap.throttle,
    slipRear,
    load: snap.load,
    lap: snap.lap,
    lapTime: snap.lapTime,
    bestLap,
  };
}

/**
 * Hand a new snapshot to the renderer. This is all a producer message costs the main thread:
 * two buffer copies. Drawing happens in the animation-frame loop below, at display rate.
 */
function feed(poses: Float32Array, n: number, hero: HeroState, highlight: number): void {
  ghostCount = n;
  ghostHighlight = highlight;
  lastHero = hero;
  renderer.updateCars(poses, n, highlight);
  renderer.pushHero(hero);
}

function onMetrics(stat: GenStat, tape?: ArrayLike<number>): void {
  metricsCount++;
  if (tape?.length) tapes.set(stat.gen, Float32Array.from(tape));
  if (stat.bestLap != null) bestLap = bestLap == null ? stat.bestLap : Math.min(bestLap, stat.bestLap);
  board.add(stat);
  charts.push(stat);
  adapt.dataset.adapt = `gen ${stat.gen} completion ${(stat.completion * 100).toFixed(0)}%`;
}

function applyInline(msg: Record<string, unknown>): void {
  if (!inline) return;
  if (msg.type === 'pause') return;
  if (msg.type === 'baseline') inlineModeBaseline = Boolean(msg.on);
  if (msg.type === 'setStage') inline.curriculum.stage = Number(msg.stage ?? 0);
  if (msg.type === 'setDR') {
    inline.grip = Number(msg.grip ?? 1);
    inline.massScale = Number(msg.mass ?? 1);
    inline.dr = (msg.level as 'off' | 'low' | 'high') ?? 'off';
  }
  if (msg.type === 'setTrackPool') void swapInline(msg);
}

let inlineModeBaseline = false;
let inlinePaused = false;

let inlineFine: FineTune | null = null;
let inlineProc: TrackSpec[] = [];

async function swapInline(msg: Record<string, unknown>): Promise<void> {
  if (!inline) return;
  const points = quantizePoints((msg.points as number[]) ?? []);
  const halfWidth = Number(msg.halfWidth ?? 1);
  const mu = Number(msg.mu ?? 1);
  const user = buildTrackFromPoints(points, { halfWidth, mu, source: 'user' });
  inlineFine?.stop();
  inlineFine = new FineTune(inline, { user, procedural: inlineProc });
  inlineFine.start();
  track = user;
  renderer.loadTrack(track);
  history.replaceState(null, '', `#${encodeTrackHash(points, halfWidth, mu)}`);
  document.body.dataset.track = user.id;
}

async function persist(name: string): Promise<void> {
  if (worker) worker.postMessage({ type: 'checkpointSave', name });
  if (!inline) return;
  const weights = inline.policy.getWeights().map((b) => Array.from(new Float32Array(b)));
  await saveCheckpoint({ name, gen: inline.gen, stage: inline.curriculum.stage, weights });
  controls.setStatus(`saved ${name}`, 'saved');
}

async function restore(name: string): Promise<void> {
  const cp = await loadCheckpoint(name);
  if (!cp || !inline) {
    controls.setStatus(cp ? 'loaded' : 'no checkpoint', cp ? 'loaded' : 'miss');
    return;
  }
  const buffers = cp.weights.map((w) => Float32Array.from(w).buffer);
  inline.policy.setWeights(buffers);
  controls.setStatus(`loaded ${name}`, 'loaded');
}

function toggleEditor(): void {
  const el = document.querySelector('#editor') as HTMLElement;
  el.hidden = !el.hidden;
}

let editorState: Partial<EditorState> = {};

function mountEditor(): void {
  const stroke = document.querySelector('#stroke') as HTMLCanvasElement;
  const fit = () => {
    stroke.width = window.innerWidth;
    stroke.height = window.innerHeight;
  };
  fit();
  const editor = new TrackEditor(stroke, {
    width: stroke.width,
    height: stroke.height,
    scale: 0.04,
    onChange: (state) => {
      editorState = state;
      adapt.textContent = state.validation.ok ? `valid ${state.ms.toFixed(1)} ms` : state.validation.issues.map((i) => i.reason).join(', ');
    },
  });
  if (location.hash.length > 2) {
    try {
      const decoded = decodeTrackHash(location.hash.slice(1));
      editor.load(decoded.points, decoded.halfWidth, decoded.mu);
    } catch {
      adapt.textContent = 'hash was not a track';
    }
  }
  window.addEventListener('resize', fit);
}

/** Count how many envs the learner reports on the drawn track; stamp when all of them are. */
function noteEnvTracks(ids: string[] | undefined): void {
  if (!swap || !ids?.length) return;
  const d = document.body.dataset;
  const on = ids.filter((id) => id === swap!.id).length;
  d.envsOnUser = String(on);
  d.envTotal = String(ids.length);
  if (on === ids.length && !d.allOnUserMs) d.allOnUserMs = String(Math.round(performance.now() - swap.t0));
}

async function trainOnDrawn(): Promise<void> {
  const state = editorState as Partial<EditorState>;
  if (!state.track || !state.validation?.ok || !state.points) {
    controls.setStatus('track is not valid', 'bad');
    return;
  }
  // Snap to mm: the track id, the URL hash and the learner's track all come from the same points.
  const points = quantizePoints(toFloat(state.points));
  const halfWidth = state.halfWidth ?? 1;
  const mu = state.mu ?? 1;
  const built = buildTrackFromPoints(points, { halfWidth, mu, source: 'user' });
  const d = document.body.dataset;
  for (const k of ['envsOnUser', 'envTotal', 'allOnUserMs', 'swapAckTrack', 'swapAckEnvs', 'swapAckMs', 'swapHashEqual']) delete d[k];
  swap = { id: built.id, t0: performance.now() };
  d.userTrack = built.id;
  adaptation.begin(built.id);
  send({ type: 'setTrackPool', points: Array.from(points), halfWidth, mu, fineTune: true });
  track = built;
  renderer.loadTrack(track);
  history.replaceState(null, '', `#${encodeTrackHash(points, halfWidth, mu)}`);
  (document.querySelector('#editor') as HTMLElement).hidden = true;
  controls.setStatus('training on drawn track', 'adapt');
}

/**
 * Synthetic 30 Hz pack for the ghost view: `cars` cars lapping the track plus one lead car.
 * It stands in for the worker's pose stream (same buffers, same cadence) so the render path is
 * identical, and it is a pure function of time so screenshots are reproducible.
 */
function syntheticStream(): (now: number) => void {
  const n = Math.min(64, Math.max(1, Number(params.get('cars') ?? 64)));
  const poses = new Float32Array(n * 7);
  const t0 = performance.now();
  let lastPush = -Infinity;
  document.body.dataset.track = track.id;
  return (now) => {
    if (now - lastPush < 30) return;
    lastPush = now;
    const t = (now - t0) / 1000;
    for (let i = 0; i < n; i++) {
      const dist = (t * (1.6 + i * 0.03) + (i / n) * track.length) % track.length;
      const s = sampleAtS(track, dist);
      const yaw = Math.atan2(s.tz, s.tx);
      const o = i * 7;
      poses[o] = s.x;
      poses[o + 1] = 0.05;
      poses[o + 2] = s.z;
      poses[o + 3] = Math.cos(yaw / 2);
      poses[o + 5] = Math.sin(yaw / 2);
    }
    const lead = sampleAtS(track, (t * 2.2) % track.length);
    feed(
      poses,
      n,
      {
        x: lead.x,
        y: 0.05,
        z: lead.z,
        yaw: Math.atan2(lead.tz, lead.tx),
        pitch: 0,
        roll: 0,
        speed: 3.2,
        steer: Math.sin(t) * 0.2,
        throttle: 0.7,
        slipRear: 0.05,
        lap: Math.floor(t / 12),
        lapTime: t % 12,
        bestLap: 11.4,
      },
      -1,
    );
  };
}

let stream: ((now: number) => void) | null = null;

async function startInline(): Promise<void> {
  const { Trainer } = await import('./rl/trainer');
  const seeds = [1, 2, 3, 4].map((seed) => generateTrack({ seed, stage: 0 }));
  track = seeds[0]!;
  inlineProc = seeds;
  renderer.loadTrack(track);
  inline = new Trainer({ n: Number(params.get('cars') ?? 8), horizon: 32, epochs: 2, minibatch: 256 });
  controls.setStatus('imitating pure pursuit', 'imitating');
  await inline.init(seeds);
  await inline.imitate(3);
  controls.setStatus('racing', 'racing');
  document.body.dataset.track = inline.env.trackId(0);
  // Training runs on the main thread in this fallback, so it blocks frames while an update runs.
  // It is decoupled from drawing: the animation-frame loop keeps presenting the latest poses.
  const loop = async () => {
    if (!inline) return;
    if (!inlinePaused && !inlineModeBaseline && inlineFine && inlineFine.phase !== 'done') {
      const gen0 = inline.gen;
      const rep = await inlineFine.step();
      adaptation.update(rep);
      const stat = inline.history[inline.history.length - 1];
      if (inline.gen !== gen0 && stat) onMetrics(stat, undefined);
      document.body.dataset.track = inline.env.trackId(0);
      noteEnvTracks(inline.env.trackIds());
      if (rep.latencyMs !== null && !document.body.dataset.swapAckMs && swap) {
        document.body.dataset.swapAckMs = String(Math.round(performance.now() - swap.t0));
        document.body.dataset.swapHashEqual = String(rep.weightHashAtDrain === rep.weightHashStart);
        document.body.dataset.swapAckEnvs = `${rep.envsOnUserAtSwap}/${rep.envTotal}`;
      }
    } else if (!inlinePaused && !inlineModeBaseline) {
      const stat = await inline.update();
      onMetrics(stat, inline.replays.get(inline.gen));
    } else if (inlineModeBaseline) {
      inline.env.step(inline.env.baselineActions());
    }
    const snap = inline.env.snapshot(0);
    feed(inline.env.poses(), inline.env.n, heroFrom(snap, lastHero ?? blankHero()), 0);
    inlinePaused = controls.read().paused;
    setTimeout(() => void loop(), 0);
  };
  void loop();
}

function blankHero(): HeroState {
  const s = track.samples[track.startIndex] ?? track.samples[0]!;
  return { x: s.x, y: 0.05, z: s.z, yaw: Math.atan2(s.tz, s.tx), pitch: 0, roll: 0, speed: 0 };
}

function startWorker(): void {
  worker = openLearner();
  const env = openEnv();
  env.postMessage({ type: 'ping', t: 7 });
  let trackLabel = '';
  worker.onmessage = (ev: MessageEvent) => {
    const msg = ev.data as Record<string, unknown>;
    if (msg.type === 'status') controls.setStatus(String(msg.text), String(msg.text).includes('racing') ? 'racing' : 'live');
    if (msg.type === 'ghosts') {
      const hero = heroFrom((msg.hero as CarSnap) ?? null, lastHero ?? blankHero());
      if (typeof msg.bestLap === 'number') bestLap = msg.bestLap;
      // A synthetic stream (ghost view with ?train=1) owns the cars; the worker then only feeds panels.
      if (!stream) feed(msg.poses as Float32Array, Number(msg.n ?? 0), hero, 0);
      const id = String(msg.trackId ?? '');
      if (id !== trackLabel) {
        trackLabel = id;
        document.body.dataset.track = id;
      }
      noteEnvTracks(msg.ids as string[] | undefined);
      worker?.postMessage({ type: 'ghostAck' });
    }
    if (msg.type === 'metrics') onMetrics(msg.stat as GenStat, msg.tape as ArrayLike<number> | undefined);
    if (msg.type === 'replayRes') {
      const poses = msg.poses as ArrayLike<number>;
      if (poses?.length) {
        tapes.set(Number(msg.gen), Float32Array.from(poses));
        startReplay(Number(msg.gen));
      }
    }
    if (msg.type === 'trackAck') {
      adapt.dataset.adapt = `track ${String(msg.trackId)} envs ${String(msg.envCount)}/${String(msg.envTotal)} hash ${String(msg.hash)}`;
      const d = document.body.dataset;
      d.swapAckTrack = String(msg.trackId);
      d.swapAckEnvs = `${String(msg.envCount)}/${String(msg.envTotal)}`;
      d.swapAckMs = swap ? String(Math.round(performance.now() - swap.t0)) : '';
      d.swapHashEqual = String(msg.hash === msg.hashBefore);
    }
    if (msg.type === 'adapt') adaptation.update(msg as unknown as FineTuneReport);
    if (msg.type === 'error') controls.setStatus(String(msg.message), 'bad');
  };
  worker.onerror = () => {
    worker?.terminate();
    worker = null;
    controls.setStatus('worker failed, inline', 'inline');
    void startInline();
  };
  worker.postMessage({ type: 'init', n: Number(params.get('cars') ?? 8), seeds: [1, 2, 3, 4], stage: 0 });
}

window.addEventListener('keydown', (ev) => {
  const cams: Record<string, CameraMode> = { '1': 'topdown', '2': 'chase', '3': 'pov' };
  const mode = cams[ev.key];
  if (!mode) return;
  renderer.setCamera(mode);
  left.dataset.camera = mode;
});
let resizePending = false;
window.addEventListener('resize', () => {
  resizePending = true;
});

/** The POV picture-in-picture sits under the controls and above the bottom panel, not behind it. */
function placePip(): void {
  const footer = document.querySelector('#bottom') as HTMLElement | null;
  const hidden = !footer || footer.offsetParent === null;
  const bottom = hidden ? 16 : Math.max(8, window.innerHeight - footer.getBoundingClientRect().top + 8);
  renderer.setPip({ left: 12, bottom, width: 240 });
}

// ---- frame loop: one requestAnimationFrame chain draws everything, whatever the producer does.

const UI_INTERVAL_MS = 100;
let uiStamp = -Infinity;
let frameStamp = 0;
let fpsFrames = 0;
let fpsStart = 0;

function uiTick(now: number): void {
  uiStamp = now;
  const h = lastHero;
  if (h) {
    hud.update(
      {
        speed: h.speed,
        lap: h.lap ?? 0,
        lapTime: h.lapTime ?? 0,
        bestLap,
        steer: h.steer ?? 0,
        throttle: h.throttle ?? 0,
        slipRear: h.slipRear ?? 0,
      },
      mapPts(track),
      h,
    );
  }
  const snap = lastSnap;
  if (snap) {
    telemetry.update({
      speed: snap.speed,
      steer: snap.steer,
      throttle: snap.throttle,
      slip: snap.slip,
      load: snap.load,
      ax: snap.vx / 9.81,
      ay: (snap.yawRate * snap.speed) / 9.81,
    });
  }
  heat.flush(track);
  charts.flush();
}

function onFrame(now: number): void {
  const t0 = performance.now();
  const dt = frameStamp ? Math.min(0.1, Math.max(0.001, (now - frameStamp) / 1000)) : 0.016;
  frameStamp = now;
  if (resizePending) {
    resizePending = false;
    renderer.resize();
    placePip();
  }
  stream?.(now);
  if (replayer.active) {
    const pose = replayer.advance(dt);
    if (pose) renderer.setReplay(pose);
    else endReplay();
  }
  const t1 = performance.now();
  const timing = renderer.frame(ghostCount, ghostHighlight, null, dt);
  const t2 = performance.now();
  if (now - uiStamp >= UI_INTERVAL_MS) uiTick(now);
  const t3 = performance.now();
  frameStats.record({
    updateMs: t1 - t0 + timing.updateMs,
    renderMs: timing.renderMs,
    uiMs: t3 - t2,
    totalMs: t3 - t0,
    calls: timing.calls,
    triangles: timing.triangles,
  });
  fpsFrames++;
  if (now - fpsStart >= 500) {
    if (fpsStart > 0) document.body.dataset.fps = ((fpsFrames * 1000) / (now - fpsStart)).toFixed(0);
    fpsFrames = 0;
    fpsStart = now;
  }
  requestAnimationFrame(onFrame);
}

mountEditor();
if (view === 'editor') (document.querySelector('#editor') as HTMLElement).hidden = false;
const cam = params.get('cam') ?? (view === 'pov' ? 'pov' : view === 'chase' ? 'chase' : '');
if (cam === 'pov' || cam === 'chase' || cam === 'topdown') {
  renderer.setCamera(cam);
  left.dataset.camera = cam;
}
const quality = params.get('quality') === 'high' ? 'high' : 'low';
left.dataset.quality = quality;
const carsSlider = left.querySelector<HTMLInputElement>('[data-field="cars"]');
if (carsSlider && params.has('cars')) carsSlider.value = String(Math.min(64, Math.max(4, Number(params.get('cars')))));

// Diagnostics: ?fx=noaa,noshadow,unlit,lit,px1.2 edits the chosen preset so each feature's cost can
// be measured on its own. Not used by the app or the eval.
for (const flag of (params.get('fx') ?? '').split(',')) {
  const q = QUALITY[quality];
  if (flag === 'noaa') q.antialias = false;
  if (flag === 'aa') q.antialias = true;
  if (flag === 'noshadow') q.shadows = false;
  if (flag === 'basic') q.shadowType = THREE.BasicShadowMap;
  if (flag.startsWith('sm')) q.shadowMapSize = Number(flag.slice(2));
  if (flag.startsWith('every')) q.shadowEvery = Number(flag.slice(5));
  if (flag === 'unlit') q.lit = false;
  if (flag === 'lit') q.lit = true;
  if (flag.startsWith('px')) q.maxPixels = Number(flag.slice(2)) * 1_000_000;
}
await renderer.init(quality, { webgpu: params.get('webgpu') === '1' });
renderer.loadTrack(track);
renderer.resize();
placePip();
renderer.pushHero(blankHero());

window.__drift = {
  viz: renderer,
  frameStats: () => ({ ...frameStats.summary(), scene: renderer.counts() }),
  resetFrameStats: () => frameStats.reset(),
  metricsCount: () => metricsCount,
  boardRows: () => board.rows.length,
  boardGens: () => board.rows.map((r) => r.gen),
  tape: (gen) => {
    const t = tapes.get(gen);
    return t ? Array.from(t) : null;
  },
  probeReplay: (gen) => {
    const tape = tapes.get(gen);
    if (!tape) return null;
    const p = new ReplayPlayer();
    if (!p.start(gen, tape)) return null;
    const out: number[] = [];
    const live = renderer.replayDrawn();
    for (let pose = p.advance(1 / 30); pose; pose = p.advance(1 / 30)) {
      renderer.setReplay(pose);
      const d = renderer.replayDrawn()!;
      out.push(d.x, d.y, d.z, d.yaw);
    }
    renderer.setReplay(live ? { x: live.x, y: live.y, z: live.z, yaw: live.yaw } : null);
    return out;
  },
  replayGen: () => replayer.gen,
};

if (view === 'ghosts' || view === 'pov' || view === 'chase') {
  stream = syntheticStream();
  controls.setStatus('ghosts', 'racing');
  if (params.get('train') === '1') startWorker();
} else if (params.get('quiet') === '1') controls.setStatus(view === 'editor' ? 'editor' : 'idle', 'editor');
else if (params.get('inline') === '1') void startInline();
else startWorker();
requestAnimationFrame(onFrame);
