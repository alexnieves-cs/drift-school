import { ThreeRenderer, type HeroState } from './viz/renderer';
import { Hud } from './viz/cameras/hud';
import { mountControls } from './ui/controls';
import { MetricCharts } from './ui/charts';
import { Leaderboard } from './ui/leaderboard';
import { TelemetryPanel } from './ui/telemetry';
import { Heatmap } from './ui/heatmaps';
import { generateTrack } from './track/generate';
import { sampleAtS } from './track/sample';
import { buildTrackFromPoints, validateTrack } from './track/build';
import type { AdaptStats } from './rl/trainer';
import { encodeTrackHash, decodeTrackHash } from './editor/share';
import { toFloat } from './editor/stroke';
import { TrackEditor } from './editor/editor';
import { openLearner } from './workers/pool';
import { saveCheckpoint, loadCheckpoint } from './rl/checkpoint';
import type { CameraMode, GenStat, TrackSpec, Validation } from './shared/types';
import type { CarSnap } from './sim/vecEnv';

const params = new URLSearchParams(location.search);
const view = params.get('view') ?? 'race';
/** `?cars=` as an integer in [1, 64]; anything else falls back to `fallback`. */
function carsParam(fallback: number): number {
  const n = Math.floor(Number(params.get('cars')));
  return Number.isFinite(n) && n >= 1 ? Math.min(64, n) : fallback;
}

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
let replay: Float32Array | null = null;
let replayCursor = 0;
let track: TrackSpec = generateTrack({ seed: 1, stage: 0 });
let bestLap: number | null = null;
let frames = 0;
let fpsStamp = 0;
let lastHero: HeroState | null = null;
let crashLatch = false;

const board = new Leaderboard(document.querySelector('#board') as HTMLElement, (gen) => {
  const tape = tapes.get(gen);
  if (!tape) return;
  replay = tape;
  replayCursor = 0;
  controls.setStatus(`replay gen ${gen}`, 'replay');
});

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

let worker: Worker | null = null;
let inline: import('./rl/trainer').Trainer | null = null;
const inlineMode = params.get('inline') === '1' || view === 'ghosts';

function send(msg: Record<string, unknown>): void {
  if (worker) worker.postMessage(msg);
  else applyInline(msg);
}

function mapPts(t: TrackSpec): { x: number; z: number }[] {
  const step = Math.max(1, Math.floor(t.samples.length / 80));
  const pts = [];
  for (let i = 0; i < t.samples.length; i += step) pts.push(t.samples[i]!);
  return pts;
}

function heroFrom(snap: CarSnap | null, fallback: HeroState): HeroState {
  if (!snap) return fallback;
  const latG = (snap.yawRate * snap.speed) / 9.81;
  const longG = snap.vx / 9.81;
  telemetry.update({
    speed: snap.speed,
    steer: snap.steer,
    throttle: snap.throttle,
    slip: snap.slip,
    load: snap.load,
    ax: longG,
    ay: latG,
  });
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

function show(poses: Float32Array, n: number, hero: HeroState): void {
  let drawn = hero;
  if (replay && replay.length >= 7) {
    const framesN = Math.floor(replay.length / 7);
    const base = (replayCursor % framesN) * 7;
    replayCursor++;
    const qy = replay[base + 5] ?? 0;
    const qw = replay[base + 3] ?? 1;
    drawn = {
      ...hero,
      x: replay[base] ?? hero.x,
      y: replay[base + 1] ?? hero.y,
      z: replay[base + 2] ?? hero.z,
      yaw: 2 * Math.atan2(qy, qw),
    };
    if (replayCursor > framesN + 5) replay = null;
  }
  lastHero = drawn;
  renderer.updateCars(poses, n, 0);
  renderer.frame(n, 0, drawn);
  hud.update(
    {
      speed: drawn.speed,
      lap: drawn.lap ?? 0,
      lapTime: drawn.lapTime ?? 0,
      bestLap,
      steer: drawn.steer ?? 0,
      throttle: drawn.throttle ?? 0,
      slipRear: drawn.slipRear ?? 0,
    },
    mapPts(track),
    drawn,
  );
  const now = performance.now();
  if (frames === 0 && fpsStamp === 0) fpsStamp = now;
  frames++;
  if (now - fpsStamp > 500) {
    document.body.dataset.fps = ((frames * 1000) / (now - fpsStamp)).toFixed(0);
    frames = 0;
    fpsStamp = now;
  }
}

function adaptText(a: AdaptStats): string {
  const pct = (x: number | null) => (x === null ? 'n/a' : `${(x * 100).toFixed(0)}%`);
  const lap = a.bestLap === null ? 'n/a' : `${a.bestLap.toFixed(1)} s`;
  return `drawn track: first 64 eps ${pct(a.first64)} · now ${pct(a.completion)} · best ${lap} · procedural ${pct(a.procBefore)} → ${pct(a.procNow)} · ${a.updatesSinceSwap} updates since swap`;
}

function onMetrics(stat: GenStat, tape?: number[], adaptStats?: AdaptStats | null): void {
  if (tape?.length) tapes.set(stat.gen, Float32Array.from(tape));
  if (adaptStats) adapt.textContent = adaptText(adaptStats);
  if (stat.bestLap != null) bestLap = bestLap == null ? stat.bestLap : Math.min(bestLap, stat.bestLap);
  board.add(stat);
  charts.push(stat);
  adapt.dataset.adapt = `gen ${stat.gen} completion ${(stat.completion * 100).toFixed(0)}%`;
}

function applyInline(msg: Record<string, unknown>): void {
  if (!inline) return;
  if (msg.type === 'pause') return;
  if (msg.type === 'baseline') inlineModeBaseline = Boolean(msg.on);
  if (msg.type === 'setStage') inline.setStage(Number(msg.stage ?? 0));
  if (msg.type === 'setDR') {
    inline.grip = Number(msg.grip ?? 1);
    inline.massScale = Number(msg.mass ?? 1);
    inline.dr = (msg.level as 'off' | 'low' | 'high') ?? 'off';
  }
  if (msg.type === 'setTrackPool') void swapInline(msg);
}

let inlineProcedural: TrackSpec[] = [];
let inlineModeBaseline = false;
let inlinePaused = false;

async function swapInline(msg: Record<string, unknown>): Promise<void> {
  if (!inline) return;
  const points = Float32Array.from((msg.points as number[]) ?? []);
  const halfWidth = Number(msg.halfWidth ?? 1);
  const mu = Number(msg.mu ?? 1);
  if (!validateTrack(points, halfWidth).ok) {
    controls.setStatus('track is not valid', 'bad');
    return;
  }
  const user = buildTrackFromPoints(points, { halfWidth, mu, source: 'user' });
  const before = inline.weightHash();
  inline.queueTrackPool([user, ...inlineProcedural], true);
  inline.forceSwap();
  track = user;
  renderer.loadTrack(track);
  heat.reset();
  history.replaceState(null, '', `#${encodeTrackHash(points, halfWidth, mu)}`);
  adapt.dataset.adapt = `swapped ${user.id} weights ${before === inline.weightHash() ? 'unchanged' : 'CHANGED'}`;
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
  if (worker) {
    worker.postMessage({ type: 'checkpointLoad', name });
    return;
  }
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

let editorState: {
  track: TrackSpec | null;
  points: { x: number; z: number }[];
  halfWidth: number;
  mu: number;
  validation: Validation;
} = { track: null, points: [], halfWidth: 1, mu: 1.05, validation: { ok: false, issues: [] } };

function mountEditor(): void {
  const stroke = document.querySelector('#stroke') as HTMLCanvasElement;
  let editor: TrackEditor | null = null;
  const fit = () => {
    stroke.width = window.innerWidth;
    stroke.height = window.innerHeight;
    // Resizing clears the canvas, so tell the editor the new size and redraw the stroke.
    if (editor) {
      editor.setSize(stroke.width, stroke.height, 0.04);
      editor.commit();
    }
  };
  fit();
  editor = new TrackEditor(stroke, {
    width: stroke.width,
    height: stroke.height,
    scale: 0.04,
    onChange: (state) => {
      editorState = state;
      const reasons = [...new Set(state.validation.issues.map((i) => i.reason))];
      adapt.textContent = state.validation.ok ? `valid ${state.ms.toFixed(1)} ms` : reasons.join(', ');
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

async function trainOnDrawn(): Promise<void> {
  const built = editorState.track;
  if (!built || !editorState.validation.ok) {
    const reasons = [...new Set(editorState.validation.issues.map((i) => i.reason))];
    controls.setStatus(reasons.length ? `track is not valid: ${reasons.join(', ')}` : 'draw a track first', 'bad');
    return;
  }
  // Send the editor's own width and grip. The built samples carry the corner-widened value, which would widen again on every swap.
  const { halfWidth, mu } = editorState;
  const points = Array.from(toFloat(editorState.points));
  send({ type: 'setTrackPool', points, halfWidth, mu, fineTune: true });
  track = built;
  renderer.loadTrack(track);
  heat.reset();
  history.replaceState(null, '', `#${encodeTrackHash(Float32Array.from(points), halfWidth, mu)}`);
  (document.querySelector('#editor') as HTMLElement).hidden = true;
  controls.setStatus('training on drawn track', 'adapt');
}

function ghostLoop(): void {
  const n = carsParam(64);
  const poses = new Float32Array(n * 7);
  let t = 0;
  const step = () => {
    t += 0.016;
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
    const yaw = Math.atan2(lead.tz, lead.tx);
    show(poses, n, {
      x: lead.x,
      y: 0.05,
      z: lead.z,
      yaw,
      pitch: 0,
      roll: 0,
      speed: 3.2,
      steer: Math.sin(t) * 0.2,
      throttle: 0.7,
      slipRear: 0.05,
      lap: Math.floor(t / 12),
      lapTime: t % 12,
      bestLap: 11.4,
    });
    requestAnimationFrame(step);
  };
  controls.setStatus('ghosts', 'racing');
  document.body.dataset.track = track.id;
  requestAnimationFrame(step);
}

async function startInline(): Promise<void> {
  const { Trainer } = await import('./rl/trainer');
  inlineProcedural = [1, 2, 3, 4].map((seed) => generateTrack({ seed, stage: 0 }));
  track = inlineProcedural[0]!;
  renderer.loadTrack(track);
  inline = new Trainer({ n: carsParam(8), horizon: 64, epochs: 3, minibatch: 128 });
  // Yield during rollouts so requestAnimationFrame can draw between steps instead of once per update.
  inline.yieldEvery = 8;
  controls.setStatus('imitating pure pursuit', 'imitating');
  await inline.init(inlineProcedural);
  await inline.imitate(4);
  controls.setStatus('racing', 'racing');
  document.body.dataset.track = inline.env.trackId(0);
  const render = () => {
    if (!inline) return;
    const snap = inline.env.snapshot(0);
    show(inline.env.poses(), inline.env.n, heroFrom(snap, lastHero ?? blankHero()));
    requestAnimationFrame(render);
  };
  requestAnimationFrame(render);
  const train = async () => {
    for (;;) {
      if (!inline) return;
      inlinePaused = controls.read().paused;
      if (!inlinePaused && !inlineModeBaseline) {
        const stat = await inline.update();
        const tape = inline.replays.get(inline.gen);
        onMetrics(stat, tape ? Array.from(tape) : undefined, inline.adaptStats());
      } else {
        if (inlineModeBaseline && !inlinePaused) inline.env.step(inline.env.baselineActions());
        await new Promise<void>((r) => setTimeout(r, 16));
      }
    }
  };
  void train();
}

function blankHero(): HeroState {
  const s = track.samples[track.startIndex] ?? track.samples[0]!;
  return { x: s.x, y: 0.05, z: s.z, yaw: Math.atan2(s.tz, s.tx), pitch: 0, roll: 0, speed: 0 };
}

function startWorker(): void {
  worker = openLearner();
  // Workers post pose frames at about 30 Hz; the page draws the latest one every animation frame.
  let latest: { poses: Float32Array; n: number; hero: CarSnap | null } | null = null;
  const render = () => {
    if (latest) show(latest.poses, latest.n, heroFrom(latest.hero, lastHero ?? blankHero()));
    requestAnimationFrame(render);
  };
  requestAnimationFrame(render);
  worker.onmessage = (ev: MessageEvent) => {
    const msg = ev.data as Record<string, unknown>;
    if (msg.type === 'status') controls.setStatus(String(msg.text), String(msg.text).includes('racing') ? 'racing' : 'live');
    if (msg.type === 'ghosts') {
      if (typeof msg.bestLap === 'number') bestLap = msg.bestLap;
      latest = { poses: msg.poses as Float32Array, n: Number(msg.n ?? 0), hero: (msg.hero as CarSnap) ?? null };
      document.body.dataset.track = String(msg.trackId ?? '');
    }
    if (msg.type === 'metrics') onMetrics(msg.stat as GenStat, msg.tape as number[] | undefined, msg.adapt as AdaptStats | null | undefined);
    if (msg.type === 'replayRes') {
      const poses = msg.poses as number[];
      if (poses?.length) {
        tapes.set(Number(msg.gen), Float32Array.from(poses));
        replay = tapes.get(Number(msg.gen)) ?? null;
        replayCursor = 0;
      }
    }
    if (msg.type === 'trackAck') {
      adapt.dataset.adapt = `track ${String(msg.trackId)} envs ${String(msg.envCount)} weights ${msg.hash === msg.hashBefore ? 'unchanged' : 'CHANGED'}`;
      document.body.dataset.track = String(msg.trackId);
      controls.setStatus(`training on drawn track (${String(msg.envCount)} envs)`, 'adapt');
    }
    if (msg.type === 'error') controls.setStatus(String(msg.message), 'bad');
  };
  worker.onerror = () => {
    worker?.terminate();
    worker = null;
    controls.setStatus('worker failed, inline', 'inline');
    void startInline();
  };
  worker.postMessage({ type: 'init', n: carsParam(8), seeds: [1, 2, 3, 4], stage: 0 });
}

window.addEventListener('keydown', (ev) => {
  const cams: Record<string, CameraMode> = { '1': 'topdown', '2': 'chase', '3': 'pov' };
  const mode = cams[ev.key];
  if (!mode) return;
  renderer.setCamera(mode);
  left.dataset.camera = mode;
});
window.addEventListener('resize', () => renderer.resize());

mountEditor();
if (view === 'editor') (document.querySelector('#editor') as HTMLElement).hidden = false;
const cam = params.get('cam') ?? (view === 'pov' ? 'pov' : view === 'chase' ? 'chase' : '');
if (cam === 'pov' || cam === 'chase' || cam === 'topdown') {
  renderer.setCamera(cam);
  left.dataset.camera = cam;
}

await renderer.init(params.get('quality') === 'high' ? 'high' : 'low');
renderer.loadTrack(track);
renderer.resize();
renderer.frame(0, -1, blankHero());

if (view === 'ghosts' || view === 'pov' || view === 'chase') ghostLoop();
else if (params.get('quiet') === '1') controls.setStatus(view === 'editor' ? 'editor' : 'idle', 'editor');
else if (inlineMode) void startInline();
else startWorker();
