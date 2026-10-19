import type { Trainer } from '../rl/trainer';
import { BROWSER_TUNE, configureFinetune, trainRecipe } from '../rl/recipe';
import { generateTrack } from '../track/generate';
import { buildTrackFromPoints, validateTrack } from '../track/build';
import { handlePing } from '../shared/messages';
import { forkCheckpoint, loadCheckpoint, saveCheckpoint, type Checkpoint } from '../rl/checkpoint';
import { quantizePoints } from '../editor/share';
import { FineTune, type FineTuneReport } from '../rl/finetune';
import type { TrackSpec } from '../shared/types';

interface WorkerScope {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage(data: unknown): void;
}

const scope = globalThis as unknown as WorkerScope;

let trainer: Trainer | null = null;
let paused = true;
let busy = false;
let baseline = false;
let procedural: TrackSpec[] = [];
/** The hot-swap in progress, if any. While it runs, each loop turn is one `FineTune.step()`. */
let fine: FineTune | null = null;
let lastAdaptPost = 0;
let lastUserSave = 0;

function post(data: unknown): void {
  scope.postMessage(data);
}

/** Messages that arrive while the learner is still starting are applied once it is ready. */
const queued: Record<string, unknown>[] = [];
const DEFERRED = new Set(['setTrackPool', 'setStage', 'setDR', 'checkpointSave', 'checkpointLoad']);

function weightsOf(t: Trainer): number[][] {
  return t.policy.getWeights().map((b) => Array.from(new Float32Array(b)));
}

/** Checkpoint persistence is best effort: IndexedDB can be blocked (private windows), and training must not depend on it. */
async function persist(cp: Checkpoint, forkTo?: string): Promise<boolean> {
  try {
    await saveCheckpoint(cp);
    if (forkTo) await forkCheckpoint(cp.name, forkTo);
    return true;
  } catch {
    post({ type: 'status', text: 'checkpoint storage unavailable' });
    return false;
  }
}

/** SPEC: the ghost stream is 30 Hz. The main thread acks every frame it draws; a main thread that
 *  is two frames behind (software GL, a busy tab) gets nothing more until it catches up, so the
 *  message queue cannot grow without bound and `ids` stays current. */
const GHOST_MIN_MS = 33;
let lastGhost = 0;
let unacked = 0;

function publish(): void {
  if (!trainer) return;
  const now = performance.now();
  if (now - lastGhost < GHOST_MIN_MS) return;
  if (unacked >= 2 && now - lastGhost < 500) return;
  lastGhost = now;
  unacked++;
  const poses = new Float32Array(trainer.env.poses());
  post({
    type: 'ghosts',
    poses,
    n: trainer.env.n,
    gen: trainer.gen,
    trackId: trainer.env.trackId(0),
    ids: trainer.env.trackIds(),
    hero: trainer.env.snapshot(0),
    bestLap: trainer.bestLap,
  });
}

function postAdapt(rep: FineTuneReport, force = false): void {
  const now = performance.now();
  if (!force && now - lastAdaptPost < 150) return;
  lastAdaptPost = now;
  post({ type: 'adapt', ...rep });
}

async function maybeSaveUser(rep: FineTuneReport): Promise<void> {
  if (!trainer || !fine || rep.phase !== 'tune') return;
  const now = performance.now();
  if (now - lastUserSave < 5000) return;
  lastUserSave = now;
  const weights = trainer.policy.getWeights().map((b) => Array.from(new Float32Array(b)));
  await persist({ name: fine.user.id, gen: trainer.gen, stage: trainer.curriculum.stage, weights });
}

async function loop(): Promise<void> {
  if (!trainer || paused || busy) return;
  busy = true;
  try {
    if (baseline) {
      trainer.env.step(trainer.env.baselineActions());
    } else if (fine && fine.phase !== 'done') {
      const gen0 = trainer.gen;
      const rep = await fine.step();
      if (trainer.gen !== gen0) {
        const stat = trainer.history[trainer.history.length - 1];
        const tape = trainer.replays.get(trainer.gen);
        if (stat) post({ type: 'metrics', stat, tape: tape ? Array.from(tape) : [] });
      }
      postAdapt(rep);
      await maybeSaveUser(rep);
    } else {
      const stat = await trainer.update();
      const tape = trainer.replays.get(trainer.gen);
      post({ type: 'metrics', stat, tape: tape ? Array.from(tape) : [] });
    }
    publish();
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    paused = true;
  }
  busy = false;
  if (!paused) setTimeout(() => void loop(), 0);
}

/** Posts `trackAck` once every env is on the new track (the hot-swap latency event). */
async function announceSwap(trackId: string, hashBefore: string): Promise<void> {
  const mine = fine;
  while (mine && fine === mine && mine.phase === 'drain') await new Promise((r) => setTimeout(r, 20));
  if (!mine || fine !== mine || !trainer) return;
  const rep = mine.report();
  post({
    type: 'trackAck',
    trackId,
    envCount: rep.envsOnUserAtSwap ?? 0,
    envTotal: rep.envTotal,
    latencyMs: rep.latencyMs,
    hash: rep.weightHashAtDrain,
    hashBefore,
  });
}

async function handle(msg: Record<string, unknown>): Promise<void> {
  if (msg.type === 'init') {
    const n = Math.max(1, Math.min(64, Math.floor(Number(msg.n ?? 8)) || 8));
    const stage = Number(msg.stage ?? 0);
    const seeds = (msg.seeds as number[]) ?? [1, 2, 3, 4];
    procedural = seeds.map((seed) => generateTrack({ seed, stage }));
    post({ type: 'status', text: 'imitating pure pursuit' });
    // Same recipe function the eval measures (src/rl/recipe.ts), with the short browser schedule:
    // stage 0 only, no validation-gated PPO phase. Later stages train by PPO from this start.
    const trained = await trainRecipe({
      n,
      maxStage: stage,
      ppoUpdates: 0,
      validate: false,
      tune: BROWSER_TUNE,
      onStatus: (text) => post({ type: 'status', text }),
    });
    trainer = trained.trainer;
    configureFinetune(trainer);
    // Live view: short rollouts, and cars keep their episodes between updates.
    trainer.horizon = 32;
    trainer.freshEpisodesPerUpdate = false;
    await persist({ name: 'base', gen: trainer.gen, stage: trainer.curriculum.stage, weights: weightsOf(trainer) });
    post({ type: 'ready', hash: trainer.weightHash(), trackId: trainer.env.trackId(0) });
    post({ type: 'status', text: 'racing' });
    paused = false;
    publish();
    void loop();
    while (queued.length) await handle(queued.shift()!);
    return;
  }
  if (msg.type === 'ghostAck') {
    unacked = Math.max(0, unacked - 1);
    return;
  }
  if (!trainer) {
    if (DEFERRED.has(String(msg.type))) queued.push(msg);
    return;
  }
  if (msg.type === 'pause') {
    paused = Boolean(msg.paused);
    if (!paused) void loop();
  }
  if (msg.type === 'baseline') baseline = Boolean(msg.on);
  if (msg.type === 'setStage') trainer.setStage(Number(msg.stage ?? 0));
  if (msg.type === 'setDR') {
    trainer.grip = Number(msg.grip ?? 1);
    trainer.massScale = Number(msg.mass ?? 1);
    trainer.dr = (msg.level as 'off' | 'low' | 'high') ?? 'off';
  }
  if (msg.type === 'replayReq') {
    const gen = Number(msg.gen ?? 0);
    const tape = trainer.replays.get(gen);
    post({ type: 'replayRes', gen, poses: tape ? Array.from(tape) : [], n: tape ? tape.length / 7 : 0 });
  }
  if (msg.type === 'setTrackPool') {
    // Snap to mm first so the track id survives a round trip through the URL hash.
    const points = quantizePoints((msg.points as number[]) ?? []);
    const halfWidth = Number(msg.halfWidth ?? 1);
    const mu = Number(msg.mu ?? 1);
    const check = validateTrack(points, halfWidth);
    if (!check.ok) {
      post({ type: 'error', message: `track rejected: ${[...new Set(check.issues.map((i) => i.reason))].join(', ')}` });
      return;
    }
    const t0 = performance.now();
    const user = buildTrackFromPoints(points, { halfWidth, mu, source: 'user' });
    fine?.stop();
    const hashBefore = trainer.weightHash();
    // Snapshot the pre-swap model now; it is saved as 'base' and forked to user-<hash> below.
    const weights = trainer.policy.getWeights().map((b) => Array.from(new Float32Array(b)));
    const gen = trainer.gen;
    const stage = trainer.curriculum.stage;
    // The swap itself: nothing resets until each env finishes its episode (FineTune drains them).
    fine = new FineTune(trainer, { user, procedural });
    fine.start();
    lastUserSave = performance.now();
    post({ type: 'swapStart', trackId: user.id, hashBefore, ms: performance.now() - t0 });
    postAdapt(fine.report(), true);
    paused = false;
    baseline = false;
    void loop();
    void announceSwap(user.id, hashBefore);
    await persist({ name: 'base', gen, stage, weights }, user.id);
    return;
  }
  if (msg.type === 'checkpointSave') {
    const name = String(msg.name ?? 'manual');
    const ok = await persist({ name, gen: trainer.gen, stage: trainer.curriculum.stage, weights: weightsOf(trainer) });
    post(ok ? { type: 'checkpointSaved', name } : { type: 'error', message: 'could not save checkpoint' });
  }
  if (msg.type === 'checkpointLoad') {
    const name = String(msg.name ?? 'manual');
    try {
      const cp = await loadCheckpoint(name);
      if (!cp) {
        post({ type: 'status', text: `no checkpoint ${name}` });
        return;
      }
      trainer.policy.setWeights(cp.weights.map((w) => Float32Array.from(w).buffer));
      post({ type: 'status', text: `loaded ${name}` });
    } catch (err) {
      post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  }
}

scope.onmessage = (ev: MessageEvent) => {
  const msg = ev.data as Record<string, unknown>;
  const pong = handlePing({ type: String(msg.type ?? ''), t: Number(msg.t ?? 0) });
  if (pong) {
    post(pong);
    return;
  }
  handle(msg).catch((err: unknown) => {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  });
};
