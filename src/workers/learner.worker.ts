import { Trainer } from '../rl/trainer';
import { generateTrack } from '../track/generate';
import { buildTrackFromPoints, validateTrack } from '../track/build';
import { handlePing } from '../shared/messages';
import { loadCheckpoint, saveCheckpoint } from '../rl/checkpoint';
import type { TrackSpec } from '../shared/types';

interface WorkerScope {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage(data: unknown): void;
}

const scope = globalThis as unknown as WorkerScope;

/** Minimum gap between pose frames posted while a rollout runs (about 30 Hz). */
const FRAME_MS = 33;

let trainer: Trainer | null = null;
let paused = true;
let busy = false;
let baseline = false;
let procedural: TrackSpec[] = [];
let lastFrame = 0;
/** Messages that change the trainer wait until the current update finishes, so a rollout never straddles a swap. */
const queued: Record<string, unknown>[] = [];
const DEFERRED = new Set(['setTrackPool', 'setStage', 'setDR', 'checkpointSave', 'checkpointLoad']);

function post(data: unknown): void {
  scope.postMessage(data);
}

function publish(): void {
  if (!trainer) return;
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

function weightsOf(t: Trainer): number[][] {
  return t.policy.getWeights().map((b) => Array.from(new Float32Array(b)));
}

/** Checkpoint persistence is best effort: IndexedDB can be blocked (private windows), and training must not depend on it. */
async function persist(name: string): Promise<boolean> {
  if (!trainer) return false;
  try {
    await saveCheckpoint({ name, gen: trainer.gen, stage: trainer.curriculum.stage, weights: weightsOf(trainer) });
    return true;
  } catch {
    post({ type: 'status', text: 'checkpoint storage unavailable' });
    return false;
  }
}

async function loop(): Promise<void> {
  if (!trainer || paused || busy) return;
  busy = true;
  try {
    if (baseline) {
      trainer.env.step(trainer.env.baselineActions());
      publish();
    } else {
      const stat = await trainer.update();
      const tape = trainer.replays.get(trainer.gen);
      post({ type: 'metrics', stat, tape: tape ? Array.from(tape) : [], adapt: trainer.adaptStats() });
      publish();
    }
    await drain();
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    paused = true;
  }
  busy = false;
  if (!paused) setTimeout(() => void loop(), 0);
}

async function drain(): Promise<void> {
  while (queued.length) await apply(queued.shift()!);
}

async function apply(msg: Record<string, unknown>): Promise<void> {
  if (!trainer) return;
  if (msg.type === 'setStage') trainer.setStage(Number(msg.stage ?? 0));
  if (msg.type === 'setDR') {
    trainer.grip = Number(msg.grip ?? 1);
    trainer.massScale = Number(msg.mass ?? 1);
    trainer.dr = (msg.level as 'off' | 'low' | 'high') ?? 'off';
  }
  if (msg.type === 'setTrackPool') {
    const points = Float32Array.from((msg.points as number[]) ?? []);
    const halfWidth = Number(msg.halfWidth ?? 1);
    const mu = Number(msg.mu ?? 1);
    const check = validateTrack(points, halfWidth);
    if (!check.ok) {
      post({ type: 'error', message: `track rejected: ${[...new Set(check.issues.map((i) => i.reason))].join(', ')}` });
      return;
    }
    const user = buildTrackFromPoints(points, { halfWidth, mu, source: 'user' });
    const before = trainer.weightHash();
    // First fine-tune: keep what the base model had learned so far, then fork the live weights for this track.
    if (!trainer.mix.userId) await persist('base');
    trainer.queueTrackPool([user, ...procedural], Boolean(msg.fineTune));
    trainer.forceSwap();
    await persist(user.id);
    post({
      type: 'trackAck',
      trackId: user.id,
      envCount: trainer.env.trackIds().filter((id) => id === user.id).length,
      hash: trainer.weightHash(),
      hashBefore: before,
    });
    publish();
  }
  if (msg.type === 'checkpointSave') {
    const name = String(msg.name ?? 'manual');
    post(await persist(name) ? { type: 'checkpointSaved', name } : { type: 'error', message: 'could not save checkpoint' });
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

async function handle(msg: Record<string, unknown>): Promise<void> {
  if (msg.type === 'init') {
    const n = Math.max(1, Math.min(64, Math.floor(Number(msg.n ?? 8)) || 8));
    const stage = Number(msg.stage ?? 0);
    const seeds = (msg.seeds as number[]) ?? [1, 2, 3, 4];
    procedural = seeds.map((seed) => generateTrack({ seed, stage }));
    trainer = new Trainer({ n, horizon: 64, epochs: 3, minibatch: 128 });
    trainer.yieldEvery = 8;
    trainer.onStep = () => {
      const now = performance.now();
      if (now - lastFrame >= FRAME_MS) {
        lastFrame = now;
        publish();
      }
    };
    await trainer.init(procedural);
    post({ type: 'status', text: 'imitating pure pursuit' });
    await trainer.imitate(4);
    await persist('base');
    post({ type: 'ready', hash: trainer.weightHash(), trackId: trainer.env.trackId(0) });
    post({ type: 'status', text: 'racing' });
    paused = false;
    publish();
    await drain();
    void loop();
    return;
  }
  if (!trainer) {
    // A swap or setting that arrives while the learner is still starting is applied once it is ready.
    if (DEFERRED.has(String(msg.type))) queued.push(msg);
    return;
  }
  if (msg.type === 'pause') {
    paused = Boolean(msg.paused);
    if (!paused) void loop();
  }
  if (msg.type === 'baseline') baseline = Boolean(msg.on);
  if (msg.type === 'replayReq') {
    const gen = Number(msg.gen ?? 0);
    const tape = trainer.replays.get(gen);
    post({ type: 'replayRes', gen, poses: tape ? Array.from(tape) : [], n: tape ? tape.length / 7 : 0 });
  }
  if (DEFERRED.has(String(msg.type))) {
    if (busy) queued.push(msg);
    else await apply(msg);
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
