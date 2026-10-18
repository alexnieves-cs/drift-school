import type { Trainer } from '../rl/trainer';
import { BROWSER_TUNE, configureFinetune, trainRecipe } from '../rl/recipe';
import { generateTrack } from '../track/generate';
import { buildTrackFromPoints } from '../track/build';
import { handlePing } from '../shared/messages';
import { forkCheckpoint, saveCheckpoint } from '../rl/checkpoint';
import { encodeTrackHash } from '../editor/share';
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

async function loop(): Promise<void> {
  if (!trainer || paused || busy) return;
  busy = true;
  try {
    if (baseline) {
      trainer.env.step(trainer.env.baselineActions());
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

async function handle(msg: Record<string, unknown>): Promise<void> {
  if (msg.type === 'init') {
    const n = Number(msg.n ?? 8);
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
    const weights = trainer.policy.getWeights().map((b) => Array.from(new Float32Array(b)));
    await saveCheckpoint({ name: 'base', gen: trainer.gen, stage: trainer.curriculum.stage, weights });
    post({ type: 'ready', hash: trainer.weightHash(), trackId: trainer.env.trackId(0) });
    post({ type: 'status', text: 'racing' });
    paused = false;
    publish();
    void loop();
    return;
  }
  if (!trainer) return;
  if (msg.type === 'pause') {
    paused = Boolean(msg.paused);
    if (!paused) void loop();
  }
  if (msg.type === 'baseline') baseline = Boolean(msg.on);
  if (msg.type === 'setStage') trainer.curriculum.stage = Number(msg.stage ?? 0);
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
    const points = Float32Array.from((msg.points as number[]) ?? []);
    const halfWidth = Number(msg.halfWidth ?? 1);
    const mu = Number(msg.mu ?? 1);
    const user = buildTrackFromPoints(points, { halfWidth, mu, source: 'user', id: 'user-track' });
    const before = trainer.weightHash();
    trainer.queueTrackPool([user, ...procedural], Boolean(msg.fineTune));
    trainer.forceSwap();
    const hash = encodeTrackHash(points, halfWidth, mu).slice(0, 16);
    await forkCheckpoint('base', `user-${hash}`);
    post({
      type: 'trackAck',
      trackId: trainer.env.trackId(0),
      envCount: trainer.env.trackIds().filter((id) => id === user.id).length || trainer.env.n,
      hash: trainer.weightHash(),
      hashBefore: before,
    });
    publish();
  }
  if (msg.type === 'checkpointSave') {
    const name = String(msg.name ?? 'manual');
    const weights = trainer.policy.getWeights().map((b) => Array.from(new Float32Array(b)));
    await saveCheckpoint({ name, gen: trainer.gen, stage: trainer.curriculum.stage, weights });
    post({ type: 'checkpointSaved', name });
  }
}

scope.onmessage = (ev: MessageEvent) => {
  const msg = ev.data as Record<string, unknown>;
  const pong = handlePing({ type: String(msg.type ?? ''), t: Number(msg.t ?? 0) });
  if (pong) {
    post(pong);
    return;
  }
  void handle(msg);
};
