/**
 * Control experiment: PPO with no imitation at all, on stage-0 tracks, for a fixed number of PPO updates.
 * Scores held-out stage-0 clean laps (a lap with no wall contact) once, at the end, so the imitation/PPO
 * split is measured, not asserted. Plain completion is logged too, but a car with the steering centred
 * bounces off the walls round every held-out stage-0 track, so completion alone shows nothing.
 * The pre-registration is NOTES/ppo-only-prereg.md; read it before changing anything here.
 *
 *   npx vite-node scripts/rl-ppo-only.ts -- --seed=1 [--updates=240] [--mode=pool] [--lr=3e-4] [--n=16] [--h=200]
 *       [--probe-every=40] [--wall-cap=4500] [--out=path.jsonl]
 *
 * Modes:
 *   pool      (default) Episodes run to termination or the step cap across updates; each new episode draws
 *             a track from the 24 training tracks. No DR, native surfaces (the stage-0 profile).
 *   scenario  The script as first written: every update resets all envs to fresh stage-0 scenarios, so no
 *             training episode lasts longer than one 200-step rollout (4 s of driving, about a quarter lap).
 *
 * The curriculum is pinned to stage 0 in both modes. Training never sees seeds 101-116.
 */
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { Trainer } from '../src/rl/trainer';
import { ScenarioSet } from '../src/rl/scenarios';
import { DEFAULT_TRAIN_SEEDS, STAGE0_HELD_OUT, assertTrainSeeds } from '../src/rl/seeds';
import { stage0Completion } from '../src/eval/stage0';
import { pct } from '../src/eval/harness';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;
const seed = Number(arg('seed', '1'));
const updates = Number(arg('updates', '240'));
const mode = arg('mode', 'pool');
const lr = Number(arg('lr', '3e-4'));
const n = Number(arg('n', '16'));
const h = Number(arg('h', '200'));
const probeEvery = Number(arg('probe-every', '40'));
const wallCap = Number(arg('wall-cap', '4500'));
const out = arg('out', '');
if (mode !== 'pool' && mode !== 'scenario') throw new Error(`unknown --mode=${mode}`);

const cpu = () => {
  const u = process.cpuUsage();
  return (u.user + u.system) / 1e6;
};
const wall0 = performance.now();
const wall = () => (performance.now() - wall0) / 1000;
const log = (rec: Record<string, unknown>) => {
  if (out) appendFileSync(out, JSON.stringify(rec) + '\n');
};
if (out) {
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, '');
}

assertTrainSeeds(DEFAULT_TRAIN_SEEDS);
const trainer = new Trainer({ n, horizon: h, epochs: 2, minibatch: 256, seed, lr });
const set = new ScenarioSet(DEFAULT_TRAIN_SEEDS);
await trainer.init(set.tracks(0));
if (mode === 'scenario') trainer.setScenarios(set);
trainer.curriculum.stage = 0;
// Stage 0 is the whole study: never move the pool to stage 1 tracks, even if training laps well.
trainer.curriculum.maybePromote = () => false;

const heldOut = () => {
  const t0 = cpu();
  const r = stage0Completion((o, k) => trainer.policy.actMean(o, k));
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
  return {
    meanCleanFraction: mean(r.cleanFractions),
    cleanFractions: r.cleanFractions,
    cleanLaps: r.cleanLaps,
    cleanRate: r.total ? r.cleanLaps / r.total : 0,
    completed: r.completed,
    total: r.total,
    rate: r.rate,
    meanLapFraction: mean(r.lapFractions),
    lapFractions: r.lapFractions,
    contactSteps: r.contactSteps,
    lapTimes: r.lapTimes,
    evalCpu: cpu() - t0,
  };
};

log({ kind: 'config', seed, updates, mode, lr, n, horizon: h, epochs: 2, minibatch: 256, probeEvery, wallCap, trainSeeds: DEFAULT_TRAIN_SEEDS, heldOut: STAGE0_HELD_OUT });
// Reference, not a policy under test: steering centred, throttle 0.5. It shows how far a car gets on the
// held-out tracks before its first wall contact without steering at all.
{
  const r = stage0Completion((_o, k) => {
    const a = new Float32Array(k * 2);
    for (let i = 0; i < k; i++) a[i * 2 + 1] = 0.5;
    return a;
  });
  const meanClean = r.cleanFractions.reduce((a, b) => a + b, 0) / r.cleanFractions.length;
  log({ kind: 'reference', policy: 'steer 0, throttle 0.5', cleanLaps: r.cleanLaps, completed: r.completed, total: r.total, meanCleanFraction: meanClean });
}
const untrained = heldOut();
log({ kind: 'untrained', update: 0, ...untrained, weightHash: trainer.weightHash() });
const line = (r: ReturnType<typeof heldOut>) =>
  `clean ${r.cleanLaps}/${r.total} (${pct(r.cleanRate)}), clean lap fraction ${r.meanCleanFraction.toFixed(3)}, completed ${r.completed}/${r.total}, lap fraction ${r.meanLapFraction.toFixed(3)}`;
console.log(`seed ${seed} untrained: held-out stage 0 ${line(untrained)}`);

let trainCpu = 0;
let done = 0;
let stoppedEarly = false;
for (let k = 1; k <= updates; k++) {
  if (wall() > wallCap) {
    stoppedEarly = true;
    break;
  }
  const c0 = cpu();
  const stat = await trainer.update();
  trainCpu += cpu() - c0;
  done = k;
  log({ kind: 'update', update: k, trainCpu, wall: wall(), ...stat, bestLapSoFar: trainer.bestLap });
  if (probeEvery > 0 && k % probeEvery === 0 && k < updates) {
    // Informational only: not used for the pass line, early stopping or picking weights.
    const p = heldOut();
    log({ kind: 'probe', update: k, informational: true, ...p });
    console.log(`seed ${seed} update ${k}: train cpu ${trainCpu.toFixed(0)} s, wall ${wall().toFixed(0)} s, probe (informational) ${line(p)}`);
  }
}

const final = heldOut();
const weightHash = trainer.weightHash();
log({ kind: 'final', update: done, stoppedEarly, trainCpu, wall: wall(), envSteps: done * n * h, bestTrainLap: trainer.bestLap, weightHash, ...final });
console.log(
  `seed ${seed} final after ${done} updates (${done * n * h} env steps, train cpu ${trainCpu.toFixed(0)} s, wall ${wall().toFixed(0)} s${stoppedEarly ? ', STOPPED AT WALL CAP' : ''}): ` +
    `held-out stage 0 ${line(final)} (untrained clean lap fraction ${untrained.meanCleanFraction.toFixed(3)}); ` +
    `best training lap ${trainer.bestLap ?? 'none'}; weights ${weightHash}`,
);
