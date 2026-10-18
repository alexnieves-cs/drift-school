import { ACT_DIM, OBS_DIM } from '../shared/types';
import { randn, type RNG } from '../shared/rng';
import type { GaussianPolicy } from './model';
import { FastImitator } from './fastfit';
import type { Trainer } from './trainer';

/**
 * DAgger-style imitation of the pure-pursuit baseline (src/track/baseline.ts).
 *
 * The expert is NOT learned: `VecEnv.baselineActions()` reads the true car pose and the track
 * spline, then labels the observation the learner is seeing. Data is collected on the states the
 * learner (or a mix of learner and expert) actually visits, so the policy sees its own mistakes and
 * their labelled recoveries, which a single pass over expert-only rollouts never gives it.
 * Everything a policy trained this way can do is bounded by the baseline's behaviour.
 */

export class Dataset {
  readonly cap: number;
  readonly obs: Float32Array;
  readonly act: Float32Array;
  size = 0;
  private head = 0;

  constructor(cap: number) {
    this.cap = cap;
    this.obs = new Float32Array(cap * OBS_DIM);
    this.act = new Float32Array(cap * ACT_DIM);
  }

  add(obs: Float32Array, act: Float32Array, n: number): void {
    for (let i = 0; i < n; i++) {
      this.obs.set(obs.subarray(i * OBS_DIM, (i + 1) * OBS_DIM), this.head * OBS_DIM);
      this.act.set(act.subarray(i * ACT_DIM, (i + 1) * ACT_DIM), this.head * ACT_DIM);
      this.head = (this.head + 1) % this.cap;
      if (this.size < this.cap) this.size++;
    }
  }
}

export interface CollectOpts {
  steps: number;
  /** Probability that an env is expert-driven during a chunk (1 = pure expert rollout). */
  beta: number;
  /** Std of Gaussian noise on executed actions in the noisy envs (DART-style perturbation). */
  actionNoise: number;
  /** Fraction of envs that execute noisy actions. */
  noisyFrac: number;
  /** Steps between re-drawing which envs the expert drives. */
  chunk: number;
  rng: RNG;
}

export interface CollectStats {
  samples: number;
  episodes: number;
  /** Lap completion over episodes that ended during the rollout (mixed expert/learner control). */
  completion: number;
}

const clamp1 = (x: number) => Math.max(-1, Math.min(1, x));

export function collect(trainer: Trainer, ds: Dataset, o: CollectOpts): CollectStats {
  const env = trainer.env;
  const n = env.n;
  trainer.resetEnvs();
  const obs = new Float32Array(env.obs);
  const expertOn = new Uint8Array(n);
  const noisy = new Uint8Array(n);
  for (let i = 0; i < n; i++) noisy[i] = o.rng.next() < o.noisyFrac ? 1 : 0;
  const exec = new Float32Array(n * ACT_DIM);
  let samples = 0;
  for (let t = 0; t < o.steps; t++) {
    if (t % o.chunk === 0) for (let i = 0; i < n; i++) expertOn[i] = o.rng.next() < o.beta ? 1 : 0;
    const expert = env.baselineActions();
    const learner = o.beta >= 1 ? expert : trainer.policy.actMean(obs, n);
    ds.add(obs, expert, n);
    samples += n;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < ACT_DIM; k++) {
        const idx = i * ACT_DIM + k;
        let a = expertOn[i] ? expert[idx]! : learner[idx]!;
        if (noisy[i] && o.actionNoise > 0) a = clamp1(a + randn(o.rng) * o.actionNoise);
        exec[idx] = a;
      }
    }
    const stepped = env.step(exec);
    obs.set(stepped.obs);
  }
  const eps = env.finished;
  const laps = eps.filter((e) => e.lap).length;
  return { samples, episodes: eps.length, completion: eps.length ? laps / eps.length : 0 };
}

export interface FitOpts {
  /** Gradient steps, each on a minibatch drawn uniformly (with replacement) from the aggregate. */
  steps: number;
  batch: number;
  lr0: number;
  lr1: number;
  rng: RNG;
}

const imitators = new WeakMap<GaussianPolicy, FastImitator>();

/**
 * Minibatch Adam regression of tanh(mean) onto the expert action; lr decays geometrically
 * lr0 -> lr1. Runs on the typed-array backprop in fastfit.ts and writes the result back to TF.
 */
export function fit(policy: GaussianPolicy, ds: Dataset, o: FitOpts): number {
  const n = ds.size;
  if (!n) return 0;
  let im = imitators.get(policy);
  if (!im) {
    im = new FastImitator();
    imitators.set(policy, im);
  }
  im.load(policy.getPiWeights());
  const bx = new Float32Array(o.batch * OBS_DIM);
  const by = new Float32Array(o.batch * ACT_DIM);
  let last = 0;
  for (let step = 0; step < o.steps; step++) {
    for (let k = 0; k < o.batch; k++) {
      const idx = o.rng.int(n);
      bx.set(ds.obs.subarray(idx * OBS_DIM, (idx + 1) * OBS_DIM), k * OBS_DIM);
      by.set(ds.act.subarray(idx * ACT_DIM, (idx + 1) * ACT_DIM), k * ACT_DIM);
    }
    const f = o.steps > 1 ? step / (o.steps - 1) : 1;
    const lr = o.lr0 * Math.pow(o.lr1 / o.lr0, f);
    last = im.step(bx, by, o.batch, lr);
  }
  policy.setPiWeights(im.params);
  return last;
}
