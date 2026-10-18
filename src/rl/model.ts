import * as tf from '@tensorflow/tfjs';
import { ACT_DIM, OBS_DIM, type Policy } from '../shared/types';
import { makeRng, randn, type RNG } from '../shared/rng';

export const HIDDEN = 128;

export class GaussianPolicy implements Policy {
  readonly pi: tf.LayersModel;
  readonly value: tf.LayersModel;
  readonly logStd: tf.Variable;
  private opt: tf.Optimizer;
  private valueOpt: tf.Optimizer;
  lr: number;
  /** Seeded stream for action sampling. Weight init is seeded from the same `seed`. */
  private rng: RNG;
  /** Global-norm gradient clip for PPO steps. */
  maxGradNorm = 0.5;

  /**
   * `seed` makes initial weights and exploration noise reproducible. TF.js's own RNG is never
   * used, so two policies built with the same seed and fed the same data stay identical.
   */
  constructor(lr = 3e-4, seed = 1) {
    this.lr = lr;
    this.rng = makeRng(seed, 'policy/act');
    const init = (k: number) => tf.initializers.glorotUniform({ seed: seed * 101 + k });
    const input = tf.input({ shape: [OBS_DIM] });
    let h = tf.layers.dense({ units: HIDDEN, activation: 'tanh', kernelInitializer: init(1) }).apply(input) as tf.SymbolicTensor;
    h = tf.layers.dense({ units: HIDDEN, activation: 'tanh', kernelInitializer: init(2) }).apply(h) as tf.SymbolicTensor;
    const mean = tf.layers
      .dense({ units: ACT_DIM, kernelInitializer: init(3), biasInitializer: tf.initializers.constant({ value: 0 }) })
      .apply(h) as tf.SymbolicTensor;
    this.pi = tf.model({ inputs: input, outputs: mean });
    const vi = tf.input({ shape: [OBS_DIM] });
    let vh = tf.layers.dense({ units: HIDDEN, activation: 'tanh', kernelInitializer: init(4) }).apply(vi) as tf.SymbolicTensor;
    vh = tf.layers.dense({ units: HIDDEN, activation: 'tanh', kernelInitializer: init(5) }).apply(vh) as tf.SymbolicTensor;
    const v = tf.layers.dense({ units: 1, kernelInitializer: init(6) }).apply(vh) as tf.SymbolicTensor;
    this.value = tf.model({ inputs: vi, outputs: v });
    this.logStd = tf.variable(tf.tensor1d([-1.4, -1.2]), true);
    const last = this.pi.getWeights();
    const bias = last[last.length - 1]!;
    const b = bias.dataSync().slice();
    if (b.length === ACT_DIM) {
      b[0] = 0;
      b[1] = 0.7;
      const next = last.slice(0, -1);
      next.push(tf.tensor1d(Array.from(b)));
      this.pi.setWeights(next);
    }
    this.opt = tf.train.adam(lr);
    this.valueOpt = tf.train.adam(1e-3);
  }

  setLr(lr: number): void {
    this.lr = lr;
    this.opt.dispose();
    this.opt = tf.train.adam(lr);
  }

  setLogStd(steer: number, throttle: number): void {
    this.logStd.assign(tf.tensor1d([steer, throttle]));
  }

  getLogStd(): number[] {
    return Array.from(this.logStd.dataSync());
  }

  /** Gaussian noise from the seeded stream (never TF's global RNG). */
  private noise(n: number): tf.Tensor2D {
    const buf = new Float32Array(n * ACT_DIM);
    for (let i = 0; i < buf.length; i++) buf[i] = randn(this.rng);
    return tf.tensor2d(buf, [n, ACT_DIM]);
  }

  /**
   * `pre` is the pre-tanh Gaussian sample. PPO must score the sample it drew: recomputing it as
   * atanh(action) is wrong once actions saturate at +-1 (the imitation policy's do), and made the
   * old/new log-ratio huge so every PPO update was clipped to nothing.
   */
  act(obs: Float32Array, n: number): { action: Float32Array; logp: Float32Array; value: Float32Array; pre: Float32Array } {
    const action = new Float32Array(n * ACT_DIM);
    const pre = new Float32Array(n * ACT_DIM);
    const logp = new Float32Array(n);
    const value = new Float32Array(n);
    tf.tidy(() => {
      const x = tf.tensor2d(obs, [n, OBS_DIM]);
      const mean = this.pi.apply(x) as tf.Tensor2D;
      const std = tf.exp(this.logStd).reshape([1, ACT_DIM]);
      const eps = this.noise(n);
      const u = mean.add(std.mul(eps));
      const a = tf.tanh(u);
      const v = (this.value.apply(x) as tf.Tensor).reshape([n]);
      const logProb = gaussianLogp(u, mean, std).sub(tf.log(tf.scalar(1).sub(a.square()).add(1e-6)).sum(-1));
      action.set(a.dataSync() as Float32Array);
      pre.set(u.dataSync() as Float32Array);
      logp.set(logProb.dataSync() as Float32Array);
      value.set(v.dataSync() as Float32Array);
    });
    return { action, logp, value, pre };
  }

  /** Deterministic mean action, used for the showcase car. */
  actMean(obs: Float32Array, n: number): Float32Array {
    const action = new Float32Array(n * ACT_DIM);
    tf.tidy(() => {
      const x = tf.tensor2d(obs, [n, OBS_DIM]);
      const mean = this.pi.apply(x) as tf.Tensor;
      action.set(tf.tanh(mean).dataSync() as Float32Array);
    });
    return action;
  }

  imitate(obs: Float32Array, actions: Float32Array, n: number, lr = this.lr): number {
    const x = tf.tensor2d(obs, [n, OBS_DIM]);
    const y = tf.tensor2d(actions, [n, ACT_DIM]);
    const vars = this.pi.trainableWeights.map((w) => (w as unknown as { val: tf.Variable }).val);
    const step = tf.variableGrads(() => {
      return tf.tidy(() => {
        const pred = tf.tanh(this.pi.apply(x) as tf.Tensor);
        return pred.sub(y).square().mean() as tf.Scalar;
      });
    }, vars);
    const loss = step.value.dataSync()[0] ?? 0;
    tf.tidy(() => {
      for (const v of vars) {
        const g = step.grads[v.name];
        if (!g) continue;
        const slot = this.moment(v);
        const b1 = 0.9;
        const b2 = 0.999;
        const m = slot.m.mul(b1).add(g.mul(1 - b1));
        const vv = slot.v.mul(b2).add(g.square().mul(1 - b2));
        slot.m.assign(m);
        slot.v.assign(vv);
        const t = this.imitStep;
        const mh = m.div(1 - b1 ** t);
        const vh = vv.div(1 - b2 ** t);
        v.assign(v.sub(mh.div(vh.sqrt().add(1e-8)).mul(lr)));
      }
    });
    this.imitStep++;
    step.value.dispose();
    tf.dispose(step.grads);
    x.dispose();
    y.dispose();
    return loss;
  }

  private imitStep = 1;
  private slots = new Map<string, { m: tf.Variable; v: tf.Variable }>();

  private moment(v: tf.Variable): { m: tf.Variable; v: tf.Variable } {
    let slot = this.slots.get(v.name);
    if (!slot) {
      slot = {
        m: tf.variable(tf.zeros(v.shape), false),
        v: tf.variable(tf.zeros(v.shape), false),
      };
      this.slots.set(v.name, slot);
    }
    return slot;
  }

  /**
   * One clipped-PPO minibatch step (Adam, global grad-norm clip). `policyStep: false` trains
   * the critic only, which the recipe uses to fit the value head before the policy is touched.
   */
  ppoLoss(
    obs: Float32Array,
    actions: Float32Array,
    oldLogp: Float32Array,
    adv: Float32Array,
    returns: Float32Array,
    n: number,
    clip: number,
    entCoef: number,
    policyStep = true,
    pre?: Float32Array,
  ): { loss: number; kl: number; entropy: number; valueLoss: number } {
    const x = tf.tensor2d(obs, [n, OBS_DIM]);
    const act = tf.tensor2d(actions, [n, ACT_DIM]);
    const old = tf.tensor1d(oldLogp);
    const advantage = tf.tensor1d(adv);
    const ret = tf.tensor1d(returns);
    const preT = pre ? tf.tensor2d(pre, [n, ACT_DIM]) : null;
    let kl = 0;
    let entropy = 0;
    let valueLoss = 0;
    let loss = 0;
    const vars = policyStep
      ? [...this.piVars(), this.logStd, ...this.valueVars()]
      : this.valueVars();
    const opt = policyStep ? this.opt : this.valueOpt;
    const { grads, value } = tf.variableGrads(() => {
      const mean = this.pi.apply(x) as tf.Tensor2D;
      const std = tf.exp(this.logStd).reshape([1, ACT_DIM]);
      const u = preT ?? tf.atanh(act.clipByValue(-0.999, 0.999));
      const logProb = gaussianLogp(u, mean, std).sub(
        tf.log(tf.scalar(1).sub(tf.tanh(u).square()).add(1e-6)).sum(-1),
      );
      const ratio = tf.exp(logProb.sub(old));
      const unclipped = ratio.mul(advantage);
      const clippedR = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(advantage);
      const policyLoss = tf.minimum(unclipped, clippedR).mean().neg();
      const vpred = (this.value.apply(x) as tf.Tensor).reshape([n]);
      const vLoss = vpred.sub(ret).square().mean();
      const ent = this.logStd.add(0.5 * (1 + Math.log(2 * Math.PI))).sum();
      kl = old.sub(logProb).mean().dataSync()[0] ?? 0;
      entropy = ent.dataSync()[0] ?? 0;
      valueLoss = vLoss.dataSync()[0] ?? 0;
      return (policyStep ? policyLoss.add(vLoss.mul(0.5)).sub(ent.mul(entCoef)) : vLoss) as tf.Scalar;
    }, vars);
    loss = value.dataSync()[0] ?? 0;
    const norm = this.clipGrads(grads);
    void norm;
    opt.applyGradients(grads);
    if (policyStep) {
      // Keep exploration std in a sane band so PPO cannot collapse or blow it up.
      const ls = tf.tidy(() => this.logStd.clipByValue(-3.2, -0.4));
      this.logStd.assign(ls);
      ls.dispose();
    }
    value.dispose();
    tf.dispose(grads);
    x.dispose();
    act.dispose();
    old.dispose();
    advantage.dispose();
    ret.dispose();
    preT?.dispose();
    return { loss, kl, entropy, valueLoss };
  }

  private piVars(): tf.Variable[] {
    return this.pi.trainableWeights.map((w) => (w as unknown as { val: tf.Variable }).val);
  }

  private valueVars(): tf.Variable[] {
    return this.value.trainableWeights.map((w) => (w as unknown as { val: tf.Variable }).val);
  }

  /** Scales `grads` in place to a global L2 norm of at most `maxGradNorm`. Returns the pre-clip norm. */
  private clipGrads(grads: tf.NamedTensorMap): number {
    const names = Object.keys(grads);
    let sq = 0;
    for (const k of names) sq += tf.tidy(() => grads[k]!.square().sum()).dataSync()[0] ?? 0;
    const norm = Math.sqrt(sq);
    if (norm > this.maxGradNorm && norm > 0) {
      const f = this.maxGradNorm / norm;
      for (const k of names) {
        const g = grads[k]!;
        grads[k] = tf.tidy(() => g.mul(f));
        g.dispose();
      }
    }
    return norm;
  }

  /** Mean-network weights as flat arrays (W1, b1, W2, b2, W3, b3), TF dense layout. */
  getPiWeights(): Float32Array[] {
    return this.pi.getWeights().map((w) => Float32Array.from(w.dataSync() as Float32Array));
  }

  setPiWeights(arrays: Float32Array[]): void {
    const shapes = this.pi.getWeights().map((w) => w.shape);
    const made = arrays.map((a, i) => tf.tensor(a, shapes[i]));
    this.pi.setWeights(made);
    for (const t of made) t.dispose();
  }

  getWeights(): ArrayBuffer[] {
    const out: ArrayBuffer[] = [];
    const push = (t: tf.Tensor) => {
      const data = t.dataSync();
      const copy = new Float32Array(data.length);
      copy.set(data);
      out.push(copy.buffer);
    };
    for (const w of this.pi.getWeights()) push(w);
    push(this.logStd);
    for (const w of this.value.getWeights()) push(w);
    return out;
  }

  setWeights(weights: ArrayBuffer[]): void {
    const piW = this.pi.getWeights();
    const vW = this.value.getWeights();
    let k = 0;
    const made: tf.Tensor[] = [];
    const take = (shape: number[]) => {
      const t = tf.tensor(new Float32Array(weights[k++]!.slice(0)), shape);
      made.push(t);
      return t;
    };
    this.pi.setWeights(piW.map((w) => take(w.shape)));
    const std = new Float32Array(weights[k++]!.slice(0));
    this.logStd.assign(tf.tensor1d(Array.from(std)));
    this.value.setWeights(vW.map((w) => take(w.shape)));
    for (const t of made) t.dispose();
  }

  dispose(): void {
    this.pi.dispose();
    this.value.dispose();
    this.logStd.dispose();
    this.opt.dispose();
    this.valueOpt.dispose();
    for (const slot of this.slots.values()) {
      slot.m.dispose();
      slot.v.dispose();
    }
    this.slots.clear();
  }
}

function gaussianLogp(u: tf.Tensor, mean: tf.Tensor, std: tf.Tensor): tf.Tensor {
  const z = u.sub(mean).div(std);
  const log = z.square().mul(-0.5).sub(tf.log(std)).sub(0.5 * Math.log(2 * Math.PI));
  return log.sum(-1);
}

export async function ensureBackend(): Promise<void> {
  await tf.ready();
  if (tf.getBackend() !== 'cpu') {
    await tf.setBackend('cpu');
    await tf.ready();
  }
}
