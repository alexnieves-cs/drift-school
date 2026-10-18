import * as tf from '@tensorflow/tfjs';
import { ACT_DIM, OBS_DIM, type Policy } from '../shared/types';
import { makeRng, randn, type RNG } from '../shared/rng';

const HIDDEN = 128;

export class GaussianPolicy implements Policy {
  readonly pi: tf.LayersModel;
  readonly value: tf.LayersModel;
  readonly logStd: tf.Variable;
  private opt: tf.Optimizer;
  lr: number;
  /** Source of the exploration noise. Seeded, so a seed reproduces the rollout. */
  private noise: RNG;

  constructor(lr = 3e-4, seed = 1) {
    this.lr = lr;
    this.noise = makeRng(seed, 'policy/noise');
    // Every initializer carries its own seed: tf's global RNG would make runs differ.
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
    // Start with some forward throttle.
    tf.tidy(() => {
      const last = this.pi.getWeights();
      const next = last.slice(0, -1);
      next.push(tf.tensor1d([0, 0.7]));
      this.pi.setWeights(next);
    });
    this.opt = tf.train.adam(lr);
  }

  setLr(lr: number): void {
    this.lr = lr;
    this.opt.dispose();
    this.opt = tf.train.adam(lr);
  }

  /**
   * Sample actions. `pre` is the pre-tanh sample; `ppoLoss` needs it back exactly, because
   * recovering it with atanh is wrong once `action` saturates.
   */
  act(obs: Float32Array, n: number): { action: Float32Array; pre: Float32Array; logp: Float32Array; value: Float32Array } {
    const action = new Float32Array(n * ACT_DIM);
    const pre = new Float32Array(n * ACT_DIM);
    const logp = new Float32Array(n);
    const value = new Float32Array(n);
    const eps = new Float32Array(n * ACT_DIM);
    for (let i = 0; i < eps.length; i++) eps[i] = randn(this.noise);
    tf.tidy(() => {
      const x = tf.tensor2d(obs, [n, OBS_DIM]);
      const mean = this.pi.apply(x) as tf.Tensor2D;
      const std = tf.exp(this.logStd).reshape([1, ACT_DIM]);
      const u = mean.add(std.mul(tf.tensor2d(eps, [n, ACT_DIM])));
      const a = tf.tanh(u);
      const v = (this.value.apply(x) as tf.Tensor).reshape([n]);
      const logProb = gaussianLogp(u, mean, std).sub(tf.log(tf.scalar(1).sub(a.square()).add(1e-6)).sum(-1));
      action.set(a.dataSync() as Float32Array);
      pre.set(u.dataSync() as Float32Array);
      logp.set(logProb.dataSync() as Float32Array);
      value.set(v.dataSync() as Float32Array);
    });
    return { action, pre, logp, value };
  }

  /** State value for each observation row. */
  valueOf(obs: Float32Array, n: number): Float32Array {
    const out = new Float32Array(n);
    tf.tidy(() => {
      const x = tf.tensor2d(obs, [n, OBS_DIM]);
      out.set((this.value.apply(x) as tf.Tensor).reshape([n]).dataSync() as Float32Array);
    });
    return out;
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

  imitate(obs: Float32Array, actions: Float32Array, n: number, lr = 1e-3): number {
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
        m: tf.variable(tf.zeros(v.shape)),
        v: tf.variable(tf.zeros(v.shape)),
      };
      this.slots.set(v.name, slot);
    }
    return slot;
  }

  /**
   * One clipped-PPO Adam step on a minibatch. `pre` are the stored pre-tanh samples from `act`.
   * `policyScale` of 0 trains only the value head (critic warm-up).
   */
  ppoLoss(
    obs: Float32Array,
    pre: Float32Array,
    oldLogp: Float32Array,
    adv: Float32Array,
    returns: Float32Array,
    n: number,
    clip: number,
    entCoef: number,
    policyScale = 1,
  ): { loss: number; kl: number; entropy: number; valueLoss: number } {
    let kl = 0;
    let entropy = 0;
    let valueLoss = 0;
    let loss = 0;
    tf.tidy(() => {
      const x = tf.tensor2d(obs, [n, OBS_DIM]);
      const u = tf.tensor2d(pre, [n, ACT_DIM]);
      const old = tf.tensor1d(oldLogp);
      const advantage = tf.tensor1d(adv);
      const ret = tf.tensor1d(returns);
      this.opt.minimize(() => {
        const mean = this.pi.apply(x) as tf.Tensor2D;
        const std = tf.exp(this.logStd).reshape([1, ACT_DIM]);
        const logProb = gaussianLogp(u, mean, std).sub(tf.log(tf.scalar(1).sub(tf.tanh(u).square()).add(1e-6)).sum(-1));
        const ratio = tf.exp(logProb.sub(old));
        const unclipped = ratio.mul(advantage);
        const clipped = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(advantage);
        const policyLoss = tf.minimum(unclipped, clipped).mean().neg();
        const vpred = (this.value.apply(x) as tf.Tensor).reshape([n]);
        const vLoss = vpred.sub(ret).square().mean();
        const ent = this.logStd.add(0.5 * (1 + Math.log(2 * Math.PI))).sum();
        const total = policyLoss.mul(policyScale).add(vLoss.mul(0.5)).sub(ent.mul(entCoef * policyScale)) as tf.Scalar;
        kl = old.sub(logProb).mean().dataSync()[0] ?? 0;
        entropy = ent.dataSync()[0] ?? 0;
        valueLoss = vLoss.dataSync()[0] ?? 0;
        loss = total.dataSync()[0] ?? 0;
        return total;
      });
    });
    return { loss, kl, entropy, valueLoss };
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
    tf.tidy(() => {
      const piW = this.pi.getWeights();
      const vW = this.value.getWeights();
      let k = 0;
      const take = (shape: number[]) => tf.tensor(new Float32Array(weights[k++]!.slice(0)), shape);
      this.pi.setWeights(piW.map((w) => take(w.shape)));
      this.logStd.assign(tf.tensor1d(new Float32Array(weights[k++]!.slice(0))));
      this.value.setWeights(vW.map((w) => take(w.shape)));
    });
  }

  dispose(): void {
    this.pi.dispose();
    this.value.dispose();
    this.logStd.dispose();
    this.opt.dispose();
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
