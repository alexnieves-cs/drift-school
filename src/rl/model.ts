import * as tf from '@tensorflow/tfjs';
import { ACT_DIM, OBS_DIM, type Policy } from '../shared/types';

const HIDDEN = 128;

export class GaussianPolicy implements Policy {
  readonly pi: tf.LayersModel;
  readonly value: tf.LayersModel;
  readonly logStd: tf.Variable;
  private opt: tf.Optimizer;
  lr: number;

  constructor(lr = 3e-4) {
    this.lr = lr;
    const input = tf.input({ shape: [OBS_DIM] });
    let h = tf.layers.dense({ units: HIDDEN, activation: 'tanh', kernelInitializer: 'glorotUniform' }).apply(input) as tf.SymbolicTensor;
    h = tf.layers.dense({ units: HIDDEN, activation: 'tanh' }).apply(h) as tf.SymbolicTensor;
    const mean = tf.layers.dense({ units: ACT_DIM, biasInitializer: tf.initializers.constant({ value: 0 }) }).apply(h) as tf.SymbolicTensor;
    this.pi = tf.model({ inputs: input, outputs: mean });
    const vi = tf.input({ shape: [OBS_DIM] });
    let vh = tf.layers.dense({ units: HIDDEN, activation: 'tanh' }).apply(vi) as tf.SymbolicTensor;
    vh = tf.layers.dense({ units: HIDDEN, activation: 'tanh' }).apply(vh) as tf.SymbolicTensor;
    const v = tf.layers.dense({ units: 1 }).apply(vh) as tf.SymbolicTensor;
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
  }

  setLr(lr: number): void {
    this.lr = lr;
    this.opt.dispose();
    this.opt = tf.train.adam(lr);
  }

  act(obs: Float32Array, n: number): { action: Float32Array; logp: Float32Array; value: Float32Array } {
    const action = new Float32Array(n * ACT_DIM);
    const logp = new Float32Array(n);
    const value = new Float32Array(n);
    tf.tidy(() => {
      const x = tf.tensor2d(obs, [n, OBS_DIM]);
      const mean = this.pi.apply(x) as tf.Tensor2D;
      const std = tf.exp(this.logStd).reshape([1, ACT_DIM]);
      const eps = tf.randomNormal([n, ACT_DIM]);
      const u = mean.add(std.mul(eps));
      const a = tf.tanh(u);
      const v = (this.value.apply(x) as tf.Tensor).reshape([n]);
      const logProb = gaussianLogp(u, mean, std).sub(tf.log(tf.scalar(1).sub(a.square()).add(1e-6)).sum(-1));
      action.set(a.dataSync() as Float32Array);
      logp.set(logProb.dataSync() as Float32Array);
      value.set(v.dataSync() as Float32Array);
    });
    return { action, logp, value };
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

  imitate(obs: Float32Array, actions: Float32Array, n: number): number {
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
        v.assign(v.sub(mh.div(vh.sqrt().add(1e-8)).mul(this.lr)));
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

  ppoLoss(
    obs: Float32Array,
    actions: Float32Array,
    oldLogp: Float32Array,
    adv: Float32Array,
    returns: Float32Array,
    n: number,
    clip: number,
    entCoef: number,
  ): { loss: number; kl: number; entropy: number; valueLoss: number } {
    const x = tf.tensor2d(obs, [n, OBS_DIM]);
    const act = tf.tensor2d(actions, [n, ACT_DIM]);
    const old = tf.tensor1d(oldLogp);
    const advantage = tf.tensor1d(adv);
    const ret = tf.tensor1d(returns);
    const u = tf.atanh(act.clipByValue(-0.999, 0.999));
    let kl = 0;
    let entropy = 0;
    let valueLoss = 0;
    const lossT = tf.tidy(() => {
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
      kl = old.sub(logProb).mean().dataSync()[0] ?? 0;
      entropy = ent.dataSync()[0] ?? 0;
      valueLoss = vLoss.dataSync()[0] ?? 0;
      return policyLoss.add(vLoss.mul(0.5)).sub(ent.mul(entCoef));
    });
    const loss = lossT.dataSync()[0] ?? 0;
    this.opt.minimize(() =>
      tf.tidy(() => {
        const mean = this.pi.apply(x) as tf.Tensor2D;
        const std = tf.exp(this.logStd).reshape([1, ACT_DIM]);
        const pre = tf.atanh(act.clipByValue(-0.999, 0.999));
        const logProb = gaussianLogp(pre, mean, std).sub(
          tf.log(tf.scalar(1).sub(tf.tanh(pre).square()).add(1e-6)).sum(-1),
        );
        const ratio = tf.exp(logProb.sub(old));
        const unclipped = ratio.mul(advantage);
        const clippedR = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(advantage);
        const policyLoss = tf.minimum(unclipped, clippedR).mean().neg();
        const vpred = (this.value.apply(x) as tf.Tensor).reshape([n]);
        const vLoss = vpred.sub(ret).square().mean();
        const ent = this.logStd.add(0.5 * (1 + Math.log(2 * Math.PI))).sum();
        return policyLoss.add(vLoss.mul(0.5)).sub(ent.mul(entCoef)) as tf.Scalar;
      }),
    );
    x.dispose();
    act.dispose();
    old.dispose();
    advantage.dispose();
    ret.dispose();
    u.dispose();
    lossT.dispose();
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
    const piW = this.pi.getWeights();
    const vW = this.value.getWeights();
    let k = 0;
    const take = (shape: number[]) => {
      const buf = weights[k++]!;
      const data = new Float32Array(buf.slice(0));
      return tf.tensor(Array.from(data), shape);
    };
    this.pi.setWeights(piW.map((w) => take(w.shape)));
    const std = new Float32Array(weights[k++]!.slice(0));
    this.logStd.assign(tf.tensor1d(Array.from(std)));
    this.value.setWeights(vW.map((w) => take(w.shape)));
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
