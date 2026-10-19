import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { GaussianPolicy, ensureBackend } from '../../src/rl/model';
import { FastImitator } from '../../src/rl/fastfit';
import { ACT_DIM, OBS_DIM } from '../../src/shared/types';
import { makeRng } from '../../src/shared/rng';

describe('FastImitator', () => {
  it('matches TF autodiff on loss and every gradient', async () => {
    await ensureBackend();
    const policy = new GaussianPolicy(3e-4, 5);
    const rng = makeRng(3, 'fastfit');
    const bs = 24;
    const x = new Float32Array(bs * OBS_DIM).map(() => rng.signed());
    const y = new Float32Array(bs * ACT_DIM).map(() => rng.signed() * 0.9);

    const im = new FastImitator();
    im.load(policy.getPiWeights());
    const loss = im.gradients(x, y, bs);
    const mine = im.lastGradients();

    const xt = tf.tensor2d(x, [bs, OBS_DIM]);
    const yt = tf.tensor2d(y, [bs, ACT_DIM]);
    const vars = policy.pi.trainableWeights.map((w) => (w as unknown as { val: tf.Variable }).val);
    const ref = tf.variableGrads(() => tf.tanh(policy.pi.apply(xt) as tf.Tensor).sub(yt).square().mean() as tf.Scalar, vars);
    expect(loss).toBeCloseTo(ref.value.dataSync()[0]!, 5);
    vars.forEach((v, i) => {
      const g = ref.grads[v.name]!.dataSync() as Float32Array;
      expect(g.length).toBe(mine[i]!.length);
      let worst = 0;
      for (let k = 0; k < g.length; k++) worst = Math.max(worst, Math.abs(g[k]! - mine[i]![k]!));
      expect(worst).toBeLessThan(1e-5);
    });
    xt.dispose();
    yt.dispose();
    policy.dispose();
  });

  it('reduces the regression loss', async () => {
    await ensureBackend();
    const policy = new GaussianPolicy(3e-4, 2);
    const rng = makeRng(8, 'fastfit2');
    const bs = 128;
    const x = new Float32Array(bs * OBS_DIM).map(() => rng.signed() * 0.5);
    const y = new Float32Array(bs * ACT_DIM);
    for (let i = 0; i < bs; i++) {
      y[i * 2] = Math.tanh(x[i * OBS_DIM + 3]! * 2);
      y[i * 2 + 1] = 0.5 * x[i * OBS_DIM + 7]!;
    }
    const im = new FastImitator();
    im.load(policy.getPiWeights());
    const first = im.step(x, y, bs, 2e-3);
    let last = first;
    for (let i = 0; i < 150; i++) last = im.step(x, y, bs, 2e-3);
    expect(last).toBeLessThan(first * 0.3);
    policy.dispose();
  });

  it('is seeded: same seed, same init and same actions', async () => {
    await ensureBackend();
    const a = new GaussianPolicy(3e-4, 11);
    const b = new GaussianPolicy(3e-4, 11);
    const obs = new Float32Array(4 * OBS_DIM).map((_, i) => Math.sin(i));
    expect(Array.from(a.actMean(obs, 4))).toEqual(Array.from(b.actMean(obs, 4)));
    expect(Array.from(a.act(obs, 4).action)).toEqual(Array.from(b.act(obs, 4).action));
    a.dispose();
    b.dispose();
  });
});
