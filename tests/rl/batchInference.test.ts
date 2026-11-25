import { describe, expect, it, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { ACT_DIM, OBS_DIM } from '../../src/shared/types';
import { makeRng } from '../../src/shared/rng';
import { GaussianPolicy, ensureBackend } from '../../src/rl/model';
import { selectBackend, smokeTest } from '../../src/rl/backend';
import { BatchedInference, batchedMeanActions, type ForwardFn } from '../../src/rl/batchInference';

function obsRows(rows: number, seed: number): Float32Array {
  const rng = makeRng(seed, 'batch-obs');
  const x = new Float32Array(rows * OBS_DIM);
  for (let i = 0; i < x.length; i++) x[i] = rng.range(-2, 2);
  return x;
}

describe('backend selection', () => {
  it('uses CPU when asked for CPU, and the smoke test passes there', async () => {
    const choice = await selectBackend(['cpu']);
    expect(choice).toEqual({ active: 'cpu', attempts: [{ name: 'cpu', ok: true }], fellBack: false });
    expect(smokeTest()).toBeNull();
  });

  it('falls back to CPU in Node, where neither GPU backend is registered', async () => {
    const choice = await selectBackend();
    expect(choice.active).toBe('cpu');
    expect(choice.fellBack).toBe(true);
    expect(choice.attempts.map((a) => [a.name, a.ok])).toEqual([
      ['webgpu', false],
      ['webgl', false],
      ['cpu', true],
    ]);
    expect(choice.attempts[0]!.reason).toMatch(/not installed/);
    expect(choice.attempts[1]!.reason).toBe('not registered');
    expect(tf.getBackend()).toBe('cpu');
  });

  it('falls back when a registered backend fails to initialise', async () => {
    // Stand-ins for a browser without a usable GPU: one factory throws, one rejects.
    expect(tf.findBackendFactory('webgl')).toBeFalsy();
    expect(tf.findBackendFactory('webgpu')).toBeFalsy();
    tf.registerBackend('webgl', () => {
      throw new Error('no WebGL context');
    });
    tf.registerBackend('webgpu', () => Promise.reject(new Error('no adapter')));
    // TF.js logs each failed initialisation; keep the test output readable.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const choice = await selectBackend();
      expect(choice.active).toBe('cpu');
      expect(choice.attempts.map((a) => a.ok)).toEqual([false, false, true]);
      expect(choice.attempts[0]!.reason).toMatch(/failed to initialise/);
      expect(choice.attempts[1]!.reason).toMatch(/failed to initialise/);
      expect(tf.getBackend()).toBe('cpu');
    } finally {
      warn.mockRestore();
      tf.removeBackend('webgl');
      tf.removeBackend('webgpu');
    }
  });

  it('always tries CPU last, even when the order leaves it out', async () => {
    const choice = await selectBackend(['webgl']);
    expect(choice.active).toBe('cpu');
    expect(choice.attempts.at(-1)).toEqual({ name: 'cpu', ok: true });
  });
});

describe('batched inference on CPU', () => {
  it('gives every caller exactly what actMean gives it alone, across padding and chunking', async () => {
    await ensureBackend();
    const policy = new GaussianPolicy(3e-4, 3);
    const sizes = [1, 5, 0, 31, 70];
    const reqs = sizes.map((rows, k) => ({ obs: obsRows(rows, k + 1), rows }));
    const alone = reqs.map((r) => (r.rows ? policy.actMean(r.obs, r.rows) : new Float32Array(0)));
    for (const opts of [{}, { maxBatch: 16 }, { buckets: [1], maxBatch: 1 }, { buckets: [4, 64], maxBatch: 50 }]) {
      const batch = batchedMeanActions(policy, opts);
      const outs = batch.run(reqs);
      outs.forEach((o, i) => expect(Array.from(o), `request ${i} with ${JSON.stringify(opts)}`).toEqual(Array.from(alone[i]!)));
      expect(batch.stats.rows).toBe(107);
    }
    policy.dispose();
  });

  it('pads to bucket sizes and splits large batches into chunks', () => {
    const shapes: number[] = [];
    const forward: ForwardFn = (x, rows) => {
      shapes.push(rows);
      expect(x.length).toBe(rows * 3);
      const y = new Float32Array(rows);
      for (let r = 0; r < rows; r++) y[r] = x[r * 3]! + x[r * 3 + 1]! + x[r * 3 + 2]!;
      return y;
    };
    const batch = new BatchedInference(forward, 3, 1, { buckets: [8, 32], maxBatch: 32 });
    const a = Float32Array.from({ length: 30 }, (_, i) => i);
    const b = Float32Array.from({ length: 120 }, (_, i) => -i);
    const [ya, yb] = batch.run([{ obs: a, rows: 10 }, { obs: b, rows: 40 }]);
    // 50 rows: one full chunk of 32, then 18 rows padded to 32.
    expect(shapes).toEqual([32, 32]);
    expect(batch.stats).toEqual({ calls: 2, rows: 50, padded: 14, shapes: [32] });
    expect(Array.from(ya!)).toEqual(Array.from({ length: 10 }, (_, r) => 3 * r * 3 + 3));
    expect(Array.from(yb!)).toEqual(Array.from({ length: 40 }, (_, r) => -(9 * r + 3)));
    batch.run([{ obs: a, rows: 2 }]);
    expect(shapes.at(-1)).toBe(8);
  });

  it('queues requests and serves them in one forward pass on flush', async () => {
    let calls = 0;
    const batch = new BatchedInference((x, rows) => {
      calls++;
      return x.slice(0, rows * 2).map((v) => v * 2);
    }, 2, 2);
    const p1 = batch.request(Float32Array.from([1, 2]), 1);
    const p2 = batch.request(Float32Array.from([3, 4, 5, 6]), 2);
    const p3 = batch.request(new Float32Array(0), 0);
    expect(batch.pending).toBe(3);
    batch.flush();
    expect(batch.pending).toBe(0);
    expect(calls).toBe(1);
    expect(Array.from(await p1)).toEqual([2, 4]);
    expect(Array.from(await p2)).toEqual([6, 8, 10, 12]);
    expect(Array.from(await p3)).toEqual([]);
  });

  it('rejects bad input and passes forward errors to every queued caller', async () => {
    const batch = new BatchedInference(() => {
      throw new Error('forward broke');
    }, 2, 1);
    expect(() => batch.run([{ obs: new Float32Array(3), rows: 2 }])).toThrow(/expected 4 inputs/);
    expect(() => batch.run([{ obs: new Float32Array(2), rows: -1 }])).toThrow(/non-negative/);
    expect(() => new BatchedInference(() => new Float32Array(0), 2, 1, { buckets: [0] })).toThrow(/positive integers/);
    const p1 = batch.request(new Float32Array(2), 1);
    const p2 = batch.request(new Float32Array(2), 1);
    batch.flush();
    await expect(p1).rejects.toThrow('forward broke');
    await expect(p2).rejects.toThrow('forward broke');
  });

  it('keeps the number of distinct shapes bounded by the buckets', async () => {
    await ensureBackend();
    const policy = new GaussianPolicy(3e-4, 4);
    const batch = batchedMeanActions(policy);
    for (let rows = 1; rows <= 60; rows += 3) batch.run([{ obs: obsRows(rows, rows), rows }]);
    expect(batch.stats.shapes.every((s) => [1, 8, 32, 128, 512].includes(s))).toBe(true);
    expect(batch.stats.shapes.length).toBeLessThanOrEqual(4);
    expect(ACT_DIM).toBe(2);
    policy.dispose();
  });
});
