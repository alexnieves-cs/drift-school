import { describe, expect, it } from 'vitest';
import { GaussianPolicy, ensureBackend } from '../../src/rl/model';
import { ToyEnv } from '../../src/rl/toy';
import { hashBuffers } from '../../src/shared/hash';
import { Curriculum } from '../../src/rl/curriculum';
import { TrackMix } from '../../src/rl/trackMix';
import { makeRng } from '../../src/shared/rng';
import { generateTrack } from '../../src/track/generate';
import { forkCheckpoint, loadCheckpoint, saveCheckpoint } from '../../src/rl/checkpoint';
import { OBS_DIM } from '../../src/shared/types';

describe('ppo', () => {
  it('solves the toy reach task in under 200 updates', async () => {
    await ensureBackend();
    const policy = new GaussianPolicy(3e-4);
    const envs = Array.from({ length: 16 }, () => new ToyEnv());
    const obs = new Float32Array(16 * OBS_DIM);
    const resetAll = () => {
      envs.forEach((e, i) => obs.set(e.reset(), i * OBS_DIM));
    };
    resetAll();
    let solved = false;
    for (let update = 0; update < 200 && !solved; update++) {
      const T = 40;
      const n = 16;
      const obsBuf = new Float32Array(T * n * OBS_DIM);
      const actBuf = new Float32Array(T * n * 2);
      const logBuf = new Float32Array(T * n);
      const valBuf = new Float32Array(T * n);
      const rewBuf = new Float32Array(T * n);
      const doneBuf = new Uint8Array(T * n);
      for (let t = 0; t < T; t++) {
        const out = policy.act(obs, n);
        obsBuf.set(obs, t * n * OBS_DIM);
        actBuf.set(out.pre, t * n * 2);
        logBuf.set(out.logp, t * n);
        valBuf.set(out.value, t * n);
        for (let i = 0; i < n; i++) {
          const step = envs[i]!.step(out.action[i * 2]!);
          rewBuf[t * n + i] = step.reward;
          doneBuf[t * n + i] = step.done ? 1 : 0;
          if (step.done) obs.set(envs[i]!.reset(), i * OBS_DIM);
          else obs.set(step.obs, i * OBS_DIM);
        }
      }
      const adv = new Float32Array(T * n);
      const ret = new Float32Array(T * n);
      for (let i = 0; i < n; i++) {
        let gae = 0;
        for (let t = T - 1; t >= 0; t--) {
          const idx = t * n + i;
          const nextV = t === T - 1 || doneBuf[idx] ? 0 : valBuf[(t + 1) * n + i]!;
          const delta = rewBuf[idx]! + 0.99 * nextV - valBuf[idx]!;
          gae = delta + 0.99 * 0.95 * (doneBuf[idx] ? 0 : gae);
          adv[idx] = gae;
          ret[idx] = gae + valBuf[idx]!;
        }
      }
      let mean = 0;
      for (const a of adv) mean += a;
      mean /= adv.length;
      let v = 0;
      for (const a of adv) v += (a - mean) ** 2;
      const std = Math.sqrt(v / adv.length) + 1e-8;
      for (let i = 0; i < adv.length; i++) adv[i] = (adv[i]! - mean) / std;
      policy.ppoLoss(obsBuf, actBuf, logBuf, adv, ret, T * n, 0.2, 0.01);
      if (update > 20 && update % 5 === 0) {
        let err = 0;
        const probes = 8;
        for (let p = 0; p < probes; p++) {
          const e = new ToyEnv();
          let o = e.reset();
          for (let s = 0; s < 40; s++) {
            const a = policy.actMean(o, 1);
            const step = e.step(a[0]!);
            o = step.obs;
            if (step.done) break;
          }
          err += Math.abs(e.x);
        }
        if (err / probes < 0.2) solved = true;
      }
    }
    policy.dispose();
    expect(solved).toBe(true);
  }, 180_000);

  it('round-trips weights', async () => {
    await ensureBackend();
    const a = new GaussianPolicy();
    const b = new GaussianPolicy();
    const w = a.getWeights();
    b.setWeights(w);
    expect(hashBuffers(a.getWeights())).toBe(hashBuffers(b.getWeights()));
    const obs = new Float32Array(OBS_DIM);
    obs[0] = 0.3;
    const aa = a.actMean(obs, 1);
    const bb = b.actMean(obs, 1);
    expect(aa[0]).toBeCloseTo(bb[0]!, 4);
    expect(aa[1]).toBeCloseTo(bb[1]!, 4);
    a.dispose();
    b.dispose();
  });
});

describe('curriculum and track mix', () => {
  it('promotes when completion clears the stage rule and replays prior stages', () => {
    const c = new Curriculum(20);
    const rng = makeRng(1);
    const counts = [0, 0, 0, 0, 0];
    for (let i = 0; i < 20; i++) c.note({ completed: true, tRef: 10, lapTime: 12 });
    expect(c.maybePromote()).toBe(true);
    expect(c.stage).toBe(1);
    for (let i = 0; i < 5000; i++) counts[c.sampleStage(rng)]!++;
    expect(counts[1]! / 5000).toBeGreaterThan(0.7);
    expect(counts[0]! / 5000).toBeGreaterThan(0.1);
  });

  it('samples a fine-tune pool near 70/30 and bumps entropy for 20 updates', () => {
    const user = generateTrack({ seed: 1, stage: 0 });
    const proc = [generateTrack({ seed: 2, stage: 0 }), generateTrack({ seed: 3, stage: 0 })];
    const mix = new TrackMix();
    mix.startFineTune(user, proc);
    expect(mix.entropy).toBe(0.02);
    expect(mix.lr).toBeCloseTo(3e-4);
    const rng = makeRng(5);
    let userN = 0;
    const N = 10_000;
    for (let i = 0; i < N; i++) if (mix.sample(rng).id === user.id) userN++;
    expect(Math.abs(userN / N - 0.7)).toBeLessThan(0.03);
    for (let i = 0; i < 20; i++) mix.onUpdate();
    expect(mix.entropy).toBe(0.01);
    expect(mix.fineTuneUpdatesLeft).toBe(0);
  });

  it('forks a checkpoint without dropping the base name', async () => {
    await saveCheckpoint({ name: 'base', gen: 3, stage: 1, weights: [[1, 2, 3]] });
    const forked = await forkCheckpoint('base', 'user-abc');
    expect(forked?.name).toBe('user-abc');
    expect(await loadCheckpoint('base')).toBeTruthy();
    expect((await loadCheckpoint('user-abc'))?.weights[0]).toEqual([1, 2, 3]);
  });
});
