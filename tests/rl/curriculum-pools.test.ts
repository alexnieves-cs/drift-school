import { describe, expect, it } from 'vitest';
import { ScenarioSet, STAGE_PROFILES, episodeConfig } from '../../src/rl/scenarios';
import { Trainer } from '../../src/rl/trainer';
import { makeRng } from '../../src/shared/rng';
import { STAGE_WIDTH } from '../../src/shared/constants';
import { SURFACES } from '../../src/sim/dr';
import { MATRIX_SEEDS, STAGE0_HELD_OUT, VALIDATION_SEEDS, DEFAULT_TRAIN_SEEDS, reservedSeeds } from '../../src/rl/seeds';
import { hashBuffers } from '../../src/shared/hash';

describe('stage-appropriate scenarios', () => {
  it('builds stage pools with the stage width and replays earlier stages ~20%', () => {
    const set = new ScenarioSet([1, 2, 3, 4, 5]);
    for (let stage = 0; stage <= 3; stage++) {
      for (const t of set.tracks(stage)) {
        expect(t.id.endsWith(`-s${stage}`)).toBe(true);
        expect(t.samples[0]!.halfWidth * 2).toBeCloseTo(STAGE_WIDTH[stage]!, 1);
      }
    }
    const rng = makeRng(4, 'sc');
    const counts = [0, 0, 0, 0];
    const N = 4000;
    for (let i = 0; i < N; i++) counts[set.sample(rng, 2).stage]!++;
    expect(counts[2]! / N).toBeGreaterThan(0.75);
    expect(counts[2]! / N).toBeLessThan(0.88);
    expect(counts[3]).toBe(0);
    expect(counts[0]! + counts[1]!).toBeGreaterThan(N * 0.12);
  });

  it('follows the per-stage DR and surface profiles', () => {
    const set = new ScenarioSet([1, 2, 3]);
    const rng = makeRng(9, 'sc2');
    // Stage 0: no DR, native surface only.
    for (let i = 0; i < 200; i++) {
      const sc = set.sample(rng, 0);
      expect(sc.dr).toBe('off');
      expect(sc.surface).toBe('native');
    }
    // Stage 3: mostly high DR, uniform surfaces appear with their mu.
    let high = 0;
    let dirt = 0;
    const N = 1500;
    for (let i = 0; i < N; i++) {
      const sc = set.sample(rng, 3);
      if (sc.stage !== 3) continue;
      if (sc.dr === 'high') high++;
      if (sc.surface === 'dirt') {
        dirt++;
        expect(sc.track.samples.every((s) => s.mu === SURFACES.dirt)).toBe(true);
      }
    }
    expect(high).toBeGreaterThan(N * 0.8 * (STAGE_PROFILES[3]!.dr[2] - 0.1));
    expect(dirt).toBeGreaterThan(0);
  });

  it('draws episode configs deterministically and samples DR levels differently', () => {
    const set = new ScenarioSet([7]);
    const sc = { track: set.tracks(2)[0]!, stage: 2, dr: 'high' as const };
    const a = episodeConfig(sc, 5, 3);
    const b = episodeConfig(sc, 5, 3);
    expect(a.car.mass).toBe(b.car.mass);
    expect(a.seed).toBe(b.seed);
    const off = episodeConfig({ ...sc, dr: 'off' }, 5, 3);
    expect(off.car.mass).toBeCloseTo(1.6, 6);
    expect(off.noiseScale).toBe(0);
  });

  it('keeps training, validation and evaluation seeds disjoint', () => {
    const reserved = reservedSeeds();
    expect(reserved.size).toBe(STAGE0_HELD_OUT.length + MATRIX_SEEDS.length + VALIDATION_SEEDS.length);
    expect(DEFAULT_TRAIN_SEEDS.some((s) => reserved.has(s))).toBe(false);
    for (let k = 1; k < 8; k++) {
      for (let i = 1; i <= 24; i++) expect(reserved.has(1000 * k + i)).toBe(false);
    }
  });
});

describe('trainer in scenario mode', () => {
  it('samples tracks from the current stage pool and resamples every rollout', async () => {
    const set = new ScenarioSet([1, 2, 3, 4]);
    const trainer = new Trainer({ n: 6, horizon: 8, epochs: 1, minibatch: 32, seed: 3 });
    await trainer.init(set.tracks(0));
    trainer.setScenarios(set);
    trainer.curriculum.stage = 3;
    trainer.resetEnvs();
    const ids = trainer.env.trackIds();
    expect(ids.every((id) => /-s[0-3](@|$)/.test(id))).toBe(true);
    expect(ids.some((id) => id.includes('-s3'))).toBe(true);
  });

  it('critic warm-up leaves the policy alone, PPO then moves it', async () => {
    const set = new ScenarioSet([1, 2]);
    const trainer = new Trainer({ n: 4, horizon: 12, epochs: 1, minibatch: 48, seed: 5, lr: 1e-3 });
    await trainer.init(set.tracks(0));
    trainer.setScenarios(set);
    const piHash = () => hashBuffers(trainer.policy.getPiWeights().map((a) => a.slice().buffer));
    const vHash = () => hashBuffers(trainer.policy.value.getWeights().map((w) => Float32Array.from(w.dataSync() as Float32Array).buffer));
    const pi0 = piHash();
    const v0 = vHash();
    trainer.trainPolicy = false;
    await trainer.update();
    expect(piHash()).toBe(pi0);
    expect(vHash()).not.toBe(v0);
    trainer.trainPolicy = true;
    await trainer.update();
    expect(piHash()).not.toBe(pi0);
  });
});
