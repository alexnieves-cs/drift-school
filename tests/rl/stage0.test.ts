import { describe, expect, it } from 'vitest';
import { Trainer } from '../../src/rl/trainer';
import { trainBaseRecipe } from '../../src/rl/recipe';
import { casesFor, evaluateCases } from '../../src/eval/harness';
import { generateTrack } from '../../src/track/generate';
import { buildTrackFromPoints } from '../../src/track/build';

describe('stage 0 training', () => {
  it('imitation warm start + PPO updates reach 80% lap completion on held-out seeds within 3 minutes', async () => {
    // The recipe is pure-pursuit imitation, a critic warm-up, then PPO. It is not PPO from scratch:
    // from-scratch PPO does not learn laps at the CPU training rate (see README).
    const { trainer, seconds } = await trainBaseRecipe();
    // Seeds 101-108 are never in the training pool (seeds 1-4).
    const tracks = [101, 102, 103, 104, 105, 106, 107, 108].map((seed) => generateTrack({ seed, stage: 0 }));
    const out = evaluateCases((obs, n) => trainer.policy.actMean(obs, n), casesFor(tracks, { copies: 2 }));
    expect(seconds).toBeLessThan(180);
    expect(out.rate).toBeGreaterThanOrEqual(0.8);
  }, 180_000);

  it('keeps the warm-start policy through further PPO updates', async () => {
    const { trainer } = await trainBaseRecipe();
    for (let i = 0; i < 6; i++) await trainer.update();
    const tracks = [101, 102, 103, 104].map((seed) => generateTrack({ seed, stage: 0 }));
    const out = evaluateCases((obs, n) => trainer.policy.actMean(obs, n), casesFor(tracks, { copies: 2 }));
    expect(out.rate).toBeGreaterThanOrEqual(0.7);
  }, 180_000);

  it('swaps the track pool without changing weights, with every env starting on the new track', async () => {
    const tracks = [generateTrack({ seed: 2, stage: 0 })];
    const trainer = new Trainer({ n: 4, horizon: 16, epochs: 1, minibatch: 64 });
    await trainer.init(tracks);
    const before = trainer.weightHash();
    const custom = buildTrackFromPoints(
      Float32Array.from([0, 0, 8, 0, 12, 4, 8, 8, 0, 8, -4, 4, 0.2, 0.1]),
      { halfWidth: 1.2, mu: 1, source: 'user', id: 'user-s' },
    );
    trainer.queueTrackPool([custom, tracks[0]!], true);
    trainer.forceSwap();
    expect(trainer.weightHash()).toBe(before);
    expect(trainer.env.trackIds()).toEqual(['user-s', 'user-s', 'user-s', 'user-s']);
    expect(trainer.mix.entropy).toBe(0.02);
    expect(trainer.mix.weights[0]).toBeCloseTo(0.7, 2);
  });
});
