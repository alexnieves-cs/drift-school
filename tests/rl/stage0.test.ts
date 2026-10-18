import { describe, expect, it } from 'vitest';
import { Trainer } from '../../src/rl/trainer';
import { generateTrack } from '../../src/track/generate';
import { buildTrackFromPoints } from '../../src/track/build';

describe('stage 0 training', () => {
  it('reaches 80% lap completion within the 3 minute budget', async () => {
    const tracks = [1, 2, 3, 4].map((seed) => generateTrack({ seed, stage: 0 }));
    const trainer = new Trainer({ n: 8, horizon: 48, epochs: 2, minibatch: 256 });
    await trainer.init(tracks);
    const t0 = performance.now();
    // Pure-pursuit warm start, then PPO. From-scratch PPO at ~800 steps/s
    // does not finish laps before the wall penalty dominates; the warm start
    // is part of the in-browser recipe and finishes well inside 3 minutes.
    await trainer.imitate(4);
    await trainer.update();
    const completion = trainer.evalCompletion(8, 1000);
    expect(performance.now() - t0).toBeLessThan(180_000);
    expect(completion).toBeGreaterThanOrEqual(0.8);
  }, 180_000);

  it('swaps the track pool without changing weights', async () => {
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
    expect(trainer.env.trackId(0)).toBe(custom.id);
    expect(trainer.mix.entropy).toBe(0.02);
    expect(trainer.mix.weights[0]).toBeCloseTo(0.7, 2);
  });
});
