import { describe, expect, it } from 'vitest';
import { Trainer } from '../../src/rl/trainer';
import { generateTrack } from '../../src/track/generate';
import { defaultCarParams } from '../../src/sim/car/params';

const pool = () => [1, 2, 3, 4].map((seed) => generateTrack({ seed, stage: 0 }));

async function trained(seed: number): Promise<string> {
  const t = new Trainer({ n: 4, horizon: 16, epochs: 1, minibatch: 32, seed });
  await t.init(pool());
  await t.imitate(1);
  await t.update();
  return t.weightHash();
}

describe('trainer', () => {
  it('is reproducible: the same seed gives identical weights after imitation and an update', async () => {
    expect(await trained(7)).toBe(await trained(7));
    expect(await trained(7)).not.toBe(await trained(8));
  }, 120_000);

  it('applies the curriculum: a new stage regenerates the track pool', async () => {
    const t = new Trainer({ n: 4, horizon: 16, epochs: 1, minibatch: 32 });
    await t.init(pool());
    expect(t.mix.tracks.every((x) => x.id.endsWith('-s0'))).toBe(true);
    t.setStage(3);
    expect(t.mix.tracks.map((x) => x.id)).toEqual([1, 2, 3, 4].map((s) => `gen-${s}-s3`));
  });

  it('draws a fresh car config for every episode, so the DR and mass controls take effect', async () => {
    const t = new Trainer({ n: 4, horizon: 16, epochs: 1, minibatch: 32 });
    await t.init(pool());
    const base = defaultCarParams().mass;
    expect(t.env.sampler!(0).cfg.car.mass).toBeCloseTo(base, 6);
    t.massScale = 2;
    t.dr = 'high';
    const masses = Array.from({ length: 8 }, () => t.env.sampler!(0).cfg.car.mass);
    expect(Math.min(...masses)).toBeGreaterThan(base * 2 * 0.74);
    expect(new Set(masses.map((m) => m.toFixed(4))).size).toBeGreaterThan(4);
  });

  it('replays earlier stages 20% of the time', async () => {
    const t = new Trainer({ n: 4, horizon: 16, epochs: 1, minibatch: 32 });
    await t.init(pool());
    t.setStage(2);
    let earlier = 0;
    const N = 2000;
    for (let i = 0; i < N; i++) if (!t.env.sampler!(0).track.id.endsWith('-s2')) earlier++;
    expect(Math.abs(earlier / N - 0.2)).toBeLessThan(0.04);
  });

  it('reports adaptation stats on the drawn track after a swap', async () => {
    const t = new Trainer({ n: 4, horizon: 16, epochs: 1, minibatch: 32 });
    await t.init(pool());
    expect(t.adaptStats()).toBeNull();
    const user = generateTrack({ seed: 50, stage: 0 });
    t.queueTrackPool([{ ...user, id: 'user-x' }, ...pool()], true);
    t.forceSwap();
    expect(t.adaptStats()?.trackId).toBe('user-x');
    expect(t.adaptStats()?.updatesSinceSwap).toBe(0);
    await t.update();
    expect(t.adaptStats()?.updatesSinceSwap).toBe(1);
  }, 60_000);
});
