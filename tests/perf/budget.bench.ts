import { describe, expect, it } from 'vitest';
import { Vehicle } from '../../src/sim/car/vehicle';
import { defaultCarParams } from '../../src/sim/car/params';
import { VecEnv } from '../../src/sim/vecEnv';
import { generateTrack } from '../../src/track/generate';

describe('perf budget', () => {
  it('steps one car in under 0.15 ms', () => {
    const car = new Vehicle(defaultCarParams());
    const n = 2000;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) car.step(((i % 20) - 10) / 10, 0.7, 0.02);
    const per = (performance.now() - t0) / n;
    expect(per).toBeLessThan(0.15);
  });

  it('vectorized env clears 6k car-steps/s at n=32', () => {
    const tracks = [1, 2, 3, 4].map((seed) => generateTrack({ seed, stage: 0 }));
    const env = new VecEnv(32);
    const params = defaultCarParams();
    env.reset(
      Array.from({ length: 32 }, (_, i) => tracks[i % tracks.length]!),
      Array.from({ length: 32 }, (_, i) => ({
        stage: 0,
        car: params,
        noiseScale: 0,
        latencyTicks: 0,
        trackId: tracks[i % tracks.length]!.id,
      })),
    );
    const actions = new Float32Array(32 * 2);
    for (let i = 0; i < 32; i++) actions[i * 2 + 1] = 0.6;
    const batches = 40;
    const t0 = performance.now();
    for (let i = 0; i < batches; i++) env.step(actions);
    const steps = batches * 32;
    const rate = steps / ((performance.now() - t0) / 1000);
    // Single process. The spec's 20k/s on 4 workers is not met: there is no SAB farm.
    expect(rate).toBeGreaterThan(6000);
  });
});
