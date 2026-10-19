import { describe, expect, it } from 'vitest';
import { Vehicle } from '../../src/sim/car/vehicle';
import { defaultCarParams } from '../../src/sim/car/params';
import { VecEnv } from '../../src/sim/vecEnv';
import { generateTrack } from '../../src/track/generate';
import { SLACK, bestOf } from '../helpers/perf';

/** SPEC perf budget: a car env step (10 physics substeps, 21 rays, spline lookup) under 0.15 ms, so 6k steps/s per worker. */
describe('perf budget', () => {
  it('steps the vehicle model alone in under 0.15 ms', () => {
    const car = new Vehicle(defaultCarParams());
    const n = 2000;
    const ms = bestOf(() => {
      for (let i = 0; i < n; i++) car.step(((i % 20) - 10) / 10, 0.7, 0.02);
    });
    expect(ms / n).toBeLessThan(0.15 * SLACK);
  });

  it('runs the full env step (rays, track lookup, reward) under 0.15 ms per car at n=32', () => {
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
    const batches = 100;
    const ms = bestOf(() => {
      for (let i = 0; i < batches; i++) env.step(actions);
    });
    const perCar = ms / (batches * 32);
    console.log(`env step: ${perCar.toFixed(3)} ms per car, ${(1000 / perCar).toFixed(0)} car-steps/s (one process)`);
    // One process. SPEC's 20k/s total assumes four workers; there is no SharedArrayBuffer pool, so that is not claimed.
    expect(perCar).toBeLessThan(0.15 * SLACK);
  });
});
