import { describe, expect, it } from 'vitest';
import { makeRng } from '../../src/shared/rng';
import { sampleCar } from '../../src/sim/dr';
import { defaultCarParams } from '../../src/sim/car/params';
import { VecEnv } from '../../src/sim/vecEnv';
import { generateTrack } from '../../src/track/generate';

describe('surfaces and DR', () => {
  it('samples cars deterministically for a seed', () => {
    const base = defaultCarParams();
    const a = sampleCar(base, makeRng(9, 'dr'), 'high', 1);
    const b = sampleCar(base, makeRng(9, 'dr'), 'high', 1);
    expect(a.mass).toBe(b.mass);
    expect(a.tire.mu0).toBe(b.tire.mu0);
    expect(a.steer.offset).toBe(b.steer.offset);
  });

  it('reads segment mu onto the tire multiplier', () => {
    const track = generateTrack({ seed: 4, stage: 4 });
    const env = new VecEnv(1);
    env.reset(
      [track],
      [{ stage: 4, car: defaultCarParams(), noiseScale: 0, latencyTicks: 0, trackId: track.id }],
    );
    const before = track.samples[track.startIndex]?.mu ?? 1;
    env.step(new Float32Array([0, 0.5]));
    const snap = env.snapshot(0);
    expect(snap).not.toBeNull();
    expect(snap!.surfaceMu).toBeCloseTo(before / 1.05, 3);
  });
});
