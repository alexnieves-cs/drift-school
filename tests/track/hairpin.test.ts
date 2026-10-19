import { describe, expect, it } from 'vitest';
import { HAIRPIN } from '../../src/drive/hairpin';
import { buildTrackFromPoints, validateTrack } from '../../src/track/build';
import { purePursuit } from '../../src/track/baseline';
import { nearestSample } from '../../src/track/sample';
import { Vehicle } from '../../src/sim/car/vehicle';
import { cloneCar, defaultCarParams } from '../../src/sim/car/params';
import { CONTROL_DT } from '../../src/shared/constants';

function lap(mu: number): { finished: boolean; time: number; maxOffset: number } {
  const track = buildTrackFromPoints(Float32Array.from(HAIRPIN), { halfWidth: 1, mu, source: 'user', id: `hairpin-${mu}` });
  const p = cloneCar(defaultCarParams());
  p.tire.mu0 = (1.15 * mu) / 1.1;
  const car = new Vehicle(p);
  const s0 = track.samples[track.startIndex]!;
  car.reset({ x: s0.x, z: s0.z, yaw: Math.atan2(s0.tz, s0.tx), speed: 1.2 });
  let hint = track.startIndex;
  let prevS = s0.s;
  let maxOffset = 0;
  for (let i = 0; i < 60 / CONTROL_DT; i++) {
    const cmd = purePursuit(car, track, hint);
    car.step(cmd.steer, cmd.throttle, CONTROL_DT);
    const q = nearestSample(track, car.px, car.pz, cmd.hint);
    hint = q.index;
    maxOffset = Math.max(maxOffset, Math.abs(q.lateral));
    if (q.sample.s < prevS - track.length * 0.5 && prevS > track.length * 0.4) {
      return { finished: true, time: (i + 1) * CONTROL_DT, maxOffset };
    }
    prevS = q.sample.s;
  }
  return { finished: false, time: 60, maxOffset };
}

describe('showcase hairpin loop', () => {
  it('is a valid track at the widths the app uses', () => {
    for (const halfWidth of [0.8, 1]) {
      expect(validateTrack(Float32Array.from(HAIRPIN), halfWidth).ok).toBe(true);
    }
  });

  it('is lapped by the pure-pursuit baseline on dry and icy grip, slower on ice', () => {
    const dry = lap(1.1);
    const ice = lap(0.4);
    expect(dry.finished).toBe(true);
    expect(ice.finished).toBe(true);
    expect(dry.maxOffset).toBeLessThan(1);
    expect(ice.maxOffset).toBeLessThan(1);
    expect(ice.time).toBeGreaterThan(dry.time);
  });
});
