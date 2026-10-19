import { describe, expect, it } from 'vitest';
import { makeRng } from '../../src/shared/rng';
import { cloneCar, defaultCarParams } from '../../src/sim/car/params';
import { driftSetup, sampleCar, type DRLevel } from '../../src/sim/dr';
import { runDrift } from './driftScript';

/**
 * SPEC section 3 / demo moment 4: low rear grip + throttle in a corner gives > 15 deg rear slip,
 * and countersteer brings the car back. The older test in vehicle.test.ts only checks the peak
 * slip and the recovery; its rear-grip setting (rearMuScale 0.42) actually spins the car to a
 * stop. These tests also demand that the car keeps moving while it slides (a drift, not a spin),
 * that the same script reports "grip" or "spun-out" when it should, and that it holds under DR.
 */

function drawFraction(level: DRLevel, n: number, tag: string) {
  const base = driftSetup(defaultCarParams());
  const notes: Record<string, number> = {};
  for (let i = 0; i < n; i++) {
    const r = runDrift(sampleCar(base, makeRng(i, `drift-test/${tag}`), level, 1));
    notes[r.note] = (notes[r.note] ?? 0) + 1;
  }
  return { ok: (notes.ok ?? 0) / n, notes };
}

describe('scripted drift', () => {
  it('drift setup slides past 15 deg rear slip at speed, then recovers', () => {
    const r = runDrift(driftSetup(defaultCarParams()));
    expect(r.peakDeg).toBeGreaterThan(15);
    expect(r.peakDeg).toBeLessThan(75);
    expect(r.slidingSeconds).toBeGreaterThanOrEqual(0.5);
    expect(r.minSpeedWhileSliding).toBeGreaterThanOrEqual(1.5);
    expect(r.recovered).toBe(true);
    expect(r.recoverSeconds).toBeLessThan(3);
    expect(r.finalSpeed).toBeGreaterThan(1);
  });

  it('the script tells grip from drift: a crawling corner on the stock 4WD tune never reaches 15 deg', () => {
    const r = runDrift(defaultCarParams(), { v0: 1.5, entrySteer: 0.3, cornerSeconds: 0.6 });
    expect(r.peakDeg).toBeLessThan(15);
    expect(r.note).toBe('grip');
  });

  it('the script tells drift from spin: the rear-grip setting of the older test spins out', () => {
    const p = cloneCar(driftSetup(defaultCarParams()));
    p.tire.mu0 = 1.05;
    p.tire.rearMuScale = 0.42;
    p.tire.front = { B: 10, C: 1.6, E: 0.9 };
    p.tire.rear = { B: 4.2, C: 1.35, E: 0.5 };
    const r = runDrift(p);
    expect(r.note).toBe('spun-out');
    // countersteer still catches it; it is the slide that is not a controlled one
    expect(r.recovered).toBe(true);
  });

  for (const level of ['low', 'high'] as DRLevel[]) {
    it(`reachable and recoverable under ${level} DR for >= 95% of 100 draws`, () => {
      const { ok, notes } = drawFraction(level, 100, level);
      expect(ok, JSON.stringify(notes)).toBeGreaterThanOrEqual(0.95);
    });
  }

  it('is deterministic', () => {
    const p = driftSetup(defaultCarParams());
    expect(runDrift(p)).toEqual(runDrift(p));
  });
});
