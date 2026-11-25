import { describe, expect, it } from 'vitest';
import { generateTrack } from '../../src/track/generate';
import { casesFor, evaluateCases } from '../../src/eval/harness';

const constant = (steer: number, throttle: number) => (_o: Float32Array, n: number) => {
  const a = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    a[i * 2] = steer;
    a[i * 2 + 1] = throttle;
  }
  return a;
};

describe('evaluateCases', () => {
  const tracks = [101, 104, 105].map((seed) => generateTrack({ seed, stage: 0 }));

  it('reports one lap fraction and one contact count per case', () => {
    const parked = evaluateCases(constant(0, 0), casesFor(tracks));
    expect(parked.lapFractions).toHaveLength(3);
    expect(parked.contactSteps).toHaveLength(3);
    expect(parked.completed).toBe(0);
    expect(parked.cleanLaps).toBe(0);
    for (const f of parked.lapFractions) {
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThan(0.05);
    }
  });

  it('a car with the steering centred rides the walls round a stage-0 lap, which only cleanLaps catches', () => {
    // Plain completion is fooled on stage 0: the tracks are wide and the walls bounce the car round.
    const straight = evaluateCases(constant(0, 0.5), casesFor(tracks));
    expect(straight.completed).toBe(3);
    expect(straight.lapFractions).toEqual([1, 1, 1]);
    expect(straight.cleanLaps).toBe(0);
    for (const c of straight.contactSteps) expect(c).toBeGreaterThan(0);
  });
});
