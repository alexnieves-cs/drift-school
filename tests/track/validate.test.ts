import { describe, expect, it } from 'vitest';
import { buildTrackFromPoints, validateSpec, validateTrack } from '../../src/track/build';
import { generateTrack } from '../../src/track/generate';

function circle(r: number, n = 24, start = 0, cx = 0, cz = 0): number[] {
  const out: number[] = [];
  for (let k = 0; k < n; k++) {
    const a = ((k + start) / n) * Math.PI * 2;
    out.push(cx + r * Math.cos(a), cz + r * Math.sin(a));
  }
  out.push(out[0]!, out[1]!);
  return out;
}

/** Two straights joined by half-circle ends. `start` rotates which point the stroke begins at. */
function stadium(half: number, r: number, perEnd = 12, start = 0): number[] {
  const pts: [number, number][] = [];
  for (let k = 0; k <= perEnd; k++) {
    const a = -Math.PI / 2 + (k / perEnd) * Math.PI;
    pts.push([half + r * Math.cos(a), r * Math.sin(a)]);
  }
  for (let k = 0; k <= perEnd; k++) {
    const a = Math.PI / 2 + (k / perEnd) * Math.PI;
    pts.push([-half + r * Math.cos(a), r * Math.sin(a)]);
  }
  const rot = pts.slice(start).concat(pts.slice(0, start));
  const out = rot.flatMap(([x, z]) => [x, z]);
  out.push(out[0]!, out[1]!);
  return out;
}

const reasons = (pts: number[], hw = 1) => validateTrack(pts, hw).issues.map((i) => i.reason);

describe('track validation', () => {
  it('accepts a loop with a straight to start on', () => {
    expect(validateTrack(stadium(8, 4), 1).ok).toBe(true);
  });

  it('rejects a loop with no straight stretch to start on', () => {
    expect(reasons(circle(8))).toContain('startStraight');
  });

  it('rejects a stroke that does not return to its start', () => {
    const open = circle(8).slice(0, 30);
    expect(reasons(open)).toContain('open');
  });

  it('rejects a loop that crosses itself', () => {
    const eight: number[] = [];
    for (let k = 0; k < 40; k++) {
      const a = (k / 40) * Math.PI * 2;
      eight.push(10 * Math.sin(a), 6 * Math.sin(2 * a));
    }
    eight.push(eight[0]!, eight[1]!);
    expect(validateTrack(eight, 0.8).ok).toBe(false);
  });

  it('rejects loops that are too long or too far from the origin', () => {
    expect(reasons(circle(40, 48), 1)).toContain('tooLong');
    expect(reasons(circle(8, 24, 0, 400, 0), 1)).toContain('bounds');
  });

  it('rejects a loop too tight for its width', () => {
    expect(validateTrack(circle(1.5), 1.2).ok).toBe(false);
  });

  it('gives the same verdict wherever the stroke started', () => {
    const verdicts = [0, 5, 11, 17].map((start) => validateTrack(stadium(8, 4, 12, start), 1).ok);
    expect(new Set(verdicts).size).toBe(1);
  });

  it('measures the same length in validation and in the built track', () => {
    const pts = stadium(9, 4.5);
    const track = buildTrackFromPoints(pts, { halfWidth: 1, mu: 1, source: 'user', id: 'v' });
    expect(validateTrack(pts, 1).ok).toBe(true);
    expect(validateSpec(track).ok).toBe(true);
    const exact = 4 * 9 + 2 * Math.PI * 4.5;
    expect(track.length).toBeGreaterThan(exact * 0.9);
    expect(track.length).toBeLessThan(exact * 1.1);
  });

  it('refuses to build from points the guard rejects', () => {
    expect(() => buildTrackFromPoints([0, 0, NaN, 1, 2, 2], { halfWidth: 1, mu: 1, source: 'user', id: 'bad' })).toThrow();
  });

  it('agrees that generated tracks are valid', () => {
    for (const seed of [1, 2, 3, 4, 5]) expect(validateSpec(generateTrack({ seed, stage: 0 })).ok).toBe(true);
  });
});
