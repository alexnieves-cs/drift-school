import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { decodeTrackHash, encodeTrackHash, quantizePoints } from '../../src/editor/share';
import { processStroke, sCurveStroke } from '../../src/editor/pipeline';
import { toFloat } from '../../src/editor/stroke';
import { buildTrackFromPoints, validateTrack } from '../../src/track/build';

describe('track share hash', () => {
  it('round-trips control points within 1 mm', () => {
    const points = new Float32Array([0, 0, 12.3456, -3.2, 8.1, 9.876, -4.25, 6, 0.4, 0.1]);
    const hash = encodeTrackHash(points, 1.25, 0.9);
    const back = decodeTrackHash(hash);
    expect(back.halfWidth).toBeCloseTo(1.25, 5);
    expect(back.mu).toBeCloseTo(0.9, 5);
    expect(back.points.length).toBe(points.length);
    for (let i = 0; i < points.length; i++) expect(Math.abs(back.points[i]! - points[i]!)).toBeLessThanOrEqual(0.001);
  });
});

/** Median of `runs` timings after `warmup` untimed calls: a live edit runs on a warm JIT, the first call of a process does not. */
function warmMedianMs(fn: () => unknown, warmup = 3, runs = 7): number {
  for (let i = 0; i < warmup; i++) fn();
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }
  return times.sort((a, b) => a - b)[Math.floor(runs / 2)]!;
}

describe('editor validate budget', () => {
  it('checks the oval fixture in under 16 ms', () => {
    const oval = JSON.parse(readFileSync('tests/fixtures/tracks-good/oval.json', 'utf8')) as {
      controlPoints: number[];
      halfWidth: number;
    };
    const flat = Float32Array.from(oval.controlPoints);
    expect(validateTrack(flat, oval.halfWidth).ok).toBe(true);
    expect(warmMedianMs(() => validateTrack(flat, oval.halfWidth))).toBeLessThan(16);
  });

  it('validates and splines the drawn S-curve (one live edit) in under 16 ms', () => {
    const points = quantizePoints(toFloat(processStroke(sCurveStroke(), 1, 1.05).points));
    const edit = () => {
      validateTrack(points, 1);
      return buildTrackFromPoints(points, { halfWidth: 1, mu: 1.05, source: 'user' });
    };
    expect(validateTrack(points, 1).ok).toBe(true);
    expect(warmMedianMs(edit)).toBeLessThan(16);
  });
});
