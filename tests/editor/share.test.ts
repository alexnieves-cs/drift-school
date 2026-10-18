import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { decodeTrackHash, encodeTrackHash } from '../../src/editor/share';
import { validateTrack } from '../../src/track/build';

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

describe('editor validate budget', () => {
  it('checks the oval fixture in under 16 ms', () => {
    const oval = JSON.parse(readFileSync('tests/fixtures/tracks-good/oval.json', 'utf8')) as {
      controlPoints: number[];
      halfWidth: number;
    };
    const flat = Float32Array.from(oval.controlPoints);
    const t0 = performance.now();
    const result = validateTrack(flat, oval.halfWidth);
    expect(performance.now() - t0).toBeLessThan(16);
    expect(result.ok).toBe(true);
  });
});
