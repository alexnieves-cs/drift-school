import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { generateTrack, trackFingerprint } from '../../src/track/generate';
import { buildTrackFromPoints, validateTrack } from '../../src/track/build';
import type { InvalidReason } from '../../src/shared/types';

describe('track generator', () => {
  it('is valid for 1000 seeds across stages 0-4 and deterministic', () => {
    for (let seed = 0; seed < 1000; seed++) {
      for (let stage = 0; stage < 5; stage++) {
        const a = generateTrack({ seed, stage });
        const b = generateTrack({ seed, stage });
        expect(a.length).toBeGreaterThanOrEqual(15);
        expect(a.length).toBeLessThanOrEqual(150);
        expect(a.samples.length).toBeGreaterThan(10);
        expect(trackFingerprint(a)).toBe(trackFingerprint(b));
        expect(Number.isFinite(a.tRef)).toBe(true);
        expect(a.tRef).toBeGreaterThan(1);
        expect(a.walls.length).toBe(2);
        expect(a.gates.length).toBeGreaterThan(3);
        let maxK = 0;
        for (const s of a.samples) {
          expect(Number.isFinite(s.x + s.z + s.curvature)).toBe(true);
          maxK = Math.max(maxK, Math.abs(s.curvature));
        }
        expect(maxK).toBeLessThan(1 / 0.34);
      }
    }
  });

  it('generates a stage-0 track in under 20 ms', () => {
    generateTrack({ seed: 1, stage: 0 });
    const t0 = performance.now();
    const n = 20;
    for (let i = 0; i < n; i++) generateTrack({ seed: 100 + i, stage: i % 5 });
    const ms = (performance.now() - t0) / n;
    expect(ms).toBeLessThan(20);
  });

  it('rejects bad fixtures with the expected reason and accepts the oval', () => {
    const good = JSON.parse(readFileSync('tests/fixtures/tracks-good/oval.json', 'utf8')) as {
      controlPoints: number[];
      halfWidth: number;
      length: number;
    };
    expect(good.length).toBeGreaterThan(28);
    expect(good.length).toBeLessThan(32);
    const built = buildTrackFromPoints(Float32Array.from(good.controlPoints), { halfWidth: good.halfWidth, mu: 1.1 });
    expect(built.length).toBeGreaterThan(15);
    const goodVal = validateTrack(Float32Array.from(good.controlPoints), good.halfWidth);
    expect(goodVal.ok, JSON.stringify(goodVal.issues)).toBe(true);

    const cases: [string, InvalidReason][] = [
      ['selfIntersect', 'selfIntersect'],
      ['minRadius', 'minRadius'],
      ['tooShort', 'tooShort'],
      ['edgeOverlap', 'edgeOverlap'],
    ];
    for (const [name, reason] of cases) {
      const fix = JSON.parse(readFileSync(`tests/fixtures/tracks-bad/${name}.json`, 'utf8')) as {
        halfWidth: number;
        points: number[];
      };
      const val = validateTrack(Float32Array.from(fix.points), fix.halfWidth);
      expect(val.ok, name).toBe(false);
      expect(val.issues.some((iss) => iss.reason === reason), `${name}: ${JSON.stringify(val.issues)}`).toBe(true);
    }
  });
});
