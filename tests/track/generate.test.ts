import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { generateTrack, generatorStats } from '../../src/track/generate';
import { buildTrackFromPoints, validateSpec, validateTrack } from '../../src/track/build';
import { fnv1a } from '../../src/shared/hash';
import type { TrackSpec } from '../../src/shared/types';
import type { InvalidReason } from '../../src/shared/types';

/** Full-precision digest of everything in a TrackSpec, typed arrays included. */
function specDigest(t: TrackSpec): string {
  return fnv1a(
    JSON.stringify(t, (_k, v: unknown) => (v instanceof Float32Array ? Array.from(v) : v)),
  );
}

describe('track generator', () => {
  it('produces valid tracks for 1000 seeds across stages 0-4, without using the fallback', () => {
    const before = generatorStats.fallbacks;
    const failures: string[] = [];
    for (let seed = 0; seed < 1000; seed++) {
      for (let stage = 0; stage < 5; stage++) {
        const a = generateTrack({ seed, stage });
        // One expect per track: a million per-sample expects made this take two minutes.
        let finite = true;
        let maxK = 0;
        for (const s of a.samples) {
          finite &&= Number.isFinite(s.x + s.z + s.curvature + s.s);
          maxK = Math.max(maxK, Math.abs(s.curvature));
        }
        const checked = validateSpec(a);
        const bad =
          !finite ||
          a.length < 15 ||
          a.length > 150 ||
          a.samples.length <= 10 ||
          !(a.tRef > 1) ||
          a.walls.length !== 2 ||
          a.gates.length <= 3 ||
          maxK >= 1 / 0.34 ||
          !checked.ok;
        if (bad) failures.push(`seed ${seed} stage ${stage}: ${checked.issues.map((i) => i.reason).join(',') || 'shape'}`);
      }
    }
    expect(failures.slice(0, 5)).toEqual([]);
    expect(generatorStats.fallbacks - before).toBe(0);
  });

  it('gives a byte-identical TrackSpec for the same seed, including edges, walls, gates and tRef', () => {
    for (let seed = 0; seed < 100; seed++) {
      for (let stage = 0; stage < 5; stage++) {
        const a = generateTrack({ seed, stage });
        const b = generateTrack({ seed, stage });
        expect(specDigest(a), `seed ${seed} stage ${stage}`).toBe(specDigest(b));
        expect(Buffer.from(a.leftEdge.buffer).equals(Buffer.from(b.leftEdge.buffer))).toBe(true);
      }
    }
  });

  it('closes the wall loops so no gap is left at the start line', () => {
    const t = generateTrack({ seed: 3, stage: 2 });
    for (const wall of t.walls) {
      expect(wall[wall.length - 2]).toBe(wall[0]);
      expect(wall[wall.length - 1]).toBe(wall[1]);
    }
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
