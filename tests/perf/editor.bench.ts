import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { validateTrack } from '../../src/track/build';
import { generateTrack } from '../../src/track/generate';
import { SLACK, bestOf } from '../helpers/perf';

/** SPEC: validity check + spline under 16 ms per live edit, and track generation under 20 ms. */
describe('editor and generator budgets', () => {
  it('validates the oval fixture in under 16 ms', () => {
    const oval = JSON.parse(readFileSync('tests/fixtures/tracks-good/oval.json', 'utf8')) as {
      controlPoints: number[];
      halfWidth: number;
    };
    const flat = Float32Array.from(oval.controlPoints);
    let ok = false;
    const ms = bestOf(() => {
      ok = validateTrack(flat, oval.halfWidth).ok;
    });
    expect(ok).toBe(true);
    expect(ms).toBeLessThan(16 * SLACK);
  });

  it('generates a track in under 20 ms (median over 100 tracks, all stages)', () => {
    for (let i = 0; i < 10; i++) generateTrack({ seed: 9000 + i, stage: i % 5 });
    const times: number[] = [];
    for (let i = 0; i < 100; i++) {
      const t0 = performance.now();
      generateTrack({ seed: 100 + i, stage: i % 5 });
      times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    console.log(`generateTrack: median ${times[50]!.toFixed(1)} ms, p99 ${times[98]!.toFixed(1)} ms`);
    expect(times[50]!).toBeLessThan(20 * SLACK);
  });
});
