import { describe, expect, it } from 'vitest';
import { generateTrack } from '../../src/track/generate';
import { simulateBaseline } from '../../src/track/baseline';

describe('baseline', () => {
  it('finishes generated stage 0-2 tracks', () => {
    for (let stage = 0; stage <= 2; stage++) {
      for (let seed = 1; seed <= 3; seed++) {
        const track = generateTrack({ seed, stage });
        const result = simulateBaseline(track, 80);
        expect(result.finished, `stage ${stage} seed ${seed} len ${track.length.toFixed(1)}`).toBe(true);
      }
    }
  }, 120_000);
});
