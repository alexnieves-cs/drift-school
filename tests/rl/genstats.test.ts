import { describe, expect, it } from 'vitest';
import { aggregateGen, type EpRecord } from '../../src/rl/genStats';

const ep = (over: Partial<EpRecord>): EpRecord => ({ lap: false, crash: false, ret: 0, trackId: 't', tRef: 10, speed: 3, ...over });

describe('generation stats', () => {
  it('says how many episodes finished, so empty updates are not read as 0% completion', () => {
    expect(aggregateGen(1, [], 't').episodes).toBe(0);
    expect(aggregateGen(2, [ep({}), ep({ lap: true, lapTime: 9 })], 't').episodes).toBe(2);
  });

  it('reports completion, crash rate and lap times', () => {
    const stat = aggregateGen(3, [ep({ lap: true, lapTime: 8 }), ep({ lap: true, lapTime: 10 }), ep({ crash: true }), ep({})], 't');
    expect(stat.completion).toBe(0.5);
    expect(stat.crashRate).toBe(0.25);
    expect(stat.bestLap).toBe(8);
    expect(stat.meanLap).toBe(9);
  });
});
