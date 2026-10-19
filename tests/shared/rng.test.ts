import { describe, expect, it } from 'vitest';
import { makeRng } from '../../src/shared/rng';

describe('rng', () => {
  it('is deterministic for a seed', () => {
    const a = makeRng(42);
    const b = makeRng(42);
    const seqA = Array.from({ length: 20 }, () => a.next());
    const seqB = Array.from({ length: 20 }, () => b.next());
    expect(seqA).toEqual(seqB);
  });

  it('fork(label) is stable and independent of draw order on the parent', () => {
    const a = makeRng(7);
    a.next();
    a.next();
    const f1 = a.fork('track');
    const b = makeRng(7);
    const f2 = b.fork('track');
    expect(Array.from({ length: 8 }, () => f1.next())).toEqual(Array.from({ length: 8 }, () => f2.next()));
    expect(f1.next()).not.toBe(makeRng(7).fork('other').next());
  });

  it('stays inside [0, 1)', () => {
    const rng = makeRng(1);
    for (let i = 0; i < 1000; i++) {
      const x = rng.next();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});
