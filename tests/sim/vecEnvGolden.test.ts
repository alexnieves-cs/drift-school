import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { record, scenarios } from './golden';

const golden = JSON.parse(readFileSync(new URL('./fixtures/vecenv-golden.json', import.meta.url), 'utf8')) as Record<
  string,
  { hashes: number[]; final: number[] }
>;

describe('VecEnv trajectory is unchanged by optimization', () => {
  for (const sc of scenarios()) {
    it(`${sc.name}: every step hash and the final pose match the pre-optimization recording`, () => {
      const want = golden[sc.name];
      expect(want, `no recording for ${sc.name}`).toBeDefined();
      const got = record(sc);
      expect(got.hashes.length).toBe(want!.hashes.length);
      const first = got.hashes.findIndex((h, i) => h !== want!.hashes[i]);
      expect(first, `first diverging control step in ${sc.name}`).toBe(-1);
      expect(got.final).toEqual(want!.final);
    });
  }

  it('records are reproducible run to run', () => {
    const sc = scenarios()[1]!;
    expect(record(sc).hashes).toEqual(record(sc).hashes);
  });
});
