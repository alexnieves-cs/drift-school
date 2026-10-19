/** Regenerates tests/sim/fixtures/vecenv-golden.json. Only run this on purpose: it redefines "same trajectory". */
import { writeFileSync } from 'node:fs';
import { record, scenarios } from '../tests/sim/golden';

const out: Record<string, { hashes: number[]; final: number[] }> = {};
for (const sc of scenarios()) {
  out[sc.name] = record(sc);
  console.error(`recorded ${sc.name}`);
}
writeFileSync('tests/sim/fixtures/vecenv-golden.json', JSON.stringify(out));
