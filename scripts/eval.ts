/**
 * `npm run eval` prints the SPEC section 3 acceptance table.
 *
 *   npm run eval                     every section
 *   npm run eval -- --only=stage0    comma list of: stage0, matrix, hotswap, fps
 *   npm run eval -- --quick          smaller sample sizes and shorter training
 *
 * Exit code is 1 when any row fails. Rows that cannot run here print "skip".
 */
import { writeFileSync } from 'node:fs';
import type { EvalRow, Section } from '../src/eval/types';
import { stage0 } from '../src/eval/stage0';
import { matrix } from '../src/eval/matrix';
import { hotswap } from '../src/eval/hotswap';
import { fps } from '../src/eval/fps';

const SECTIONS: Record<string, Section> = { stage0, hotswap, fps, matrix };

const args = process.argv.slice(2);
const quick = args.includes('--quick');
const onlyArg = args.find((a) => a.startsWith('--only='));
const only = onlyArg ? onlyArg.slice('--only='.length).split(',') : Object.keys(SECTIONS);
for (const name of only) {
  if (!SECTIONS[name]) {
    console.error(`unknown section "${name}". Known: ${Object.keys(SECTIONS).join(', ')}`);
    process.exit(2);
  }
}

const log = (line: string) => console.error(`[eval] ${line}`);
const rows: EvalRow[] = [];
for (const name of only) {
  const t0 = performance.now();
  try {
    rows.push(...(await SECTIONS[name]!({ quick, log })));
  } catch (err) {
    rows.push({
      milestone: name,
      test: `section ${name} threw`,
      target: '-',
      measured: err instanceof Error ? err.message : String(err),
      pass: false,
    });
  }
  log(`${name} done in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

const mark = (p: boolean | null) => (p === null ? 'skip' : p ? 'PASS' : 'FAIL');
const cols = ['', 'Test', 'Target', 'Measured', 'Result'];
const body = rows.map((r) => [r.milestone, r.test, r.target, r.measured, mark(r.pass)]);
const width = cols.map((c, i) => Math.max(c.length, ...body.map((b) => b[i]!.length)));
const line = (cells: string[]) => cells.map((c, i) => c.padEnd(width[i]!)).join('  ');
console.log(line(cols));
console.log(width.map((w) => '-'.repeat(w)).join('  '));
for (const b of body) console.log(line(b));
const notes = rows.filter((r) => r.note);
if (notes.length) {
  console.log('');
  for (const r of notes) console.log(`${r.milestone} ${r.test}: ${r.note}`);
}
const failed = rows.filter((r) => r.pass === false).length;
const skipped = rows.filter((r) => r.pass === null).length;
console.log(`\n${rows.length - failed - skipped} pass, ${failed} fail, ${skipped} skip${quick ? ' (quick mode)' : ''}`);
writeFileSync('eval-report.json', JSON.stringify({ quick, rows }, null, 2));
process.exit(failed ? 1 : 0);
