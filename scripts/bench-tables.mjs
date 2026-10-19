#!/usr/bin/env node
// Prints the Markdown tables in docs/BENCHMARKS.md from the JSON that the other scripts wrote.
//   npm run eval && cp eval-report.json docs/media/eval.json
//   npm run perf > docs/media/perf.json
//   node scripts/record-demos.mjs hairpin race draw
//   node scripts/bench-tables.mjs
import { existsSync, readFileSync } from 'node:fs';

const read = (f) => (existsSync(`docs/media/${f}`) ? JSON.parse(readFileSync(`docs/media/${f}`, 'utf8')) : null);
const out = [];
const table = (head, rows) => {
  out.push(`| ${head.join(' | ')} |`, `| ${head.map(() => '---').join(' | ')} |`, ...rows.map((r) => `| ${r.join(' | ')} |`), '');
};
const f = (v, d = 2) => (v == null ? '-' : Number(v).toFixed(d));

const evalReport = read('eval.json');
if (evalReport) {
  out.push('### Acceptance table (`npm run eval`)', '');
  table(['', 'Test', 'Target', 'Measured', 'Result'],
    evalReport.rows.map((r) => [r.milestone, r.test, r.target, r.measured, r.pass === null ? 'skip' : r.pass ? 'PASS' : 'FAIL']));
}

const perf = read('perf.json');
if (perf) {
  out.push('### Throughput (`npm run perf`, one Node process, warmed up)', '');
  table(['Measure', 'Result', 'SPEC budget'], [
    ['Vehicle step (10 substeps)', `${f(perf.vehicleStepMsMedian, 4)} ms median`, '< 0.15 ms'],
    ...Object.entries(perf.vecEnvStepsPerSec).map(([k, v]) => [k.replace(' car-steps/s', ''), `${v.toLocaleString('en-US')} car-steps/s`, '>= 6,000 per worker']),
    ['Track generation', `${f(perf.trackGenMsMedian)} ms median`, '< 20 ms'],
    ['Validate the oval fixture', `${f(perf.validateOvalMsMedian)} ms median`, '< 16 ms'],
  ]);
}

const hp = read('hairpin-pov-results.json');
if (hp) {
  out.push('### Pure-pursuit baseline on the hairpin loop (real vehicle model)', '');
  table(['Grip mu', 'Finished', 'Lap time', 'Max offset from centerline', 'Top speed'],
    hp.map((r) => [f(r.mu, 2), r.finished ? 'yes' : 'no', `${f(r.lapTime, 1)} s`, `${f(r.maxOffset)} m`, `${f(r.topSpeed)} m/s`]));
}

const race = read('race-run.json');
if (race) {
  const laps = race.generations.filter((g) => g.best != null);
  const med = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const k = Math.min(5, Math.floor(laps.length / 2));
  out.push(`### Live training, ${race.cars} cars (\`node scripts/record-demos.mjs race\`)`, '');
  table(['Measure', 'Value'], [
    ['Wall time / updates read', `${race.wallSeconds} s / ${race.generations.length}`],
    ['Updates in which a car finished a lap', `${laps.length}`],
    ['Best lap, first 5 such updates (median)', `${f(med(laps.slice(0, k).map((g) => g.best)))} s`],
    ['Best lap, last 5 such updates (median)', `${f(med(laps.slice(-k).map((g) => g.best)))} s`],
    ['Fastest lap seen', `${f(Math.min(...laps.map((g) => g.best)))} s`],
  ]);
}

const draw = read('draw-run.json');
if (draw) {
  const panels = draw.adaptation.filter((a) => a.panel?.phase && a.panel.phase !== 'idle');
  const last = panels.at(-1)?.panel;
  const firstTune = panels.find((a) => a.panel.phase === 'tune');
  const pct = (x) => (x === undefined ? 'not reached' : `${Math.round(Number(x) * 100)}%`);
  out.push('### Draw a track and train on it (`node scripts/record-demos.mjs draw`)', '');
  out.push(`Editor verdict: \`${draw.editor}\`.`, '');
  if (last) {
    table(['Measure', 'Value'], [
      ['Swap latency (click to every env on the drawn track)', last.latencyMs ? `${(Number(last.latencyMs) / 1000).toFixed(1)} s` : 'not reached'],
      ['Weights unchanged at swap', last.unchanged ?? 'not reported'],
      ['Zero-shot completion (no update yet)', pct(last.zeroShot)],
      ['Fine-tuning started after', firstTune ? `${firstTune.t} s` : 'not reached'],
      [`Completion on the drawn track at ${panels.at(-1).t} s`, pct(last.completion)],
      ['Best lap on the drawn track', last.bestLap ? `${last.bestLap} s` : 'none'],
      ['Learner steps since the swap', last.steps ?? '0'],
    ]);
    out.push(`Last panel text: \`${last.text}\`.`, '');
  }
}
console.log(out.join('\n'));
