/**
 * Baseline completion matrix with knobs, for hunting physics problems.
 *   npx vite-node scripts/phys-matrix.ts -- --stages=1,2,3,4 --surfaces=asphalt,carpet,dirt,ice --drs=off,low,high --seeds=20 --copies=3
 */
import { baselineCorners, baselineMatrix, formatMatrix, PHYS_SEEDS } from '../src/eval/physics';
import type { DRLevel, Surface } from '../src/sim/dr';

const arg = (name: string, def: string) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? def;
const nSeeds = Number(arg('seeds', '8'));
const seeds = Array.from({ length: nSeeds }, (_, i) => PHYS_SEEDS[i] ?? 101 + i);
const ctl = process.argv.includes('--oracle') ? { gripAware: true, recover: true } : undefined;
const cells = baselineMatrix({
  ctl,
  setup: process.argv.includes('--drift') ? 'drift' : 'default',
  stages: arg('stages', '1,2,3').split(',').map(Number),
  surfaces: arg('surfaces', 'asphalt,carpet,dirt').split(',') as Surface['name'][],
  drs: arg('drs', 'low,high').split(',') as DRLevel[],
  seeds,
  copies: Number(arg('copies', '2')),
});
for (const l of formatMatrix(cells)) console.log(l);
const lt = cells.flatMap((c) => c.run.lapTimes);
console.log(`laps ${lt.length}, mean lap time ${(lt.reduce((a, b) => a + b, 0) / Math.max(1, lt.length)).toFixed(1)} s`);
console.log(`peak rear slip over all episodes ${Math.max(...cells.map((c) => c.run.peakRearSlipDeg)).toFixed(1)} deg`);

if (process.argv.includes('--corners')) {
  const surfaces = arg('surfaces', 'asphalt,carpet,dirt').split(',') as Surface['name'][];
  for (const c of baselineCorners({ ctl, stages: arg('stages', '1,2,3').split(',').map(Number), surfaces, seeds, copies: Number(arg('copies', '4')) })) {
    const r = c.run.reasons;
    console.log(`corners stage ${c.stage} ${c.surface.padEnd(8)} ${(c.run.rate * 100).toFixed(0)}% (${c.run.completed}/${c.run.total}) wall ${r.wall} off ${r.offTrack} wrong ${r.wrongWay} stuck ${r.stuck} flip ${r.flip} tmo ${r.timeout}`);
  }
}
