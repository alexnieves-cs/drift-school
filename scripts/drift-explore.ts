/** Poke at drift behaviour under DR: `npx vite-node scripts/drift-explore.ts` */
import { makeRng } from '../src/shared/rng';
import { runDrift } from '../tests/sim/driftScript';
import { cloneCar, defaultCarParams } from '../src/sim/car/params';
import { driftSetup, sampleCar, type DRLevel } from '../src/sim/dr';

const show = (label: string, r: ReturnType<typeof runDrift>) =>
  console.log(
    `${label.padEnd(36)} peak ${r.peakDeg.toFixed(0).padStart(3)} slide ${r.slidingSeconds.toFixed(2)}s minV ${r.minSpeedWhileSliding.toFixed(1)} reached ${r.reached} recovered ${r.recovered} (${r.recoverSeconds.toFixed(2)}s) vEnd ${r.finalSpeed.toFixed(1)} ${r.note}`,
  );

const d = driftSetup(defaultCarParams());
show('drift setup, 4.5 m/s, steer .6', runDrift(d));
show('drift setup, 2.0 m/s, steer .6', runDrift(d, { v0: 2 }));
show('drift setup, 4.5 m/s, steer .15', runDrift(d, { entrySteer: 0.15 }));
show('default 4wd, 4.5 m/s, steer .6', runDrift(defaultCarParams()));
const p0 = cloneCar(d);
p0.tire.rearMuScale = 0.42;
p0.tire.front = { B: 10, C: 1.6, E: 0.9 };
p0.tire.rear = { B: 4.2, C: 1.35, E: 0.5 };
p0.tire.mu0 = 1.05;
show('existing test params', runDrift(p0));

for (const dr of ['off', 'low', 'high'] as DRLevel[]) {
  for (const rms of [1, 0.95, 0.92, 0.9, 0.87, 0.85, 0.8, 0.7]) {
    const base = cloneCar(d);
    base.tire.rearMuScale = rms;
    const why: Record<string, number> = {};
    const N = dr === 'off' ? 1 : 100;
    for (let i = 0; i < N; i++) {
      const r = runDrift(dr === 'off' ? base : sampleCar(base, makeRng(i, `drift/${rms}`), dr, 1));
      why[r.note] = (why[r.note] ?? 0) + 1;
    }
    console.log(`${dr.padEnd(5)} rms ${rms}`, JSON.stringify(why));
  }
}
