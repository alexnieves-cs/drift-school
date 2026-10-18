/** Re-run failing corner cases one by one and print what happened. */
import { cornerCases, evaluateBaseline, PHYS_SEEDS } from '../src/eval/physics';
import { withSurface } from '../src/eval/harness';
import { VecEnv } from '../src/sim/vecEnv';
import { generateTrack } from '../src/track/generate';
import type { Surface } from '../src/sim/dr';

const arg = (name: string, def: string) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? def;
const stage = Number(arg('stage', '1'));
const surface = arg('surface', 'ice') as Surface['name'];
const base = PHYS_SEEDS.slice(0, Number(arg('seeds', '16'))).map((seed) => generateTrack({ seed, stage }));
const tracks = base.map((t) => withSurface(t, surface));
const cases = cornerCases(tracks, stage, Number(arg('copies', '6')));
for (const [ci, c] of cases.entries()) {
  const r = evaluateBaseline([c]);
  if (r.completed) continue;
  const p = c.car;
  console.log(
    `case ${ci} ${c.track.id} ${JSON.stringify(r.reasons)} mass ${p.mass.toFixed(2)} cg ${p.cgHeight.toFixed(3)} mu0 ${p.tire.mu0} ` +
      `ls ${p.tire.loadSens.toFixed(3)} fB ${p.tire.front.B.toFixed(1)} fC ${p.tire.front.C.toFixed(2)} rB ${p.tire.rear.B.toFixed(1)} rC ${p.tire.rear.C.toFixed(2)} ` +
      `torque ${p.motor.torque.toFixed(3)} steerOff ${((p.steer.offset * 180) / Math.PI).toFixed(1)} rate ${p.steer.rate.toFixed(1)} k ${p.susp.k.toFixed(0)} c ${p.susp.c.toFixed(0)}`,
  );
  // replay with a trace
  const env = new VecEnv(1);
  env.reset([c.track], [{ stage, car: c.car, noiseScale: c.noiseScale, latencyTicks: c.latencyTicks, trackId: c.track.id, lapsTarget: 1, seed: 90_000 }]);
  for (let t = 0; t < 4000; t++) {
    const s0 = env.snapshot(0)!;
    const res = env.step(env.baselineActions());
    if (t % 25 === 0 || res.done[0] || res.trunc[0])
      console.log(
        `  t=${(t * 0.02).toFixed(1)} s=${s0.s.toFixed(1)}/${c.track.length.toFixed(0)} v=${s0.speed.toFixed(2)} yawRate=${s0.yawRate.toFixed(2)} slipF=${((s0.slip[0]! * 180) / Math.PI).toFixed(0)} slipR=${((s0.slip[2]! * 180) / Math.PI).toFixed(0)} thr=${s0.throttle.toFixed(2)} ${res.done[0] ? 'DONE' : ''}${res.trunc[0] ? 'TRUNC' : ''}`,
      );
    if (res.done[0] || res.trunc[0]) break;
  }
}
