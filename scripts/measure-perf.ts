/**
 * Warmed-up throughput numbers for docs/BENCHMARKS.md.
 *   npx vite-node scripts/measure-perf.ts
 * Prints JSON. Single process, Node, no rendering.
 */
import { Vehicle } from '../src/sim/car/vehicle';
import { defaultCarParams } from '../src/sim/car/params';
import { VecEnv } from '../src/sim/vecEnv';
import { generateTrack } from '../src/track/generate';
import { validateTrack } from '../src/track/build';
import { CONTROL_DT } from '../src/shared/constants';
import { readFileSync } from 'node:fs';

function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}

function timeMs(fn: () => void, reps: number, warm: number): number[] {
  for (let i = 0; i < warm; i++) fn();
  const out: number[] = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    fn();
    out.push(performance.now() - t0);
  }
  return out;
}

const car = new Vehicle(defaultCarParams());
let k = 0;
const stepMs = timeMs(() => {
  for (let i = 0; i < 500; i++) car.step((((k++ % 20) - 10) / 10), 0.7, CONTROL_DT);
}, 20, 5).map((t) => t / 500);

const tracks = [1, 2, 3, 4].map((seed) => generateTrack({ seed, stage: 0 }));
const params = defaultCarParams();
const rates: Record<string, number> = {};
for (const n of [8, 32, 64]) {
  const env = new VecEnv(n);
  env.reset(
    Array.from({ length: n }, (_, i) => tracks[i % tracks.length]!),
    Array.from({ length: n }, (_, i) => ({ stage: 0, car: params, noiseScale: 0, latencyTicks: 0, trackId: tracks[i % tracks.length]!.id })),
  );
  const actions = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) actions[i * 2 + 1] = 0.6;
  const ms = timeMs(() => env.step(actions), 150, 30);
  rates[`vecEnv n=${n} car-steps/s`] = Math.round((n * 1000) / median(ms));
}

const gen = timeMs(() => { generateTrack({ seed: 7 + (k++ % 50), stage: 2 }); }, 100, 10);
const oval = JSON.parse(readFileSync('tests/fixtures/tracks-good/oval.json', 'utf8')) as { controlPoints: number[]; halfWidth: number };
const val = timeMs(() => { validateTrack(oval.controlPoints, oval.halfWidth); }, 100, 10);

console.log(JSON.stringify({
  node: process.version,
  vehicleStepMsMedian: Number(median(stepMs).toFixed(4)),
  vecEnvStepsPerSec: rates,
  trackGenMsMedian: Number(median(gen).toFixed(2)),
  validateOvalMsMedian: Number(median(val).toFixed(2)),
}, null, 2));
