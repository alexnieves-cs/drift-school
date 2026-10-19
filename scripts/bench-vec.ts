/** Single-process VecEnv throughput. `npx vite-node scripts/bench-vec.ts [n] [batches] [repeats]` */
import { defaultCarParams } from '../src/sim/car/params';
import { VecEnv } from '../src/sim/vecEnv';
import { generateTrack } from '../src/track/generate';

const n = Number(process.argv[2] ?? 32);
const batches = Number(process.argv[3] ?? 300);
const repeats = Number(process.argv[4] ?? 5);
const tracks = [1, 2, 3, 4].flatMap((seed) => [0, 2, 3].map((stage) => generateTrack({ seed, stage })));
const params = defaultCarParams();
const rates: number[] = [];
for (let r = 0; r < repeats; r++) {
  const env = new VecEnv(n);
  env.reset(
    Array.from({ length: n }, (_, i) => tracks[i % tracks.length]!),
    Array.from({ length: n }, (_, i) => ({
      stage: 0,
      car: params,
      noiseScale: 0,
      latencyTicks: 0,
      trackId: tracks[i % tracks.length]!.id,
    })),
  );
  const act = new Float32Array(n * 2);
  const t0 = performance.now();
  for (let b = 0; b < batches; b++) {
    // pure-pursuit driver on a mix of stages: realistic, few early resets
    act.set(env.baselineActions());
    env.step(act);
  }
  rates.push((n * batches) / ((performance.now() - t0) / 1000));
}
rates.sort((a, b) => a - b);
console.log(
  `n=${n} rates (sorted): ${rates.map((x) => x.toFixed(0)).join(' ')}  median ${rates[Math.floor(rates.length / 2)]!.toFixed(0)} car-steps/s`,
);
