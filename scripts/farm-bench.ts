/**
 * Env farm throughput: one VecEnv in this process against the farm with worker threads.
 * `npx vite-node scripts/farm-bench.ts -- [n=64] [steps=400] [workers=4]`
 *
 * Actions are fixed (steer 0, throttle 0.5) so the policy costs nothing and both sides step the
 * same cars. The machine's load average is printed with the result because other processes on a
 * shared machine change the numbers a lot.
 */
import { loadavg, cpus } from 'node:os';
import { defaultCarParams } from '../src/sim/car/params';
import { VecEnv } from '../src/sim/vecEnv';
import { generateTrack } from '../src/track/generate';
import { EnvFarm } from '../src/workers/envFarm';
import { nodeSpawn } from '../tests/helpers/nodeFarm';

const args = process.argv.slice(2).filter((a) => a !== '--');
const n = Number(args[0] ?? 64);
const steps = Number(args[1] ?? 400);
const workers = Number(args[2] ?? 4);

const shapes = [1, 2, 3, 4].flatMap((seed) => [0, 2, 3].map((stage) => generateTrack({ seed, stage })));
const tracks = Array.from({ length: n }, (_, i) => shapes[i % shapes.length]!);
const cfgs = tracks.map((t) => ({ stage: 0, car: defaultCarParams(), noiseScale: 1, latencyTicks: 0, trackId: t.id }));
const act = new Float32Array(n * 2);
for (let i = 0; i < n; i++) act[i * 2 + 1] = 0.5;

function single(): number {
  const env = new VecEnv(n);
  env.reset(tracks, cfgs);
  for (let s = 0; s < 50; s++) env.step(act);
  const t0 = performance.now();
  for (let s = 0; s < steps; s++) env.step(act);
  return (n * steps) / ((performance.now() - t0) / 1000);
}

async function farmRate(): Promise<number> {
  const farm = await EnvFarm.open({ total: n, shards: workers, spawn: await nodeSpawn() });
  try {
    await farm.reset(tracks, cfgs);
    for (let s = 0; s < 50; s++) farm.step(act);
    const t0 = performance.now();
    for (let s = 0; s < steps; s++) farm.step(act);
    return (n * steps) / ((performance.now() - t0) / 1000);
  } finally {
    await farm.close();
  }
}

const load0 = loadavg()[0]!;
const one = single();
const many = await farmRate();
const load1 = loadavg()[0]!;
console.log(
  JSON.stringify({
    n,
    steps,
    workers,
    cores: cpus().length,
    loadAvg1m: [Number(load0.toFixed(2)), Number(load1.toFixed(2))],
    singleProcess: Math.round(one),
    farm: Math.round(many),
    speedup: Number((many / one).toFixed(2)),
  }),
);
