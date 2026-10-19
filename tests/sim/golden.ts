import { makeRng } from '../../src/shared/rng';
import type { EpisodeConfig, TrackSpec } from '../../src/shared/types';
import { defaultCarParams } from '../../src/sim/car/params';
import { latencyFor, noiseFor, sampleCar, type DRLevel } from '../../src/sim/dr';
import { VecEnv } from '../../src/sim/vecEnv';
import { withSurface } from '../../src/eval/harness';
import { generateTrack } from '../../src/track/generate';

/**
 * Recorded VecEnv trajectories. The fixture in tests/sim/fixtures/vecenv-golden.json was
 * produced by the code before the throughput work; the test replays the same actions and
 * demands identical per-step hashes of (obs, reward, done, trunc, pose) for every env.
 */

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export class Fnv {
  h = 0x811c9dc5;
  num(x: number): void {
    f32[0] = x;
    this.h = Math.imul(this.h ^ u32[0]!, 0x01000193) >>> 0;
  }
  arr(a: ArrayLike<number>): void {
    for (let i = 0; i < a.length; i++) this.num(a[i]!);
  }
}

export interface Scenario {
  name: string;
  steps: number;
  /** 'random' is a seeded wild driver (walls, spins, resets); 'baseline' uses pure pursuit. */
  driver: 'random' | 'baseline';
  tracks: TrackSpec[];
  cfgs: EpisodeConfig[];
  seed: number;
}

export function scenarios(): Scenario[] {
  const out: Scenario[] = [];
  const mk = (
    name: string,
    driver: Scenario['driver'],
    stage: number,
    seeds: number[],
    surface: 'asphalt' | 'carpet' | 'dirt' | 'ice' | null,
    dr: DRLevel,
    steps: number,
  ) => {
    const tracks = seeds.map((s) => {
      const t = generateTrack({ seed: s, stage });
      return surface ? withSurface(t, surface) : t;
    });
    const cfgs: EpisodeConfig[] = tracks.map((t, i) => {
      const rng = makeRng(500 + i, `golden/${name}`);
      return {
        stage,
        car: dr === 'off' ? defaultCarParams() : sampleCar(defaultCarParams(), rng, dr, 1),
        noiseScale: noiseFor(rng, dr),
        latencyTicks: latencyFor(rng, dr),
        trackId: t.id,
        seed: 1000 + i,
      };
    });
    out.push({ name, steps, driver, tracks, cfgs, seed: 77 + out.length });
  };
  mk('random-s0-off', 'random', 0, [1, 2, 3, 4], null, 'off', 500);
  mk('random-s3-high-dirt', 'random', 3, [5, 6, 7, 8, 9, 10], 'dirt', 'high', 500);
  mk('baseline-s1-low-asphalt', 'baseline', 1, [11, 12, 13, 14], 'asphalt', 'low', 900);
  mk('baseline-s2-high-carpet', 'baseline', 2, [15, 16, 17, 18], 'carpet', 'high', 900);
  mk('baseline-s3-off', 'baseline', 3, [19, 20, 21, 22], null, 'off', 900);
  return out;
}

/** One 32-bit hash per control step (all envs together), plus a final full-precision pose. */
export function record(sc: Scenario): { hashes: number[]; final: number[] } {
  const n = sc.tracks.length;
  const env = new VecEnv(n);
  env.reset(sc.tracks, sc.cfgs);
  const rng = makeRng(sc.seed, `golden-act/${sc.name}`);
  const hashes: number[] = [];
  let act = new Float32Array(n * 2);
  for (let t = 0; t < sc.steps; t++) {
    if (sc.driver === 'baseline') act = env.baselineActions();
    else if (t % 12 === 0) {
      // Hold each random action for several steps so the car actually goes places.
      for (let i = 0; i < n; i++) {
        act[i * 2] = rng.range(-1, 1);
        act[i * 2 + 1] = rng.range(-0.3, 1);
      }
    }
    const r = env.step(act);
    const h = new Fnv();
    h.arr(r.obs);
    h.arr(r.rew);
    h.arr(r.done);
    h.arr(r.trunc);
    h.arr(env.poses());
    hashes.push(h.h);
  }
  const final: number[] = [];
  for (let i = 0; i < n; i++) {
    const s = env.snapshot(i)!;
    final.push(s.x, s.z, s.yaw, s.vx, s.vz, s.yawRate);
  }
  return { hashes, final };
}
