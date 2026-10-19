import type { CarParams, EpisodeConfig, TrackSpec } from '../shared/types';
import { makeRng } from '../shared/rng';
import { VecEnv } from '../sim/vecEnv';
import { defaultCarParams } from '../sim/car/params';
import { latencyFor, noiseFor, sampleCar, SURFACES, type DRLevel, type Surface } from '../sim/dr';

export interface EvalCase {
  track: TrackSpec;
  car: CarParams;
  noiseScale: number;
  latencyTicks: number;
  stage: number;
  label?: string;
}

export interface EvalOutcome {
  completed: number;
  total: number;
  rate: number;
  lapTimes: number[];
}

export type ActFn = (obs: Float32Array, n: number) => Float32Array;

/** Same track with a uniform surface friction. */
export function withSurface(track: TrackSpec, surface: Surface['name']): TrackSpec {
  const mu = SURFACES[surface];
  return {
    ...track,
    id: `${track.id}@${surface}`,
    samples: track.samples.map((s) => ({ ...s, mu })),
  };
}

/**
 * Deterministic one-lap evaluation. Each case is one episode; a case counts when the
 * car finishes a lap before it terminates or times out. Cases run in batches of `batch`.
 */
export function evaluateCases(act: ActFn, cases: EvalCase[], batch = 32): EvalOutcome {
  let completed = 0;
  const lapTimes: number[] = [];
  for (let start = 0; start < cases.length; start += batch) {
    const chunk = cases.slice(start, start + batch);
    const env = new VecEnv(chunk.length);
    const cfgs: EpisodeConfig[] = chunk.map((c, i) => ({
      stage: c.stage,
      car: c.car,
      noiseScale: c.noiseScale,
      latencyTicks: c.latencyTicks,
      trackId: c.track.id,
      lapsTarget: 1,
      seed: 90_000 + start + i,
    }));
    env.reset(
      chunk.map((c) => c.track),
      cfgs,
    );
    const obs = new Float32Array(env.obs);
    const settled = new Array<boolean>(chunk.length).fill(false);
    let open = chunk.length;
    const cap = Math.max(...chunk.map((c) => Math.round((c.track.length / 1.5) * 1.8 * 50))) + 50;
    for (let t = 0; t < cap && open > 0; t++) {
      const action = act(obs, chunk.length);
      const stepped = env.step(action);
      obs.set(stepped.obs);
      for (let i = 0; i < chunk.length; i++) {
        if (settled[i]) continue;
        if (stepped.infos[i]!.lapDone) {
          settled[i] = true;
          open--;
          completed++;
          const lt = stepped.infos[i]!.lapTime;
          if (lt) lapTimes.push(lt);
        } else if (stepped.done[i] || stepped.trunc[i]) {
          settled[i] = true;
          open--;
        }
      }
    }
  }
  return { completed, total: cases.length, rate: cases.length ? completed / cases.length : 0, lapTimes };
}

export function casesFor(
  tracks: TrackSpec[],
  opts: { dr?: DRLevel; seed?: number; stage?: number; copies?: number } = {},
): EvalCase[] {
  const dr = opts.dr ?? 'off';
  const copies = opts.copies ?? 1;
  const out: EvalCase[] = [];
  tracks.forEach((track, ti) => {
    for (let c = 0; c < copies; c++) {
      const rng = makeRng((opts.seed ?? 7) * 1009 + ti * 31 + c, `eval/${track.id}`);
      out.push({
        track,
        car: dr === 'off' ? defaultCarParams() : sampleCar(defaultCarParams(), rng, dr, 1),
        noiseScale: noiseFor(rng, dr),
        latencyTicks: latencyFor(rng, dr),
        stage: opts.stage ?? 0,
      });
    }
  });
  return out;
}

export function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}

