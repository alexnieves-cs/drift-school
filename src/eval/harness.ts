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
  /**
   * Per case, in case order: the furthest forward point along the centerline the car reached
   * before its episode ended, as a fraction of one lap (0 to 1; 1 when the lap was completed).
   * Measured passively from `info.progress`; it does not change how the episode is stepped.
   */
  lapFractions: number[];
  /**
   * Laps finished with no wall contact at any step of the episode (`info.wallHit` never set).
   * Stage-0 tracks are wide enough that a car with the steering centred can bounce off the walls
   * all the way round, so plain completion does not show that a policy drives (NOTES/ppo-only-prereg.md).
   */
  cleanLaps: number;
  /** Per case, in case order: control steps with wall contact before the episode ended. */
  contactSteps: number[];
  /**
   * Per case, in case order: like `lapFractions`, but only counting the road covered before the
   * first wall contact (1 for a clean lap). A car that rides the walls stops scoring at its first touch.
   */
  cleanFractions: number[];
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
  const lapFractions = new Array<number>(cases.length).fill(0);
  const contactSteps = new Array<number>(cases.length).fill(0);
  const cleanFractions = new Array<number>(cases.length).fill(0);
  let cleanLaps = 0;
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
    // Forward progress per case, unwrapped across the start line.
    const prevS = chunk.map((c) => c.track.samples[c.track.startIndex]?.s ?? 0);
    const covered = new Array<number>(chunk.length).fill(0);
    const best = new Array<number>(chunk.length).fill(0);
    const cleanBest = new Array<number>(chunk.length).fill(0);
    let open = chunk.length;
    const cap = Math.max(...chunk.map((c) => Math.round((c.track.length / 1.5) * 1.8 * 50))) + 50;
    for (let t = 0; t < cap && open > 0; t++) {
      const action = act(obs, chunk.length);
      const stepped = env.step(action);
      obs.set(stepped.obs);
      for (let i = 0; i < chunk.length; i++) {
        if (settled[i]) continue;
        const length = chunk[i]!.track.length;
        let ds = stepped.infos[i]!.progress - prevS[i]!;
        if (ds < -length / 2) ds += length;
        else if (ds > length / 2) ds -= length;
        prevS[i] = stepped.infos[i]!.progress;
        covered[i] = covered[i]! + ds;
        best[i] = Math.max(best[i]!, covered[i]!);
        if (stepped.infos[i]!.wallHit) contactSteps[start + i] = contactSteps[start + i]! + 1;
        if (contactSteps[start + i] === 0) cleanBest[i] = best[i]!;
        if (stepped.infos[i]!.lapDone) {
          settled[i] = true;
          open--;
          completed++;
          if (contactSteps[start + i] === 0) cleanLaps++;
          lapFractions[start + i] = 1;
          cleanFractions[start + i] = contactSteps[start + i] === 0 ? 1 : Math.min(1, Math.max(0, cleanBest[i]! / length));
          const lt = stepped.infos[i]!.lapTime;
          if (lt) lapTimes.push(lt);
        } else if (stepped.done[i] || stepped.trunc[i]) {
          settled[i] = true;
          open--;
          lapFractions[start + i] = Math.min(1, Math.max(0, best[i]! / length));
          cleanFractions[start + i] = Math.min(1, Math.max(0, cleanBest[i]! / length));
        }
      }
    }
    // Cases still running at the step cap: count the distance they reached.
    for (let i = 0; i < chunk.length; i++) {
      if (settled[i]) continue;
      const length = chunk[i]!.track.length;
      lapFractions[start + i] = Math.min(1, Math.max(0, best[i]! / length));
      cleanFractions[start + i] = Math.min(1, Math.max(0, cleanBest[i]! / length));
    }
  }
  return { completed, total: cases.length, rate: cases.length ? completed / cases.length : 0, lapTimes, lapFractions, cleanLaps, contactSteps, cleanFractions };
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

