import type { BaselineOpts } from '../track/baseline';
import type { EpisodeConfig, TrackSpec } from '../shared/types';
import { VecEnv } from '../sim/vecEnv';
import { makeRng } from '../shared/rng';
import { defaultCarParams } from '../sim/car/params';
import { driftSetup, latencyFor, noiseFor, sampleCar, sampleCarCorner, type DRLevel, type Surface } from '../sim/dr';
import { generateTrack } from '../track/generate';
import { casesFor, pct, withSurface, type EvalCase } from './harness';
import type { EvalRow, Section } from './types';

/**
 * Physics sanity: the pure-pursuit baseline (src/track/baseline.ts purePursuit) driven through
 * the same surface x DR matrix the learner is judged on. It is the upper-bound reference: if the
 * baseline cannot finish a cell, no policy can be expected to, and the cause is the controller or
 * the car model, not the learner.
 */

export type EndReason = 'lap' | 'wall' | 'offTrack' | 'wrongWay' | 'stuck' | 'flip' | 'timeout' | 'other';

export interface CaseResult {
  reason: EndReason;
  lapTime?: number;
  /** Peak rear slip angle over the episode, degrees. */
  peakRearSlipDeg: number;
}

export interface BaselineRun {
  completed: number;
  total: number;
  rate: number;
  reasons: Record<EndReason, number>;
  lapTimes: number[];
  peakRearSlipDeg: number;
}

const emptyReasons = (): Record<EndReason, number> => ({
  lap: 0,
  wall: 0,
  offTrack: 0,
  wrongWay: 0,
  stuck: 0,
  flip: 0,
  timeout: 0,
  other: 0,
});

/**
 * Same episode setup as `evaluateCases` in harness.ts (same cfg fields, per-case seed, step cap),
 * but the action comes from `VecEnv.baselineActions()` and every end is classified.
 */
export function evaluateBaseline(cases: EvalCase[], ctl?: BaselineOpts, batch = 32): BaselineRun {
  const reasons = emptyReasons();
  const lapTimes: number[] = [];
  let peakRear = 0;
  for (let start = 0; start < cases.length; start += batch) {
    const chunk = cases.slice(start, start + batch);
    const n = chunk.length;
    const env = new VecEnv(n);
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
    const settled = new Array<boolean>(n).fill(false);
    const lastRoll = new Float64Array(n);
    let open = n;
    const cap = Math.max(...chunk.map((c) => Math.round((c.track.length / 1.5) * 1.8 * 50))) + 50;
    for (let t = 0; t < cap && open > 0; t++) {
      for (let i = 0; i < n; i++) {
        const sn = env.snapshot(i)!;
        lastRoll[i] = Math.abs(sn.roll);
        if (!settled[i]) peakRear = Math.max(peakRear, (Math.abs(sn.slip[2]! + sn.slip[3]!) / 2) * (180 / Math.PI));
      }
      const stepped = env.step(env.baselineActions(ctl));
      for (let i = 0; i < n; i++) {
        if (settled[i]) continue;
        const info = stepped.infos[i]!;
        let why: EndReason | null = null;
        if (info.lapDone) {
          why = 'lap';
          if (info.lapTime) lapTimes.push(info.lapTime);
        } else if (stepped.done[i] || stepped.trunc[i]) {
          if (lastRoll[i]! > (70 * Math.PI) / 180) why = 'flip';
          else if (info.stuck) why = 'stuck';
          else if (info.wrongWay) why = 'wrongWay';
          else if (info.offTrack) why = 'offTrack';
          else if (info.wallHit) why = 'wall';
          else if (stepped.trunc[i]) why = 'timeout';
          else why = 'other';
        }
        if (why) {
          settled[i] = true;
          open--;
          reasons[why]++;
        }
      }
    }
    // anything still open ran out of steps
    reasons.timeout += open;
  }
  const completed = reasons.lap;
  return {
    completed,
    total: cases.length,
    rate: cases.length ? completed / cases.length : 0,
    reasons,
    lapTimes,
    peakRearSlipDeg: peakRear,
  };
}

export interface MatrixCell {
  stage: number;
  surface: Surface['name'];
  dr: DRLevel;
  run: BaselineRun;
}

export interface MatrixOpts {
  stages?: number[];
  surfaces?: Surface['name'][];
  drs?: DRLevel[];
  /** Track seeds per stage. */
  seeds?: number[];
  /** DR draws per track seed. */
  copies?: number;
  /** Controller variant; none is the imitation teacher. */
  ctl?: BaselineOpts;
  /** Car setup: the default 4WD tune or the drift-favorable RWD one. */
  setup?: 'default' | 'drift';
}

export const MATRIX_SURFACES: Surface['name'][] = ['asphalt', 'carpet', 'dirt'];
export const MATRIX_DRS: DRLevel[] = ['low', 'high'];
export const PHYS_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110];

/** Re-draw each case's car from the drift setup (same per-case seeds as casesFor otherwise). */
function withSetup(cases: EvalCase[], dr: DRLevel, setup?: 'default' | 'drift'): EvalCase[] {
  if (setup !== 'drift') return cases;
  const base = driftSetup(defaultCarParams());
  return cases.map((c, i) => {
    const rng = makeRng(70_000 + i, `drift-setup/${c.track.id}`);
    return { ...c, car: dr === 'off' ? base : sampleCar(base, rng, dr, 1), noiseScale: noiseFor(rng, dr), latencyTicks: latencyFor(rng, dr) };
  });
}

export function baselineMatrix(opts: MatrixOpts = {}): MatrixCell[] {
  const stages = opts.stages ?? [1, 2, 3];
  const surfaces = opts.surfaces ?? MATRIX_SURFACES;
  const drs = opts.drs ?? MATRIX_DRS;
  const seeds = opts.seeds ?? PHYS_SEEDS.slice(0, 8);
  const copies = opts.copies ?? 2;
  const cells: MatrixCell[] = [];
  for (const stage of stages) {
    const base: TrackSpec[] = seeds.map((seed) => generateTrack({ seed, stage }));
    for (const surface of surfaces) {
      const tracks = base.map((t) => withSurface(t, surface));
      for (const dr of drs) {
        const cases = withSetup(casesFor(tracks, { dr, seed: 7, stage, copies }), dr, opts.setup);
        cells.push({ stage, surface, dr, run: evaluateBaseline(cases, opts.ctl) });
      }
    }
  }
  return cells;
}

/** Corners of the high-DR box (see sampleCarCorner) with the worst latency and sensor noise. */
export function cornerCases(tracks: TrackSpec[], stage: number, copies = 4, seed = 11): EvalCase[] {
  const out: EvalCase[] = [];
  tracks.forEach((track, ti) => {
    for (let c = 0; c < copies; c++) {
      const rng = makeRng(seed * 7919 + ti * 31 + c, `corner/${track.id}`);
      out.push({
        track,
        car: sampleCarCorner(defaultCarParams(), rng, 1),
        noiseScale: 2,
        latencyTicks: 2,
        stage,
      });
    }
  });
  return out;
}

export function formatMatrix(cells: MatrixCell[]): string[] {
  const lines = ['stage surface  DR    completion  lap  wall off  wrong stuck flip tmo'];
  for (const c of cells) {
    const r = c.run.reasons;
    lines.push(
      `${String(c.stage).padStart(5)} ${c.surface.padEnd(8)} ${c.dr.padEnd(5)} ${pct(c.run.rate).padStart(5)} (${c.run.completed}/${c.run.total})`.padEnd(44) +
        `${r.wall} ${r.offTrack} ${r.wrongWay} ${r.stuck} ${r.flip} ${r.timeout}`,
    );
  }
  return lines;
}

export interface CornerCell {
  stage: number;
  surface: Surface['name'];
  run: BaselineRun;
}

export function baselineCorners(opts: { ctl?: BaselineOpts; stages?: number[]; surfaces?: Surface['name'][]; seeds?: number[]; copies?: number } = {}): CornerCell[] {
  const out: CornerCell[] = [];
  for (const stage of opts.stages ?? [1, 2, 3]) {
    const base = (opts.seeds ?? PHYS_SEEDS.slice(0, 8)).map((seed) => generateTrack({ seed, stage }));
    for (const surface of opts.surfaces ?? MATRIX_SURFACES) {
      const tracks = base.map((t) => withSurface(t, surface));
      out.push({ stage, surface, run: evaluateBaseline(cornerCases(tracks, stage, opts.copies ?? 4), opts.ctl) });
    }
  }
  return out;
}

export const physics: Section = async ({ quick, log }) => {
  const seeds = PHYS_SEEDS.slice(0, quick ? 8 : 10);
  const cells = baselineMatrix({ seeds, copies: quick ? 1 : 2 });
  for (const l of formatMatrix(cells)) log(l);
  const rows: EvalRow[] = [];
  for (const stage of [...new Set(cells.map((c) => c.stage))]) {
    const mine = cells.filter((c) => c.stage === stage);
    const worst = mine.reduce((a, b) => (b.run.rate < a.run.rate ? b : a));
    const total = mine.reduce((a, c) => a + c.run.total, 0);
    const done = mine.reduce((a, c) => a + c.run.completed, 0);
    rows.push({
      milestone: 'M5',
      test: `Pure-pursuit baseline (upper bound), stage ${stage}, surface x DR`,
      target: '>= 70% in every cell',
      measured: `${pct(done / total)} overall, worst ${pct(worst.run.rate)} (${worst.surface}/${worst.dr}), ${total} laps`,
      pass: mine.every((c) => c.run.rate >= 0.7),
    });
  }
  const corners = baselineCorners({ ctl: { gripAware: true, recover: true }, seeds, copies: quick ? 2 : 4 });
  for (const stage of [...new Set(corners.map((c) => c.stage))]) {
    const mine = corners.filter((c) => c.stage === stage);
    const worst = mine.reduce((a, b) => (b.run.rate < a.run.rate ? b : a));
    const total = mine.reduce((a, c) => a + c.run.total, 0);
    const done = mine.reduce((a, c) => a + c.run.completed, 0);
    rows.push({
      milestone: 'M5',
      test: `Car model at the corners of the high-DR box (oracle controller), stage ${stage}`,
      target: '>= 70% on every surface',
      measured: `${pct(done / total)} overall, worst ${pct(worst.run.rate)} (${worst.surface}), ${total} laps`,
      pass: mine.every((c) => c.run.rate >= 0.7),
      note: stage === 1 ? 'corners = mass, CG, mu0, B/C/E, springs, motor, servo each at an end of the spec range, latency 2, noise x2' : undefined,
    });
  }
  return rows;
};
