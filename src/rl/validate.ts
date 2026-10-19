import { casesFor, evaluateCases, withSurface, type ActFn, type EvalCase } from '../eval/harness';
import { generateTrack } from '../track/generate';
import type { DRLevel, Surface } from '../sim/dr';

export interface ValSuite {
  cases: EvalCase[];
  /** Index into the `stages` list the suite was built from. */
  stageOf: number[];
  stages: number[];
}

export interface ValResult {
  rate: number;
  completed: number;
  total: number;
  /** Mean lap time (s) of completed cases; 0 when none completed. Tie-break when rates are equal. */
  meanLap: number;
  /** Completion per stage, in the order of `suite.stages`. */
  perStage: number[];
}

const DRS: DRLevel[] = ['off', 'low', 'high'];
const SURFACES: (Surface['name'] | 'native')[] = ['native', 'asphalt', 'carpet', 'dirt'];

/**
 * Fixed mixed suite: for each stage the cases cycle through DR level and surface, so one number
 * tracks "does the policy still work everywhere the recipe trains it". Stage 0 stays DR-off on its
 * native surface (SPEC stage 0).
 */
export function buildSuite(seeds: number[], stages: number[]): ValSuite {
  const cases: EvalCase[] = [];
  const stageOf: number[] = [];
  stages.forEach((stage, si) => {
    seeds.forEach((seed, k) => {
      let track = generateTrack({ seed, stage });
      const dr: DRLevel = stage === 0 ? 'off' : DRS[(k + si) % 3]!;
      const surf = stage === 0 ? 'native' : SURFACES[(k + si) % SURFACES.length]!;
      if (surf !== 'native') track = withSurface(track, surf);
      cases.push(...casesFor([track], { dr, seed: seed + stage * 17, stage }));
      stageOf.push(si);
    });
  });
  return { cases, stageOf, stages };
}

export function validate(act: ActFn, suite: ValSuite): ValResult {
  const done = new Array<number>(suite.stages.length).fill(0);
  const total = new Array<number>(suite.stages.length).fill(0);
  let completed = 0;
  let lapSum = 0;
  let lapN = 0;
  for (let s = 0; s < suite.stages.length; s++) {
    const cs = suite.cases.filter((_, i) => suite.stageOf[i] === s);
    if (!cs.length) continue;
    const out = evaluateCases(act, cs, cs.length);
    done[s] = out.completed;
    total[s] = out.total;
    completed += out.completed;
    for (const lt of out.lapTimes) {
      lapSum += lt;
      lapN++;
    }
  }
  const n = suite.cases.length;
  return {
    rate: n ? completed / n : 0,
    completed,
    total: n,
    meanLap: lapN ? lapSum / lapN : 0,
    perStage: done.map((d, i) => (total[i] ? d / total[i]! : 0)),
  };
}
