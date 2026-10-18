import type { Section } from './types';
import { sharedTrained } from './context';
import { casesFor, evaluateCases, pct, withSurface } from './harness';
import { generateTrack } from '../track/generate';

const SEEDS = [101, 102, 103, 104, 105, 106, 107, 108];
const SURFACE_NAMES = ['asphalt', 'carpet', 'dirt'] as const;

/** M5: held-out seeds x {asphalt, carpet, dirt} x DR {low, high}, one episode set per cell. */
export const matrix: Section = async ({ quick, log }) => {
  const { trainer } = await sharedTrained();
  const act = (o: Float32Array, n: number) => trainer.policy.actMean(o, n);
  const base = SEEDS.map((seed) => generateTrack({ seed, stage: 0 }));
  const copies = quick ? 1 : 3;
  let done = 0;
  let total = 0;
  const cells: string[] = [];
  for (const surface of SURFACE_NAMES) {
    for (const dr of ['low', 'high'] as const) {
      const out = evaluateCases(act, casesFor(base.map((t) => withSurface(t, surface)), { dr, copies, seed: 11 }));
      done += out.completed;
      total += out.total;
      cells.push(`${surface}/${dr} ${pct(out.rate)}`);
      log(`matrix ${surface}/${dr}: ${pct(out.rate)}`);
    }
  }
  const rate = total ? done / total : 0;
  return [
    {
      milestone: 'M5',
      test: 'Eval matrix completion (8 seeds x asphalt/carpet/dirt x DR low/high)',
      target: '>= 70%',
      measured: `${pct(rate)} (${done}/${total})`,
      pass: rate >= 0.7,
      note: cells.join(', '),
    },
  ];
};
