import type { Section } from './types';
import { sharedTrained } from './context';
import { casesFor, evaluateCases, pct } from './harness';
import { generateTrack } from '../track/generate';

const HELD_OUT = [101, 102, 103, 104, 105, 106, 107, 108];

/** M2: stage-0 lap completion >= 80% within 3 minutes, headless, on seeds the trainer never saw. */
export const stage0: Section = async ({ log }) => {
  log('stage 0: training');
  const { trainer, seconds } = await sharedTrained();
  const act = (o: Float32Array, n: number) => trainer.policy.actMean(o, n);
  const tracks = HELD_OUT.map((seed) => generateTrack({ seed, stage: 0 }));
  const out = evaluateCases(act, casesFor(tracks, { copies: 2 }));
  return [
    {
      milestone: 'M2',
      test: 'Stage 0 lap completion, headless, held-out seeds',
      target: '>= 80% within 180 s',
      measured: `${pct(out.rate)} (${out.completed}/${out.total}) after ${seconds.toFixed(0)} s`,
      pass: out.rate >= 0.8 && seconds <= 180,
    },
  ];
};
