import type { Section } from './types';
import { sharedTrained } from './context';
import { casesFor, evaluateCases, pct } from './harness';
import { generateTrack } from '../track/generate';
import { STAGE0_HELD_OUT } from '../rl/seeds';

/**
 * M2: stage-0 lap completion >= 80% within 3 minutes of wall clock (training included), headless,
 * on track seeds the recipe never trained or validated on (src/rl/seeds.ts).
 * Stage 0 has no DR, so the 16 episodes are 16 distinct tracks.
 */
export const stage0: Section = async ({ log, quick }) => {
  log('stage 0: training');
  const { trainer, seconds, report } = await sharedTrained(quick);
  const act = (o: Float32Array, n: number) => trainer.policy.actMean(o, n);
  const tracks = STAGE0_HELD_OUT.map((seed) => generateTrack({ seed, stage: 0 }));
  const out = evaluateCases(act, casesFor(tracks));
  const s0 = report.stage0;
  const ppo = report.ppo;
  return [
    {
      milestone: 'M2',
      test: 'Stage 0 lap completion, headless, held-out seeds',
      target: '>= 80% within 180 s',
      measured: `${pct(out.rate)} (${out.completed}/${out.total}) after ${seconds.toFixed(0)} s`,
      pass: out.rate >= 0.8 && seconds <= 180,
      note:
        `${seconds.toFixed(0)} s is the whole recipe (imitation of pure pursuit stages 0-3, then PPO); ` +
        `the stage-0 imitation phase alone ended at ${s0 ? s0.seconds.toFixed(0) : '?'} s. ` +
        `Skill comes from imitating the baseline (DAgger); PPO ran ${ppo ? `${ppo.proposed} gated updates, ${ppo.accepted} accepted` : 'not at all (quick mode)'}.`,
    },
  ];
};
