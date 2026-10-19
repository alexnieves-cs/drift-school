import type { Section } from './types';
import { sharedTrained } from './context';
import { casesFor, evaluateCases, pct } from './harness';
import { generateTrack } from '../track/generate';
import { STAGE0_HELD_OUT } from '../rl/seeds';
import { GaussianPolicy } from '../rl/model';

/**
 * M2: stage-0 lap completion >= 80% within 3 minutes of wall clock (training included), headless,
 * on track seeds the recipe never trained or validated on (src/rl/seeds.ts).
 * Stage 0 has no DR, so the 16 episodes are 16 distinct tracks.
 */
export function stage0Completion(act: (o: Float32Array, n: number) => Float32Array) {
  const tracks = STAGE0_HELD_OUT.map((seed) => generateTrack({ seed, stage: 0 }));
  return evaluateCases(act, casesFor(tracks));
}

export const stage0: Section = async ({ log, quick }) => {
  log('stage 0: training');
  const { trainer, seconds, report } = await sharedTrained(quick);
  const finalOut = stage0Completion((o, n) => trainer.policy.actMean(o, n));
  // The stage-0 curriculum phase is snapshotted when it ends, with the wall clock it took
  // (process-relative, TF.js start-up included). That snapshot is what the 3 minute bar is about;
  // the final policy (after stages 1-3 and PPO) must hold the 80% as well.
  const s0 = report.stage0!;
  const snap = new GaussianPolicy(3e-4, 1);
  snap.setWeights(s0.weights);
  const snapOut = stage0Completion((o, n) => snap.actMean(o, n));
  snap.dispose();
  const ppo = report.ppo;
  return [
    {
      milestone: 'M2',
      test: 'Stage 0 lap completion, headless, held-out seeds',
      target: '>= 80% within 180 s',
      measured: `${pct(snapOut.rate)} (${snapOut.completed}/${snapOut.total}) after ${s0.seconds.toFixed(0)} s; final policy ${pct(finalOut.rate)} (${finalOut.completed}/${finalOut.total}) after ${seconds.toFixed(0)} s total`,
      pass: snapOut.rate >= 0.8 && s0.seconds <= 180 && finalOut.rate >= 0.8 && seconds <= 180,
      note:
        `First figure: the stage-0 phase snapshot (${s0.seconds.toFixed(0)} s wall clock from start). Final policy: whole recipe ` +
        `(imitation of pure pursuit through stages 0-3, then PPO), ${seconds.toFixed(0)} s. Pass needs both >= 80% and both <= 180 s. ` +
        `Held-out seeds ${STAGE0_HELD_OUT[0]}-${STAGE0_HELD_OUT[STAGE0_HELD_OUT.length - 1]}, no DR (${STAGE0_HELD_OUT.length} distinct tracks). ` +
        `The driving skill comes from DAgger imitation of the pure-pursuit baseline; PPO ${ppo ? `ran ${ppo.proposed} gated updates (${ppo.accepted} kept)` : 'did not run (quick mode)'}.`,
    },
  ];
};
