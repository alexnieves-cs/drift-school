import type { Section } from './types';
import { sharedTrained } from './context';
import { casesFor, evaluateCases, pct } from './harness';
import { sCurveLoop } from './scurve';
import { closeLoop } from '../editor/editor';
import { smoothStroke, toFloat, trimClosingTail } from '../editor/stroke';
import { buildTrackFromPoints, validateTrack } from '../track/build';
import { generateTrack } from '../track/generate';
import { decodeTrackHash, encodeTrackHash } from '../editor/share';

const HALF_WIDTH = 1;
const MU = 1.05;
const HELD_OUT = [101, 102, 103, 104, 105, 106, 107, 108];

/**
 * M4, in process: the drawn S-curve goes through the editor's stroke pipeline, the trained learner is hot-swapped
 * onto it, then trained on for the spec's two minutes. The browser half (draw with the mouse, click Train,
 * measure end to end) is `npm run test:e2e`.
 */
export const hotswap: Section = async ({ quick, log }) => {
  const { trainer } = await sharedTrained();
  const act = (o: Float32Array, n: number) => trainer.policy.actMean(o, n);
  const rows = [] as Awaited<ReturnType<Section>>;

  // Same steps the editor takes on pointer-up.
  const raw = sCurveLoop(0.12).map(([x, z]) => ({ x, z }));
  const stroke = closeLoop(smoothStroke(trimClosingTail(raw), true, HALF_WIDTH), true);
  const points = toFloat(stroke);
  const check = validateTrack(points, HALF_WIDTH);
  rows.push({
    milestone: 'M4',
    test: 'Drawn S-curve is valid',
    target: 'valid',
    measured: check.ok ? 'valid' : `rejected: ${[...new Set(check.issues.map((i) => i.reason))].join(', ')}`,
    pass: check.ok,
  });
  if (!check.ok) return rows;
  const user = buildTrackFromPoints(points, { halfWidth: HALF_WIDTH, mu: MU, source: 'user' });

  const hash = encodeTrackHash(points, HALF_WIDTH, MU);
  const back = decodeTrackHash(hash);
  const roundTrip =
    back.points.length === points.length && back.points.every((v, i) => Math.abs(v - points[i]!) < 0.002);
  rows.push({ milestone: 'M4', test: 'URL hash round-trips the track', target: 'same points (1 mm)', measured: roundTrip ? `${hash.length} chars, same points` : 'points differ', pass: roundTrip });

  const heldOut = HELD_OUT.map((seed) => generateTrack({ seed, stage: 0 }));
  const procedural = (copies: number) => evaluateCases(act, casesFor(heldOut, { copies }));
  const procBefore = procedural(2);
  const zeroShot = evaluateCases(act, casesFor([user], { copies: 16 }));
  log(`S-curve zero-shot ${pct(zeroShot.rate)}, procedural before ${pct(procBefore.rate)}`);

  // The swap: every env on the new track, weights bit-identical.
  const hashBefore = trainer.weightHash();
  const t0 = performance.now();
  trainer.queueTrackPool([user, ...heldOut.slice(0, 4)], true);
  trainer.forceSwap();
  const swapMs = performance.now() - t0;
  const ids = trainer.env.trackIds();
  const onNew = ids.filter((id) => id === user.id).length;
  rows.push({
    milestone: 'M4',
    test: 'Hot-swap: all envs report the new track id',
    target: '<= 15 s',
    measured: `${onNew}/${ids.length} envs in ${swapMs.toFixed(0)} ms (in process; the browser run adds the wait for the running update)`,
    pass: onNew === ids.length && swapMs <= 15_000,
  });
  rows.push({
    milestone: 'M4',
    test: 'Weights unchanged at swap',
    target: 'hash equal',
    measured: trainer.weightHash() === hashBefore ? 'hash equal' : 'hash differs',
    pass: trainer.weightHash() === hashBefore,
  });

  // Fine-tune for two minutes of wall-clock training.
  const budget = quick ? 20 : 120;
  const t1 = performance.now();
  let updates = 0;
  while ((performance.now() - t1) / 1000 < budget) {
    await trainer.update();
    updates++;
  }
  const trained = (performance.now() - t1) / 1000;
  const after = evaluateCases(act, casesFor([user], { copies: 16 }));
  const procAfter = procedural(2);
  const retained = procBefore.rate > 0 ? procAfter.rate / procBefore.rate : 1;
  rows.push({
    milestone: 'M4',
    test: `Completion on the drawn track after ${budget === 120 ? '2 min' : `${budget} s (quick)`}`,
    target: '>= 80%',
    measured: `${pct(after.rate)} (${after.completed}/${after.total}); zero-shot was ${pct(zeroShot.rate)}; ${updates} updates in ${trained.toFixed(0)} s`,
    pass: after.rate >= 0.8,
  });
  rows.push({
    milestone: 'M4',
    test: 'Procedural completion retained (no forgetting)',
    target: '>= 90% of pre-swap',
    measured: `${pct(procAfter.rate)} vs ${pct(procBefore.rate)} before = ${pct(retained)} retained`,
    pass: retained >= 0.9,
  });
  return rows;
};
