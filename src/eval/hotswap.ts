import type { Section, EvalRow } from './types';
import { sharedTrained } from './context';
import { casesFor, evaluateCases, pct, type ActFn } from './harness';
import { generateTrack } from '../track/generate';
import { buildTrackFromPoints, validateTrack } from '../track/build';
import { Trainer } from '../rl/trainer';
import { FineTune, type FineTuneMode, type FineTuneReport } from '../rl/finetune';
import { processStroke, sCurveStroke, triangleStroke } from '../editor/pipeline';
import { decodeTrackHash, encodeTrackHash, quantizePoints } from '../editor/share';
import { toFloat } from '../editor/stroke';
import type { TrackSpec } from '../shared/types';

/** Procedural seeds the base recipe never trained on (the replay pool is whatever the trainer holds). */
const HELD_OUT_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116];
const HALF_WIDTH = 1;
const MU = 1.05;

export interface Snapshot {
  /** User track, deterministic, DR off. */
  user: number;
  /** User track, DR low, 24 draws. Secondary: the fine-tune did not train with DR. */
  userDr: number;
  /** Second drawn shape (rounded triangle) the fine-tune never saw. */
  heldDrawn: number;
  /** Held-out procedural seeds, 2 draws each. */
  heldProc: number;
  /** The procedural replay pool the fine-tune did train on. */
  poolProc: number;
}

export function snapshot(policyAct: ActFn, user: TrackSpec, drawn: TrackSpec, held: TrackSpec[], pool: TrackSpec[]): Snapshot {
  return {
    user: evaluateCases(policyAct, casesFor([user], { copies: 1 })).rate,
    userDr: evaluateCases(policyAct, casesFor([user], { copies: 24, dr: 'low', seed: 11 })).rate,
    heldDrawn: evaluateCases(policyAct, casesFor([drawn], { copies: 1 })).rate,
    heldProc: evaluateCases(policyAct, casesFor(held, { copies: 2 })).rate,
    poolProc: evaluateCases(policyAct, casesFor(pool, { copies: 2 })).rate,
  };
}

/** Same net, same envs, same shared recipe weights, fresh optimizer: the shared trainer stays untouched. */
export async function cloneTrainer(src: Trainer, pool: TrackSpec[]): Promise<Trainer> {
  const t = new Trainer({ n: src.env.n, horizon: src.horizon, epochs: src.epochs, minibatch: src.minibatch });
  await t.init(pool);
  t.policy.setWeights(src.policy.getWeights());
  t.curriculum.stage = src.curriculum.stage;
  t.gen = src.gen;
  return t;
}

export interface RunResult {
  report: FineTuneReport;
  after: Snapshot;
  trainSeconds: number;
  /** Training seconds at the first probe with user completion >= 80%, or null. */
  secondsTo80: number | null;
  probes: string;
  /** User-track and held-out procedural completion at each probe (every 15 training seconds). */
  userProbes: number[];
  heldProbes: number[];
}

/**
 * One swap + fine-tune from `base` weights. `windowS` counts training time only: the wall
 * clock spent inside tune-phase steps, not the probes between them.
 */
export async function runFineTune(
  base: Trainer,
  pool: TrackSpec[],
  user: TrackSpec,
  drawn: TrackSpec,
  held: TrackSpec[],
  mode: FineTuneMode,
  windowS: number,
  log: (s: string) => void,
): Promise<RunResult> {
  const t = await cloneTrainer(base, pool);
  const ft = new FineTune(t, { user, procedural: pool, mode });
  ft.start();
  const act: ActFn = (o, n) => t.policy.actMean(o, n);
  let secondsTo80: number | null = null;
  const probes: string[] = [];
  const userProbes: number[] = [];
  const heldProbes: number[] = [];
  let nextProbe = 15;
  let guard = 0;
  while (ft.report().tuneMs / 1000 < windowS || ft.phase !== 'tune') {
    await ft.step();
    if (ft.phase === 'tune' && ft.report().tuneMs / 1000 >= nextProbe) {
      const s = ft.report().tuneMs / 1000;
      const u = evaluateCases(act, casesFor([user], { copies: 1 })).rate;
      const h = evaluateCases(act, casesFor(held, { copies: 1 })).rate;
      probes.push(`${s.toFixed(0)}s:${pct(u)}/${pct(h)}`);
      userProbes.push(u);
      heldProbes.push(h);
      if (secondsTo80 === null && u >= 0.8) secondsTo80 = s;
      nextProbe += 15;
    }
    if (++guard > 200_000) throw new Error('fine-tune loop did not finish');
  }
  const report = ft.report();
  ft.stop();
  log(`${mode}: ${report.steps} steps, ${report.ppoUpdates} PPO updates, ${report.imitationBatches} DAgger batches`);
  return {
    report,
    after: snapshot(act, user, drawn, held, pool),
    trainSeconds: report.tuneMs / 1000,
    secondsTo80,
    probes: probes.join(' '),
    userProbes,
    heldProbes,
  };
}

const lowest = (xs: number[]) => (xs.length ? Math.min(...xs) : NaN);
const at = (r: RunResult) => (r.secondsTo80 === null ? 'never' : `${r.secondsTo80.toFixed(0)} s`);

/**
 * Plain statement of what imitation and PPO each contribute, from three runs that start at the
 * same weights and get the same training budget: combined (what ships), DAgger only, PPO only.
 */
function attribution(zeroShot: number, combined: RunResult, imitation: RunResult, ppo: RunResult, windowS: number): string {
  const line = (name: string, r: RunResult) =>
    `${name}: user ${pct(r.after.user)} at the end, >= 80% ${at(r)}, lowest user probe ${pct(lowest(r.userProbes))}, ` +
    `held-out procedural ${pct(r.after.heldProc)} at the end and ${pct(lowest(r.heldProbes))} at worst`;
  const parts = [
    `Attribution, same starting weights, ${windowS} s of training each (zero-shot ${pct(zeroShot)}):`,
    `${line('imitation only', imitation)};`,
    `${line('PPO only', ppo)};`,
    `${line('combined (shipped)', combined)}.`,
  ];
  if (zeroShot >= 0.8) return `${parts.join(' ')} The base already drove this track, so there is no improvement to attribute.`;
  const iOk = imitation.after.user >= 0.8;
  const pOk = ppo.after.user >= 0.8;
  let verdict: string;
  if (iOk && !pOk) verdict = 'Imitation supplies the improvement; PPO alone did not get there in the budget.';
  else if (!iOk && pOk) verdict = 'PPO alone supplies the improvement; imitation alone did not get there in the budget.';
  else if (iOk && pOk) {
    const iFirst = (imitation.secondsTo80 ?? Infinity) <= (ppo.secondsTo80 ?? Infinity);
    verdict =
      `Either part alone ends above 80%. ${iFirst ? 'Imitation' : 'PPO'} gets there first (${at(iFirst ? imitation : ppo)} vs ${at(iFirst ? ppo : imitation)}), ` +
      `and the combined run reaches it at ${at(combined)}, so the early improvement is ${iFirst ? "imitation's" : "PPO's"}; ` +
      `${lowest(ppo.userProbes) < 0.8 || lowest(ppo.heldProbes) < 0.8 ? 'PPO alone dips below 80% at some probe (unstable)' : 'PPO alone held up at every probe'}.`;
  } else verdict = 'Neither part alone reached 80% in the budget.';
  return `${parts.join(' ')} ${verdict}`;
}

/**
 * M4: draw an S-curve (mouse stroke through the editor pipeline), hot-swap it into training,
 * fine-tune for 2 minutes, then check the swap latency, that the weights did not change at the
 * swap, completion on the user's track, no forgetting on procedural tracks, and the URL hash.
 */
export const hotswap: Section = async ({ quick, log }) => {
  const rows: EvalRow[] = [];
  const row = (r: Omit<EvalRow, 'milestone'>) => rows.push({ milestone: 'M4', ...r });

  // 1. The drawing, through the real editor pipeline.
  const stroke = processStroke(sCurveStroke(), HALF_WIDTH, MU);
  if (!stroke.track || !stroke.validation.ok || !stroke.closed) {
    row({
      test: 'S-curve stroke becomes a valid track',
      target: 'closed and valid',
      measured: `invalid: ${stroke.validation.issues.map((i) => i.reason).join(', ') || 'no track'}`,
      pass: false,
    });
    return rows;
  }
  const flat = quantizePoints(toFloat(stroke.points));
  const user = buildTrackFromPoints(flat, { halfWidth: HALF_WIDTH, mu: MU, source: 'user' });
  const t0 = performance.now();
  validateTrack(flat, HALF_WIDTH);
  buildTrackFromPoints(flat, { halfWidth: HALF_WIDTH, mu: MU, source: 'user' });
  const editMs = performance.now() - t0;
  row({
    test: 'S-curve stroke becomes a valid track',
    target: 'closed, valid',
    measured: `valid, ${user.length.toFixed(1)} m, ${stroke.points.length} control points, validate+spline ${editMs.toFixed(1)} ms`,
    pass: true,
  });
  const drawn = processStroke(triangleStroke(), HALF_WIDTH, MU).track;
  if (!drawn) throw new Error('held-out drawn track did not build');

  // 2. Base policy and the pre-swap numbers.
  log('hotswap: training base policy');
  const shared = await sharedTrained();
  const pool = shared.trainer.mix.tracks.slice();
  const poolIds = new Set(pool.map((p) => p.id));
  const held = HELD_OUT_SEEDS.map((seed) => generateTrack({ seed, stage: 0 })).filter((tr) => !poolIds.has(tr.id));
  let base = shared.trainer;
  let before = snapshot((o, n) => base.policy.actMean(o, n), user, drawn, held, pool);
  log(`hotswap: pre-swap user ${pct(before.user)} (DR low ${pct(before.userDr)}), held-out procedural ${pct(before.heldProc)}, pool ${pct(before.poolProc)}`);
  let baseNote = '';
  if (before.heldProc < 0.8) {
    // "Procedural completion stays >= 90% of its pre-swap value" says nothing when the pre-swap
    // value is already low, so say so and measure it on a base that actually drives the tracks:
    // the same recipe without its single PPO update (that update is what makes the shared base flaky).
    const weak = before;
    base = new Trainer({ n: shared.trainer.env.n, horizon: shared.trainer.horizon, epochs: shared.trainer.epochs, minibatch: shared.trainer.minibatch });
    await base.init(pool);
    await base.imitate(4);
    before = snapshot((o, n) => base.policy.actMean(o, n), user, drawn, held, pool);
    baseNote =
      ` The shared recipe's policy drove only ${pct(weak.heldProc)} of held-out and ${pct(weak.poolProc)} of pool procedural cases pre-swap (user track ${pct(weak.user)}), ` +
      `so the retention check ran on an imitation-only base instead (pre-swap ${pct(before.heldProc)} held-out, ${pct(before.poolProc)} pool, user ${pct(before.user)}).`;
    log(`hotswap: shared base weak (${pct(weak.heldProc)} held-out); using imitation-only base: held-out ${pct(before.heldProc)}`);
  }

  // 3. Swap + fine-tune.
  const windowS = quick ? 20 : 120;
  log(`hotswap: swap and fine-tune for ${windowS} s of training`);
  const main = await runFineTune(base, pool, user, drawn, held, 'mixed', windowS, log);
  const r = main.report;
  const quickNote = quick ? ' (quick: shortened window)' : '';

  row({
    test: 'Hot-swap: all envs report the new track id',
    target: '<= 15 s',
    measured: r.latencyMs === null ? 'never' : `${(r.latencyMs / 1000).toFixed(1)} s, ${r.envsOnUserAtSwap}/${r.envTotal} envs on ${user.id}`,
    pass: r.latencyMs !== null && r.latencyMs <= 15_000 && r.envsOnUserAtSwap === r.envTotal,
    note: `Each env finishes its episode, then restarts on the user track; any env still mid-episode at 8 s is retargeted. Wall clock on a shared machine. Zero-shot on the new track, sampled policy, no update: ${
      r.zeroShot ? `${pct(r.zeroShot.completion)} of ${r.zeroShot.episodes} episodes` : 'n/a'
    }; deterministic pre-swap ${pct(before.user)}.`,
  });
  row({
    test: 'Weights unchanged at the swap (hash equal)',
    target: 'hash equal',
    measured: `${r.weightHashStart} -> ${r.weightHashAtSwap} (${r.weightsUnchanged ? 'equal' : 'DIFFERENT'})`,
    pass: r.weightsUnchanged === true,
  });
  const dagShare = r.imitationBatches;
  row({
    test: `Completion on the user track after ${quick ? `${windowS} s (quick)` : '2 min'} fine-tune`,
    target: '>= 80%',
    measured: `${pct(main.after.user)} (was ${pct(before.user)}); DR low ${pct(main.after.userDr)} (was ${pct(before.userDr)}); held-out drawn track ${pct(main.after.heldDrawn)} (was ${pct(before.heldDrawn)})`,
    pass: quick ? null : main.after.user >= 0.8,
    note:
      `${quick ? 'quick mode: ' : ''}${main.trainSeconds.toFixed(0)} s of training${quickNote}, ${r.ppoUpdates} PPO updates + ${dagShare} DAgger batches ` +
      `(3 per PPO update), probes (user/held-out procedural) ${main.probes || 'none'}. Adaptation is imitation of the pure-pursuit baseline (DAgger on the user track and the replay pool) interleaved with PPO; ` +
      `see the ablation line for what each contributes.`,
  });
  const ratio = (post: number, pre: number) => (pre > 0 ? post / pre : post >= 0 ? 1 : 0);
  const ratioHeld = ratio(main.after.heldProc, before.heldProc);
  const ratioPool = ratio(main.after.poolProc, before.poolProc);
  row({
    test: `Procedural completion retained, held-out seeds${quick ? ' (quick)' : ''}`,
    target: '>= 90% of pre-swap',
    measured: `${pct(main.after.heldProc)} vs ${pct(before.heldProc)} pre-swap (${pct(ratioHeld)} retained, ${held.length} seeds x 2)`,
    pass: quick ? null : ratioHeld >= 0.9,
    note: `Seeds the base recipe and the fine-tune never trained on, same cases before and after, deterministic policy.${baseNote}`,
  });
  row({
    test: `Procedural completion retained, replay-pool seeds${quick ? ' (quick)' : ''}`,
    target: '>= 90% of pre-swap',
    measured: `${pct(main.after.poolProc)} vs ${pct(before.poolProc)} pre-swap (${pct(ratioPool)} retained, ${pool.length} seeds x 2)`,
    pass: quick ? null : ratioPool >= 0.9,
    note: 'These seeds are in the 30% replay share, so they were trained on during the fine-tune: reported next to the held-out row, not instead of it.',
  });

  // 4. Ablation: what PPO alone and imitation alone do from the same weights, same budget.
  if (!quick) {
    const abl: Partial<Record<FineTuneMode, RunResult>> = {};
    for (const mode of ['ppo', 'imitation'] as const) {
      log(`hotswap: ablation ${mode}`);
      abl[mode] = await runFineTune(base, pool, user, drawn, held, mode, windowS, log);
    }
    const userRow = rows.find((x) => x.test.startsWith('Completion on the user track'));
    if (userRow) userRow.note += ` ${attribution(before.user, main, abl.imitation!, abl.ppo!, windowS)}`;
  }

  // 5. URL hash round trip.
  const hash = encodeTrackHash(flat, HALF_WIDTH, MU);
  const back = decodeTrackHash(hash);
  let worst = 0;
  for (let i = 0; i < flat.length; i++) worst = Math.max(worst, Math.abs(back.points[i]! - flat[i]!));
  const rebuilt = buildTrackFromPoints(back.points, { halfWidth: back.halfWidth, mu: back.mu, source: 'user' });
  row({
    test: 'URL hash round-trips the track',
    target: 'control points within 1 mm, same track id',
    measured: `${flat.length / 2} points, worst ${(worst * 1000).toFixed(2)} mm, hash ${hash.length} chars, id ${rebuilt.id === user.id ? 'same' : `${user.id} -> ${rebuilt.id}`}`,
    pass: worst <= 0.001 + 1e-6 && back.points.length === flat.length && rebuilt.id === user.id,
  });
  return rows;
};
