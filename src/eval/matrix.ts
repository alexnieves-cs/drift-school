import type { Section } from './types';
import { sharedTrained } from './context';
import { casesFor, evaluateCases, pct, withSurface } from './harness';
import { generateTrack } from '../track/generate';
import { MATRIX_SEEDS } from '../rl/seeds';
import type { DRLevel, Surface } from '../sim/dr';

/**
 * M5 eval matrix: seeds x {asphalt, carpet, dirt} x DR {low, high}.
 *
 * The design is fixed here, before any result was looked at:
 *  - PASS track set: SPEC stage 2 tracks (width 1.0 m, tighter corners), the 12 seeds in MATRIX_SEEDS
 *    (201-212), none of them used for training or validation. Each surface forces a uniform mu on
 *    the whole track (native mixed-mu zones are replaced).
 *  - 2 episodes per (seed, cell) with different DR draws -> 24 episodes per cell, 144 in total.
 *  - Pass bar: overall completion >= 70% on the stage-2 matrix. The minimum cell is reported too.
 *  - Stages 1 and 3 are run with the identical protocol for information only. They do not affect
 *    the pass flag, and nothing about this choice depends on how they come out.
 */
const SURFACES: Surface['name'][] = ['asphalt', 'carpet', 'dirt'];
const DRS: DRLevel[] = ['low', 'high'];
const PASS_STAGE = 2;
const INFO_STAGES = [1, 3];
const COPIES = 2;

interface Grid {
  completed: number;
  total: number;
  cells: { label: string; completed: number; total: number }[];
}

function runStage(act: (o: Float32Array, n: number) => Float32Array, stage: number, copies: number): Grid {
  const tracks = MATRIX_SEEDS.map((seed) => generateTrack({ seed, stage }));
  const grid: Grid = { completed: 0, total: 0, cells: [] };
  for (const surface of SURFACES) {
    for (const dr of DRS) {
      const cases = casesFor(
        tracks.map((t) => withSurface(t, surface)),
        { dr, stage, copies, seed: 11 },
      );
      const out = evaluateCases(act, cases);
      grid.completed += out.completed;
      grid.total += out.total;
      grid.cells.push({ label: `${surface}/${dr}`, completed: out.completed, total: out.total });
    }
  }
  return grid;
}

const describe = (g: Grid) =>
  `${pct(g.completed / g.total)} (${g.completed}/${g.total}); ` +
  g.cells.map((c) => `${c.label} ${pct(c.completed / c.total)}`).join(', ');

export const matrix: Section = async ({ log, quick }) => {
  log('matrix: training (shared)');
  const { trainer, report } = await sharedTrained(quick);
  const act = (o: Float32Array, n: number) => trainer.policy.actMean(o, n);
  const copies = quick ? 1 : COPIES;
  log(`matrix: stage ${PASS_STAGE}, ${MATRIX_SEEDS.length} seeds x 3 surfaces x 2 DR x ${copies}`);
  const main = runStage(act, PASS_STAGE, copies);
  const info = quick
    ? []
    : INFO_STAGES.map((stage) => {
        log(`matrix: stage ${stage} (information only)`);
        return { stage, grid: runStage(act, stage, copies) };
      });
  const rate = main.completed / main.total;
  const minCell = Math.min(...main.cells.map((c) => c.completed / c.total));
  const ppo = report.ppo;
  const note =
    `Pre-registered: stage ${PASS_STAGE} tracks, seeds ${MATRIX_SEEDS[0]}-${MATRIX_SEEDS[MATRIX_SEEDS.length - 1]} ` +
    `(${MATRIX_SEEDS.length}, never trained on), ${copies} DR draw(s) per seed and cell, uniform surface mu. ` +
    `Stage ${PASS_STAGE} per cell: ${describe(main)}; worst cell ${pct(minCell)}. ` +
    info.map((i) => `Stage ${i.stage} (info only): ${describe(i.grid)}. `).join('') +
    `Policy = DAgger imitation of the pure-pursuit baseline, then ${ppo ? `${ppo.accepted}/${ppo.proposed} validation-gated PPO updates kept` : 'no PPO (quick mode)'}.`;
  return [
    {
      milestone: 'M5',
      test: 'Eval matrix completion (seeds x surfaces x DR)',
      target: '>= 70%',
      measured: `${pct(rate)} (${main.completed}/${main.total}), worst cell ${pct(minCell)}`,
      pass: rate >= 0.7,
      note,
    },
  ];
};
