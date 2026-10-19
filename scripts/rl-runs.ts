/**
 * Repeat the whole recipe over different policy seeds and training-seed sets and print stage-0 and
 * matrix completion for each run. This is how robustness is shown (one lucky run proves nothing).
 *
 *   npx vite-node scripts/rl-runs.ts -- --runs=4 [--first=0] [--stages=1,2,3] [--ppo=N] [--copies=2]
 *
 * Run k uses policy seed k+1 and training track seeds 1000*k+1 .. 1000*k+24 (k = 0 uses 1..24),
 * all disjoint from the evaluation (101-116, 201-212) and validation (301-306) seeds.
 */
import { trainRecipe } from '../src/rl/recipe';
import { GaussianPolicy } from '../src/rl/model';
import { stage0Completion } from '../src/eval/stage0';
import { describeGrid, runMatrixStage } from '../src/eval/matrix';
import { pct } from '../src/eval/harness';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;
const runs = Number(arg('runs', '4'));
const first = Number(arg('first', '0'));
const stages = arg('stages', '2').split(',').filter(Boolean).map(Number);
const copies = Number(arg('copies', '2'));
const ppo = arg('ppo', '');
const cpu = () => {
  const u = process.cpuUsage();
  return (u.user + u.system) / 1e6;
};

for (let k = first; k < first + runs; k++) {
  const seeds = Array.from({ length: 24 }, (_, i) => (k === 0 ? 1 : 1000 * k) + i + (k === 0 ? 0 : 1));
  const cpu0 = cpu();
  const { trainer, seconds, report } = await trainRecipe({ seed: k + 1, seeds, ...(ppo !== '' ? { ppoUpdates: Number(ppo) } : {}) });
  const cpuTrain = cpu() - cpu0;
  const act = (o: Float32Array, n: number) => trainer.policy.actMean(o, n);
  const fin = stage0Completion(act);
  const snap = new GaussianPolicy(3e-4, 1);
  snap.setWeights(report.stage0!.weights);
  const s0 = stage0Completion((o, n) => snap.actMean(o, n));
  snap.dispose();
  console.log(
    `run ${k} policySeed ${k + 1} trainSeeds ${seeds[0]}..${seeds[seeds.length - 1]}: train ${seconds.toFixed(0)} s wall / ${cpuTrain.toFixed(0)} s cpu; ` +
      `stage0 snapshot ${pct(s0.rate)} (${s0.completed}/${s0.total}) at ${report.stage0!.seconds.toFixed(0)} s; stage0 final ${pct(fin.rate)} (${fin.completed}/${fin.total}); ` +
      `ppo ${report.ppo ? `${report.ppo.accepted}/${report.ppo.proposed} kept, val lap ${report.ppo.startVal?.meanLap.toFixed(2)} -> ${report.ppo.endVal?.meanLap.toFixed(2)}` : 'off'}; hash ${trainer.weightHash()}`,
  );
  for (const stage of stages) {
    const g = runMatrixStage(act, stage, copies);
    console.log(`  matrix stage ${stage}: ${describeGrid(g)}`);
  }
}
