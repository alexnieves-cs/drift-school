/**
 * Control experiment: PPO with no imitation at all, on the stage-0 scenario pool, for a fixed CPU budget.
 * Reports held-out stage-0 completion over time so the imitation/PPO split is measured, not asserted.
 *
 *   npx vite-node scripts/rl-ppo-only.ts -- [--cpu=120] [--lr=3e-4] [--n=16] [--h=200]
 */
import { Trainer } from '../src/rl/trainer';
import { ScenarioSet } from '../src/rl/scenarios';
import { DEFAULT_TRAIN_SEEDS } from '../src/rl/seeds';
import { stage0Completion } from '../src/eval/stage0';
import { pct } from '../src/eval/harness';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;
const budget = Number(arg('cpu', '120'));
const lr = Number(arg('lr', '3e-4'));
const n = Number(arg('n', '16'));
const h = Number(arg('h', '200'));
const cpu = () => {
  const u = process.cpuUsage();
  return (u.user + u.system) / 1e6;
};

const trainer = new Trainer({ n, horizon: h, epochs: 2, minibatch: 256, seed: 1, lr });
const set = new ScenarioSet(DEFAULT_TRAIN_SEEDS);
await trainer.init(set.tracks(0));
trainer.setScenarios(set);
trainer.curriculum.stage = 0;
const t0 = cpu();
let updates = 0;
let nextReport = 30;
while (cpu() - t0 < budget) {
  await trainer.update();
  updates++;
  if (cpu() - t0 >= nextReport) {
    const out = stage0Completion((o, k) => trainer.policy.actMean(o, k));
    console.log(`cpu ${(cpu() - t0).toFixed(0)} s, ${updates} updates, ${updates * n * h} env steps: held-out stage 0 ${pct(out.rate)} (${out.completed}/${out.total})`);
    nextReport += 30;
  }
}
const out = stage0Completion((o, k) => trainer.policy.actMean(o, k));
console.log(`final after ${updates} updates: held-out stage 0 ${pct(out.rate)} (${out.completed}/${out.total}); best lap seen ${trainer.bestLap}`);
