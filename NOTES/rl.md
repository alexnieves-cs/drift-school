# RL stream notes

## What changed
- `src/rl/recipe.ts` `trainRecipe()` is the one recipe (eval and browser worker). Fully seeded (own RNG for init, action noise, shuffles, scenarios; no TF.js RNG, no wall-clock branching).
  1. DAgger imitation of the pure-pursuit baseline, stage by stage 0..3 (`scenarios.ts`: stage track pools, DR off/low/high per stage, uniform asphalt/carpet/dirt surfaces mixed in, 20% replay of earlier stages). Gradient steps run on a typed-array backprop (`fastfit.ts`, checked against TF autodiff in `tests/rl/fastfit.test.ts`), about 3x faster than TF.js CPU.
  2. Short PPO fine-tune: critic-only warm-up, then PPO at lr 3e-5, clip 0.1, grad-norm clip 0.5, fixed small exploration std, validation-gated on held-out seeds (an update group that is not better is reverted).
- `Trainer`: scenario mode (`setScenarios`, curriculum stage now selects the pool), seeded, value bootstrap at rollout end, shuffled minibatches, critic-only mode. TrackMix/hot-swap path unchanged (a swapped-in pool takes over from scenarios).
- PPO bug fixed: `ppoLoss` re-derived the pre-tanh sample as atanh(action) which is wrong for saturated actions; log-ratio KL was about 2 before any update, so every PPO update was clipped to nothing. Now `act()` returns `pre` and PPO scores the sample it drew (KL about 0).
- Seeds (`seeds.ts`): training 1..24 by default (alt sets 1000k+1..24); stage-0 held-out 101-116; matrix 201-212; validation 301-306. `trainRecipe` throws on overlap.
- `src/eval/matrix.ts` (pre-registered: stage 2 tracks, 12 seeds, 2 DR draws per cell = 144 episodes, stages 1 and 3 informational) and `src/eval/stage0.ts` (stage-0 phase snapshot + final policy).
- Browser: `learner.worker.ts` calls `trainRecipe` with `BROWSER_TUNE` (stage 0 only, no PPO phase, no validation), then `configureFinetune`, horizon 32.

## How much is imitation, how much is PPO
Essentially all of the lap-completing skill is imitation of the pure-pursuit baseline (which itself finishes about 100% of stage 0-3 tracks, all surfaces and DR levels, on seeds 101-116 x 2 copies). The policy is bounded by it. The runs below were measured before the PPO log-ratio fix: PPO then changed nothing (0/4 updates kept). After the fix PPO moves weights with sane KL, but its effect on completion or lap time is NOT measured yet (`scripts/rl-runs.ts` and `scripts/rl-ppo-only.ts` exist for that). From-scratch PPO (no imitation) is not claimed to work.

## Measured (policy seed k+1, training seeds as shown; imitation-only effective)
| run | train seeds | stage0 snapshot (16 held-out) | stage0 final | train wall/cpu | matrix s1 | s2 (pass set) | s3 |
|---|---|---|---|---|---|---|---|
| 0 | 1..24 | 100% at 35 s | 100% | 123 s / 123 s | 99% | 99% (142/144) | 99% |
| 1 | 1001..1024 | 100% at 34 s | 100% | 126 s / 121 s | 99% | not finished | not finished |
Run 0 stage-2 cells: asphalt low 100 / high 96, carpet low 100 / high 100, dirt low 100 / high 96.
Earlier single runs with the same recipe family, 97-100% on stage 2. Runs 2 and 3 (seeds 2001.., 3001..) were still running when this was written.

## Still weak / unmeasured
- Matrix numbers above are for the pre-fix PPO (no-op). Re-run `npx vite-node scripts/rl-runs.ts -- --runs=4 --stages=1,2,3` after merge for final numbers.
- PPO's real contribution is unmeasured; expect small at this budget (TF.js CPU, about 5 s per update).
- Total recipe is about 125 s CPU; the stage-0 phase alone about 35 s.
- Browser path trains stage 0 only by imitation; later stages rely on PPO there.
