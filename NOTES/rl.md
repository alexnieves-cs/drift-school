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
Essentially all of the lap-completing skill is imitation of the pure-pursuit baseline (which itself finishes about 100% of stage 0-3 tracks, all surfaces and DR levels, on seeds 101-116 x 2 copies). The policy is bounded by it. From-scratch PPO (no imitation) is not claimed to work; `scripts/rl-ppo-only.ts` is the control and has not been run.

PPO after the log-ratio fix, imitation policy through stage 3, 2 critic + 8 PPO updates, 24-episode validation suite (completion stayed 100% at every lr):

| PPO lr | mean lap after 8 updates | vs imitation (11.23 s) |
|---|---|---|
| 3e-5 (recipe default) | 11.31 s | about 0.7% slower |
| 1e-4 | 11.26 s | about 0.3% slower |
| 3e-4 | 11.12 s | about 1.0% faster |

So PPO adds zero to about 1% lap time at this budget (TF.js CPU, about 5 s per update). At the default lr the gate (lap >= 0.5% faster at equal completion) reverts every group.

## Measured
Four training-seed sets (policy seed k+1), measured before the PPO fix, so effectively imitation-only:

| run | train seeds | stage0 snapshot (16 held-out) | stage0 final | train wall/cpu | matrix s1 (info) | s2 (pass set) | s3 (info) |
|---|---|---|---|---|---|---|---|
| 0 | 1..24 | 100% at 35 s | 100% | 123 s / 123 s | 99% | 99% (142/144) | 99% |
| 1 | 1001..1024 | 100% at 34 s | 100% | 126 s / 121 s | 99% | 100% (144/144) | 99% |
| 2 | 2001..2024 | 100% at 35 s | 100% | 123 s / 122 s | 90% | 96% (138/144) | 92% |
| 3 | 3001..3024 | 100% at 36 s | 100% | 126 s / 125 s | 99% | 99% (142/144) | 98% |

Stage-2 cells (low / high DR): run 0 asphalt 100/96, carpet 100/100, dirt 100/96; run 2 (weakest) asphalt 96/92, carpet 100/96, dirt 100/92.

Serial `npm run eval` on the merged branch (fixed PPO, run 0 seeds, idle machine): stage-0 snapshot 100% (16/16) at 27 s, final 100% at 90 s; matrix stage 2 99% (142/144), worst cell 96%; PPO 0/4 gated updates kept.

## Still weak
- Weakest pattern is high DR on dirt and carpet; run 2's informational stage-1 cells dropped to 75% (dirt/high) and 79% (carpet/high).
- Browser path trains stage 0 only by imitation; later stages rely on PPO there, which is untested.
