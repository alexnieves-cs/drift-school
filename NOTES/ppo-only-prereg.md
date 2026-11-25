# Pre-registration: PPO from scratch on stage 0 (control study)

Written and committed before the run. The result goes in `NOTES/ppo-only-result.md`, scored once against the pass line below, whatever it is. Nothing in this file changes after the run starts; any deviation is listed in the result file.

## Question

`NOTES/rl.md` says the lap skill is almost all DAgger imitation of the pure-pursuit driver, and that `scripts/rl-ppo-only.ts`, the from-scratch control, has never been run. Does the repo's PPO, with no imitation at all, learn to drive clean laps of stage-0 tracks it has never seen?

## Hypotheses

- **H1.** PPO from a fresh policy (no DAgger, no pure-pursuit labels, no imitation warm start) drives a clean lap (one lap, no wall contact) on at least 80% of the 16 held-out stage-0 cases within the compute budget below.
- **H0.** It does not.

Prior expectation, stated so it can be checked: H0 is more likely. Evidence before this run: the hot-swap study in `NOTES/swap.md` had a PPO-only fine-tune arm that ended at 0% on a drawn track after 2 minutes (from an imitation base, not from scratch), and `docs/ACCEPTANCE.md` says stage 0 "does not learn laps from a random policy at this training rate" without a measured run behind it. This study is that measured run.

## Why the metric is clean laps, not plain completion

While setting up the metric (before any study seed was trained or evaluated), constant actions were run through the held-out evaluation. With the steering centred and throttle 0.5 or more, the car completes **16 of 16** held-out stage-0 laps: the tracks are wide, there is no termination for wall contact, and the wall impulse bounces the car round. Plain completion, the M2 acceptance metric, therefore does not show that a policy drives on stage 0. The same calibration on the reference drivers:

| Policy (held-out seeds 101-116, stage 0) | Completed | Clean laps | Mean clean lap fraction |
| --- | --- | --- | --- |
| Steering centred, throttle 0.5 | 16/16 | 0/16 | 0.134 |
| Steering centred, throttle 1.0 | 16/16 | 0/16 | 0.135 |
| Pure-pursuit baseline | 16/16 | 16/16 | 1.000 |
| Imitation recipe, stage 0 only, seed 1 (trained in 29 s) | 16/16 | 16/16 | 1.000 |

So the primary metric counts a lap only if the car never touches a wall during the episode. It separates wall-riding from driving, and the policies the repo already has score 16/16 on it.

A note on the held-out set: the 16 seeds give 9 distinct track shapes at stage 0 (seeds 101-103 build the same track, as do 107-111 and 112-113), because neighbouring seeds can map to the same geometry. None of the 9 matches any of the 12 distinct training shapes (seeds 1-24). The set is kept as is, so the numbers stay comparable with the M2 row; repeated shapes simply weigh more.

## What runs

`scripts/rl-ppo-only.ts` at the commit that adds this file, once per seed:

```bash
npx vite-node scripts/rl-ppo-only.ts -- --seed=1 --updates=300 --mode=pool --out=NOTES/ppo-only-runs/seed1.jsonl
npx vite-node scripts/rl-ppo-only.ts -- --seed=2 --updates=300 --mode=pool --out=NOTES/ppo-only-runs/seed2.jsonl
```

The two processes run in parallel (two cores of a shared 4-core Linux container, Node 22, TF.js CPU backend; no GPU here).

Configuration, all from the repo's defaults unless marked:

| Setting | Value |
| --- | --- |
| Policy | `GaussianPolicy`: MLP 48-128-128 tanh, tanh-squashed Gaussian, learned log-std starting at [-1.4, -1.2], clipped to [-3.2, -0.4]; separate 48-128-128 value net. Throttle mean bias initialised to 0.7 (repo default). Weights from the policy seed, never TF.js's RNG. |
| PPO | `Trainer.update`: 16 envs, horizon 200, 2 epochs, minibatch 256, clip 0.2, Adam lr 3e-4 (value Adam 1e-3), GAE lambda 0.95, gamma 0.99, entropy 0.01, advantage normalisation, running reward scale, global grad-norm clip 0.5. |
| Env | Stage-0 tracks from training seeds 1-24, one lap per episode, no DR, native surface grip. Reward and termination as in `src/sim/reward.ts` (wall contact is penalised, not terminal). Policy acts every 50 Hz control step. |
| Curriculum | Pinned to stage 0 (promotion disabled in the script). |
| Imitation | None. `Trainer.imitate`, DAgger and `fastfit` are never called. |
| Seeds | Policy seeds **1** and **2**. Training track seeds 1-24. Held-out track seeds 101-116 (`STAGE0_HELD_OUT`), never trained on; the script calls `assertTrainSeeds`. |

### Changes to the script before registration

As first written, the script put the trainer in scenario mode, where `Trainer.update` resets every env to a fresh scenario at the start of each 200-step rollout. A training episode could therefore never last more than 4 s of driving. Stage-0 tracks are 33-42 m long and their analytic reference lap time (`tRef`, the fastest pace the speed profile allows) is 6.0-7.4 s, so no training episode could ever finish a lap or see the back half of a track. That is a test PPO cannot pass by construction, so it is not a useful control.

The revised script (`--mode=pool`, the default and the mode run here) uses the trainer's track-pool path instead: episodes run until they terminate or hit the step cap (about 2,000-2,500 steps on these tracks), across update boundaries, and each new episode draws a track from the 24 training tracks. `--mode=scenario` keeps the old behaviour; it is not run in this study. Other changes: a fixed update count instead of a CPU-seconds budget (so the run is reproducible and held-out evaluations cannot eat into training), a `--seed` flag, a JSONL log, a wall-clock safety cap, and the clean-lap fields from `evaluateCases` (`src/eval/harness.ts`, which gained `cleanLaps`, `contactSteps`, `lapFractions` and `cleanFractions`; these are measured passively and do not change how an episode is stepped).

## Compute budget

300 PPO updates per seed = 300 x 16 envs x 200 steps = **960,000 env steps per seed** (19,200 s, about 5.3 hours, of simulated driving per seed).

Calibration on this machine (6 updates, seed 99, timing only, no held-out evaluation looked at): 4.6 to 5.3 s of CPU per update, so about 25 minutes of CPU per seed. Wall time is higher because the machine is shared with other jobs; expected 30 to 45 minutes with both seeds in parallel. A 2-update smoke test with seed 99 was also run to check the script works; its numbers are not part of the study.

For scale: the imitation recipe reaches 16/16 clean laps on the same held-out tracks after about 29 s of training (table above), and SPEC M2 asks for stage-0 laps within 3 minutes. This budget is about 50x the imitation recipe's stage-0 time and about 8x the SPEC's 3 minutes.

Safety cap: `--wall-cap=4500` (75 min, the script default). If a seed hits it, the run stops, is scored at the update it reached, and the shortfall is reported as a deviation. Total wall time stays well under 90 minutes.

## Metric

Per seed, after update 300, scored once, all from `stage0Completion` (`src/eval/stage0.ts`, the same evaluation as the M2 acceptance row): 16 held-out stage-0 cases, deterministic mean action (`actMean`), one episode each, one lap.

- **Primary: clean laps.** Cases where the car finishes one lap with no wall contact at any step (`cleanLaps` out of 16).
- **Secondary: mean clean lap fraction.** Per case, the furthest point along the centerline reached before the first wall contact, as a fraction of a lap (1 for a clean lap), averaged over the 16 (`cleanFractions`).
- **Also reported:** plain completion (the M2 number, fooled by wall-riding as shown above), mean lap fraction ignoring walls, wall-contact steps per case, and all of these for the untrained policy (update 0, same seed) and for the steering-centred reference, both logged by the script before the first update.

Informational, not used for the verdict: held-out probes every 40 updates (no early stopping, no picking the best checkpoint; the scored policy is the one after the last update), per-update training stats (episodes, training completion, mean return, best training lap, entropy, KL, value loss), CPU and wall time.

## Pass line and what counts as "learns"

| Verdict | Rule |
| --- | --- |
| **Learns to lap (H1 holds)** | Both seeds drive at least 13 of 16 clean laps (80%). |
| **Seed-dependent** | One seed reaches 13 of 16 clean laps, the other does not. No claim that PPO learns to lap. |
| **Learns something, not laps** | Neither seed reaches 13 of 16, but on both seeds the final mean clean lap fraction is at least 0.25 and above that seed's untrained value. 0.25 is almost twice the steering-centred reference (0.134): a car only gets there by steering into the first bend without touching a wall. |
| **No measurable learning (H0, strong form)** | None of the above. |

The verdict is read off the two `kind: "final"` records in the JSONL logs, by these rules only.

## What would not follow from the result

- A fail does not show that PPO cannot learn this task. It shows that this PPO configuration does not, at this budget, on CPU. Other hyperparameters (exploration noise, reward shaping such as ending episodes on wall contact, a longer horizon), a larger budget or a GPU might. Tuning after seeing the result would be a new study with its own pre-registration, not a re-run of this one.
- A pass would show from-scratch learning on stage 0 only. Stages 1-3, DR and the user-track hot-swap are not tested here.
- Either way, the recipe the app ships (`trainRecipe`) is unchanged by this study.
