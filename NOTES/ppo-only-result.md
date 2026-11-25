# Result: PPO from scratch on stage 0 (control study)

Pre-registration: `NOTES/ppo-only-prereg.md`, committed in 8cc93b7 before either seed was trained. Logs: `NOTES/ppo-only-runs/seed1.jsonl` and `seed2.jsonl`. Scored once, from the two `kind: "final"` records, by the pre-registered rules.

## Verdict

**Learns to lap (H1 holds).** Both seeds drove at least 13 of 16 clean laps on the held-out stage-0 tracks after 300 updates: seed 1 drove 16 of 16, seed 2 drove 15 of 16.

The pre-registered prior expectation (H0 more likely) was wrong.

## Numbers

Held-out stage 0, seeds 101-116, deterministic mean action, one lap per case, after update 300:

| | Seed 1 | Seed 2 |
| --- | --- | --- |
| **Clean laps (primary)** | **16/16** | **15/16** |
| Mean clean lap fraction (secondary) | 1.000 | 0.951 |
| Completed (M2 metric, counts wall-riding) | 16/16 | 16/16 |
| Wall-contact steps per case | 0 on all 16 | 0 on 15; 1 on seed 115, at 22% of the lap |
| Held-out lap times | 6.24 to 7.38 s | 6.26 to 7.46 s |
| Untrained policy (update 0): clean laps, clean lap fraction, completed | 0/16, 0.165, 0/16 | 0/16, 0.095, 16/16 |
| Steering-centred reference (throttle 0.5) | 0/16 clean, 16/16 completed, 0.134 | same |
| Training CPU / wall | 1,409 s / 1,935 s | 1,508 s / 2,043 s |
| Env steps | 960,000 | 960,000 |
| Best training lap | 6.28 s | 6.36 s |
| Final weight hash | 81d1249e | 39d9ab67 |

For scale, the analytic reference lap times (`tRef`) of stage-0 tracks are 6.0 to 7.4 s. The untrained seed-2 policy "completes" 16 of 16 laps with 18 to 42 wall-contact steps each, which is the wall-riding that made plain completion unusable as the metric.

## Informational, not part of the verdict

Held-out probes every 40 updates (the scored policy is the one after update 300; no checkpoint was picked):

| Update | 40 | 80 | 120 | 160 | 200 | 240 | 280 | 300 (scored) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Seed 1 clean laps | 16 | 16 | 16 | 14 | 14 | 12 | 16 | 16 |
| Seed 2 clean laps | 16 | 2 | 12 | 15 | 16 | 13 | 10 | 15 |

- Both seeds already drove 16 of 16 clean held-out laps at the first probe, after 40 updates (128,000 env steps, about 190 to 206 s of training CPU). This was not pre-registered as a pass line and is not a claim that PPO learns stage 0 in about 3 minutes.
- The policy is not stable from update to update. Seed 2 fell to 2 of 16 at update 80 and to 10 of 16 at update 280. Had the study stopped at a different update, the verdict could have been different. The pass is for the policy after 300 updates, not for every policy along the way.
- Training episodes first finished a lap at update 11 (seed 1) and update 3 (seed 2); training completion was 1.0 from then on, which includes wall-riding laps. Policy entropy fell from about 0.24 to 0.009 (seed 1) and -0.087 (seed 2), so exploration was close to gone by the end.

## Deviations from the pre-registration

- Both processes ran under `nice -n 5`, which only lowers scheduling priority on the shared machine. It does not change what is computed.
- While the runs were going, other work on the same branch edited source files that the script imports (`src/sim/vecEnv.ts`, `src/sim/car/vehicle.ts`, `src/sim/reward.ts`, `src/shared/types.ts`; each change keeps the old behaviour by default). The two processes loaded all of their modules when they started at 22:43:25 UTC, after commit 8cc93b7, and the script has no dynamic imports, so they ran the registered code. `scripts/rl-ppo-only.ts` and `src/eval/harness.ts` are unchanged since 8cc93b7. Short benchmarks and tests also ran on the machine during the study; that affects wall time only.
- None otherwise: seeds, commands, update count, metric and scoring are as registered. Neither seed hit the wall-clock cap (4,500 s).

## What this does and does not show

- It shows that the repo's PPO, from a fresh policy with no imitation, learns clean laps of unseen stage-0 tracks in this configuration on CPU, on two of two seeds. The earlier statements that stage 0 "does not learn laps from a random policy at this training rate" (`docs/ACCEPTANCE.md`) and that the from-scratch control had not been run (`NOTES/rl.md`) are superseded for stage 0.
- The likely reason earlier from-scratch attempts failed is the scenario-mode reset described in the pre-registration: episodes capped at 4 s could never finish a lap. That is a reading of the code, not something this study tested.
- Stages 1 to 4, domain randomization, surfaces and the user-track hot-swap were not tested. The shipped recipe (`trainRecipe`) is unchanged and still trains by imitation first.
- Two seeds is a small sample, and the probe table shows the policy swings between checks. A study of stability (more seeds, a fixed evaluation schedule, or an entropy floor) would need its own pre-registration.
- The weight hashes are logged so a re-run can be checked against them; the runs were not repeated here.
