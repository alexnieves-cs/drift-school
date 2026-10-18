# Benchmarks

Every number on this page was printed by a script in this repository, and the raw JSON is committed in [`docs/media`](media). Nothing here is rounded up, and rows that were not measured say so.

## Where the numbers came from

| | |
| --- | --- |
| Commit | 7d81d7f, the head of PR #1 when measured (the sim and RL fixes), plus the files in this pull request |
| Machine | 4 vCPU Intel Xeon @ 2.80 GHz, 15 GB RAM, Linux container, Node v22.22.0 |
| Browser | Chromium headless with ANGLE software WebGL. No GPU, so frame rates from a real laptop are not in this page |
| Date | 2026-10-18 |

Reproduce:

```bash
npm ci
npm run eval && cp eval-report.json docs/media/eval.json   # acceptance table
npm run perf > docs/media/perf.json                   # warmed-up throughput
npm run dev &                                          # the recorder drives the running app
node scripts/record-demos.mjs hairpin race draw       # media plus the JSON below
node scripts/bench-tables.mjs                         # prints the tables on this page
```

Timing numbers move with the machine. The same checks printed 3,700 to 4,800 car-steps/s on an earlier commit in this container and about 11,000 after the grid-culled ray code landed, so compare a number to the commit it was measured on.

### Acceptance table (`npm run eval`)

|  | Test | Target | Measured | Result |
| --- | --- | --- | --- | --- |
| M2 | Stage 0 lap completion, headless, held-out seeds | >= 80% within 180 s | 100% (16/16) after 5 s | PASS |
| M4 | Drawn S-curve is valid | valid | valid | PASS |
| M4 | URL hash round-trips the track | same points (1 mm) | 1826 chars, same points | PASS |
| M4 | Hot-swap: all envs report the new track id | <= 15 s | 8/8 envs in 3 ms (in process; the browser run adds the wait for the running update) | PASS |
| M4 | Weights unchanged at swap | hash equal | hash equal | PASS |
| M4 | Completion on the drawn track after 2 min | >= 80% | 100% (16/16); zero-shot was 100%; 167 updates in 121 s | PASS |
| M4 | Procedural completion retained (no forgetting) | >= 90% of pre-swap | 100% vs 100% before = 100% retained | PASS |
| M3 | 64 ghost cars frame rate | >= 50 fps | not measured here | skip |
| M5 | Eval matrix completion (8 seeds x asphalt/carpet/dirt x DR low/high) | >= 70% | 99% (143/144) | PASS |

### Throughput (`npm run perf`, one Node process, warmed up)

| Measure | Result | SPEC budget |
| --- | --- | --- |
| Vehicle step (10 substeps) | 0.0243 ms median | < 0.15 ms |
| vecEnv n=8 | 9,918 car-steps/s | >= 6,000 per worker |
| vecEnv n=32 | 11,257 car-steps/s | >= 6,000 per worker |
| vecEnv n=64 | 11,169 car-steps/s | >= 6,000 per worker |
| Track generation | 1.20 ms median | < 20 ms |
| Validate the oval fixture | 0.89 ms median | < 16 ms |

### Pure-pursuit baseline on the hairpin loop (real vehicle model)

| Grip mu | Finished | Lap time | Max offset from centerline | Top speed |
| --- | --- | --- | --- | --- |
| 1.10 | yes | 16.4 s | 0.27 m | 5.63 m/s |
| 0.40 | yes | 22.6 s | 0.21 m | 5.04 m/s |

### Live training, 64 cars (`node scripts/record-demos.mjs race`)

| Measure | Value |
| --- | --- |
| Wall time / updates read | 301 s / 28 |
| Updates in which a car finished a lap | 20 |
| Best lap, first 5 such updates (median) | 8.98 s |
| Best lap, last 5 such updates (median) | 6.18 s |
| Fastest lap seen | 5.92 s |

### Draw a track and train on it (`node scripts/record-demos.mjs draw`)

Editor verdict: `valid 12.9 ms`. Last adaptation readout in the run: `drawn track: first 64 eps 100% · now 100% · best 7.8 s · procedural 100% → 100% · 84 updates since swap`.

## How to read these

- **Stage-0 completion.** The eval trains with the app's recipe (a pure-pursuit imitation warm start, then one PPO update) on seeds 1 to 4 and scores the policy's deterministic mean action on seeds 101 to 108, two copies each. It finishes in about 5 s, which means the imitation step alone is enough on stage 0. It says nothing about PPO learning from scratch.
- **Hot-swap and matrix rows.** They run in Node, in process, on the same cloned-then-fine-tuned policy. "8/8 envs in 3 ms" is the in-process swap and leaves out the wait for the running update in the browser, so it is not a user-visible latency. The matrix is 8 seeds by three surfaces by two randomization levels, 144 episodes.
- **Skipped row.** The 64-ghost frame rate needs a GPU. `E2E_STRICT_PERF=1 npm run test:e2e -- m3` asserts it on a machine that has one. Until it has been run, the README does not claim it.
- **Vehicle step.** The SPEC budget of 0.15 ms covers a full environment step including rays. The vehicle row is the physics alone. The vectorized-env rows cover the whole step (rays, track lookup, reward) per car: at 11,000 car-steps/s that is about 0.09 ms per car, inside the 0.15 ms budget but short of the 20k steps/s four-worker target, which needs a worker farm that does not exist.
- **Hairpin loop.** The loop and its dry and ice laps are covered by `tests/track/hairpin.test.ts`, so the table is a regression test as well as a recording. The driver is the pure-pursuit baseline, not the learned policy. Ice here means track `mu` of 0.4 with the tire `mu0` scaled to match, so the controller's speed profile brakes earlier by design.
- **Live training.** The race table reads the leaderboard while 64 cars train in the worker. An update only reports a lap time if a car finishes one inside that update's rollout, so many updates show no lap. The fall from 9.0 s to 6.2 s is the median of the first and last five updates that did, over 28 updates. It is one run, on one short stage-0 track, with no confidence interval. The spec's 40 s to 14 s is not reproduced.
- **Draw and train.** The adaptation readout is the app's own text. Completion is counted from the lap flag the environment sets, with 16 environments, and the first-64-episodes figure is the zero-shot reading. The editor verdict time was measured under software rendering and a busy machine, so it is not a clean number for the 16 ms live-edit budget; the clean number is the oval-fixture validation above.
