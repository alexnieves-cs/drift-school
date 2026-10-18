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
| M2 | Stage 0 lap completion, headless, held-out seeds | >= 80% within 180 s | 100% (16/16) after 6 s | PASS |
| M4 | Hot-swap: all envs on the new track | <= 15 s | not measured | skip |
| M3 | 64 ghost cars frame rate | >= 50 fps | not measured | skip |
| M5 | Eval matrix completion (seeds x surfaces x DR) | >= 70% | not measured | skip |

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
| Wall time / updates read | 300 s / 29 |
| Updates in which a car finished a lap | 21 |
| Best lap, first 5 such updates (median) | 8.98 s |
| Best lap, last 5 such updates (median) | 6.36 s |
| Fastest lap seen | 5.92 s |

### Draw a track and train on it (`node scripts/record-demos.mjs draw`)

Editor verdict: `valid 22.2 ms`. Last adaptation readout in the run: `drawn track: first 64 eps 100% · now 100% · best 7.8 s · procedural 100% → 100% · 89 updates since swap`.

## How to read these

- **Stage-0 completion.** The eval trains with the app's recipe (a pure-pursuit imitation warm start, then one PPO update) on seeds 1 to 4 and scores the policy's deterministic mean action on seeds 101 to 108, two copies each. It passes in 6 s, which means the imitation step alone is enough on stage 0. It says nothing about PPO learning from scratch.
- **Skipped rows.** `npm run eval` prints `skip` for hot-swap latency, ghost-car frame rate and the surface by randomization matrix. Until those sections exist, the README does not claim them.
- **Vehicle step.** The SPEC budget of 0.15 ms covers a full environment step including rays. The vehicle row is the physics alone. The vectorized-env rows cover the whole step (rays, track lookup, reward) per car: at 11,000 car-steps/s that is about 0.09 ms per car, inside the 0.15 ms budget but short of the 20k steps/s four-worker target, which needs a worker farm that does not exist.
- **Hairpin loop.** The loop and its dry and ice laps are covered by `tests/track/hairpin.test.ts`, so the table is a regression test as well as a recording. The driver is the pure-pursuit baseline, not the learned policy. Ice here means track `mu` of 0.4 with the tire `mu0` scaled to match, so the controller's speed profile brakes earlier by design.
- **Live training.** The race table reads the leaderboard while 64 cars train in the worker. An update only reports a lap time if a car finishes one inside that update's rollout, so many updates show no lap. The fall from 9.0 s to 6.4 s is the median of the first and last five updates that did, over about 30 updates. It is one run, on one short stage-0 track, with no confidence interval. The spec's 40 s to 14 s is not reproduced.
- **Draw and train.** The adaptation readout is the app's own text. Completion is counted from the lap flag the environment sets, with 16 environments, and the first-64-episodes figure is the zero-shot reading. The editor verdict time was measured under software rendering and a busy machine, so it is not a clean number for the 16 ms live-edit budget; the clean number is the oval-fixture validation above.
