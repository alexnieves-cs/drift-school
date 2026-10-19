# Benchmarks

Every number on this page was printed by a script in this repository, and the raw JSON is committed in [`docs/media`](media). Nothing here is rounded up, and rows that were not measured say so.

## Where the numbers came from

| | |
| --- | --- |
| Commit | a3316e9, `main` after PRs #1 and #3 merged, plus the files in this pull request |
| Machine | 4 vCPU Intel Xeon @ 2.80 GHz, 15 GB RAM, Linux container, Node v22.22.0 |
| Browser | Chromium headless with ANGLE software WebGL. No GPU, so frame rates from a real laptop are not in this page |
| Date | 2026-10-19 |

Reproduce:

```bash
npm ci
npm run eval && cp eval-report.json docs/media/eval.json   # acceptance table
npm run perf > docs/media/perf.json                   # warmed-up throughput
npm run dev &                                          # the recorder drives the running app
node scripts/record-demos.mjs hairpin race draw       # media plus the JSON below
node scripts/bench-tables.mjs                         # prints the tables on this page
```

Timing numbers move with the machine and its load. `npm run perf` steps the bare environment with a fixed throttle on stage-0 tracks while the machine is otherwise idle, so it reads higher than `scripts/bench-vec.ts` in [NOTES/phys.md](../NOTES/phys.md). That script drives a pure-pursuit car on mixed stage 0, 2 and 3 tracks and was measured with three other heavy jobs running, and it printed about 17,000 car-steps/s. Compare a number only to the commit and the script it came from.

### Acceptance table (`npm run eval`)

|  | Test | Target | Measured | Result |
| --- | --- | --- | --- | --- |
| M2 | Stage 0 lap completion, headless, held-out seeds | >= 80% within 180 s | 100% (16/16) after 20 s; final policy 100% (16/16) after 67 s total | PASS |
| M4 | S-curve stroke becomes a valid track | closed, valid | valid, 44.6 m, 111 control points, validate+spline 2.3 ms | PASS |
| M4 | Hot-swap: all envs report the new track id | <= 15 s | 1.8 s, 16/16 envs on user-5d19ea90 | PASS |
| M4 | Weights unchanged at the swap (hash equal) | hash equal | f24be80b -> f24be80b (equal) | PASS |
| M4 | Completion on the user track after 2 min fine-tune | >= 80% | 100% (was 100%); DR low 100% (was 8%); held-out drawn track 100% (was 100%) | PASS |
| M4 | Procedural completion retained, held-out seeds | >= 90% of pre-swap | 100% vs 100% pre-swap (100% retained, 16 seeds x 2) | PASS |
| M4 | Procedural completion retained, replay-pool seeds | >= 90% of pre-swap | 100% vs 100% pre-swap (100% retained, 24 seeds x 2) | PASS |
| M4 | URL hash round-trips the track | control points within 1 mm, same track id | 111 points, worst 0.00 mm, hash 815 chars, id same | PASS |
| M3 | 64 ghost cars frame rate, Low preset | >= 60 fps | 60.0 fps median (59.9..60.0 over 3 windows), p95 frame 16.7 ms, 40 CPU-ms/frame in software GL | PASS |
| M3 | 64 ghost cars frame rate, High preset | >= 50 fps | 22.1 fps median (21.1..22.7 over 3 windows), p95 frame 50.0 ms, 160 CPU-ms/frame in software GL | FAIL |
| M3 | main-thread long tasks > 50 ms, Low preset | 0 | 0 in 3 x 8 s, worst 0.0 ms | PASS |
| M3 | main-thread long tasks > 50 ms, High preset | 0 | 73 in 3 x 8 s, worst 114.0 ms | FAIL |
| M3 | JS time per frame (hardware independent), Low | <= 8 ms mean and p95 | mean 0.47 ms, p95 0.79, max 6.8 (update 0.02, render submit 0.41, ui 0.04) over 900 frames | PASS |
| M3 | JS time per frame (hardware independent), High | <= 8 ms mean and p95 | mean 1.01 ms, p95 2.45, max 6.2 (update 0.04, render submit 0.85, ui 0.12) over 533 frames | PASS |
| M3 | draw calls / triangles per frame | <= 50 / 100000 | Low 13 / 6656, High 13 / 6656 | PASS |
| M3 | leaderboard rows == PPO updates | equal | 24 rows (24 in the DOM) vs 24 metrics updates, gens 1..N contiguous | PASS |
| M3 | click row replays a deterministic ghost lap | identical poses | click -> replay -> end x2: ok; two probes identical (30 steps); max error vs tape 2.2e-16 | PASS |
| M5 | Eval matrix completion (seeds x surfaces x DR) | >= 70% | 99% (143/144), worst cell 96% | PASS |
| M5 | Pure-pursuit baseline (upper bound), stage 1, surface x DR | >= 70% in every cell | 100% overall, worst 100% (asphalt/low), 120 laps | PASS |
| M5 | Pure-pursuit baseline (upper bound), stage 2, surface x DR | >= 70% in every cell | 100% overall, worst 100% (asphalt/low), 120 laps | PASS |
| M5 | Pure-pursuit baseline (upper bound), stage 3, surface x DR | >= 70% in every cell | 100% overall, worst 100% (asphalt/low), 120 laps | PASS |
| M5 | Car model at the corners of the high-DR box (oracle controller), stage 1 | >= 70% on every surface | 99% overall, worst 98% (carpet), 120 laps | PASS |
| M5 | Car model at the corners of the high-DR box (oracle controller), stage 2 | >= 70% on every surface | 98% overall, worst 95% (dirt), 120 laps | PASS |
| M5 | Car model at the corners of the high-DR box (oracle controller), stage 3 | >= 70% on every surface | 97% overall, worst 95% (dirt), 120 laps | PASS |

### Throughput (`npm run perf`, one Node process, warmed up)

| Measure | Result | SPEC budget |
| --- | --- | --- |
| Vehicle step (10 substeps) | 0.0159 ms median | < 0.15 ms |
| vecEnv n=8 | 31,978 car-steps/s | >= 6,000 per worker |
| vecEnv n=32 | 37,892 car-steps/s | >= 6,000 per worker |
| vecEnv n=64 | 39,392 car-steps/s | >= 6,000 per worker |
| Track generation | 1.06 ms median | < 20 ms |
| Validate the oval fixture | 0.71 ms median | < 16 ms |

### Pure-pursuit baseline on the hairpin loop (real vehicle model)

| Grip mu | Finished | Lap time | Max offset from centerline | Top speed |
| --- | --- | --- | --- | --- |
| 1.10 | yes | 16.4 s | 0.27 m | 5.63 m/s |
| 0.40 | yes | 22.6 s | 0.21 m | 5.04 m/s |

### Live training, 64 cars (`node scripts/record-demos.mjs race`)

| Measure | Value |
| --- | --- |
| Wall time / updates read | 241 s / 182 |
| Updates in which a car finished a lap | 130 |
| Best lap, first 5 such updates (median) | 8.84 s |
| Best lap, last 5 such updates (median) | 9.14 s |
| Fastest lap seen | 8.04 s |

### Draw a track and train on it (`node scripts/record-demos.mjs draw`)

Editor verdict: `valid 7.9 ms`.

| Measure | Value |
| --- | --- |
| Swap latency (click to every env on the drawn track) | 1.6 s |
| Weights unchanged at swap | true |
| Zero-shot completion (no update yet) | 100% |
| Fine-tuning started after | 11 s |
| Completion on the drawn track at 180 s | 100% |
| Best lap on the drawn track | 12.10 s |
| Learner steps since the swap | 102 |

Last panel text: `fine-tuning: 14/16 envs on the new track | swap latency 1.6 s, weights unchanged | zero-shot 100% of 64 episodes | steps 102 (94 PPO, 306 imitation) · completion 100% · best lap 12.1 s`.

## How to read these

- **Stage-0 completion.** The eval runs the app's training recipe: DAgger imitation of the pure-pursuit controller through stages 0 to 3, then a short PPO phase whose updates are reverted unless they help on validation seeds. It scores the policy's deterministic action on held-out seeds 101 to 116. The stage-0 snapshot reaches 100% after 20 s and the whole recipe after 67 s. The skill comes from the imitation, so this says nothing about PPO learning from scratch.
- **Hot-swap rows.** They run in Node with the same recipe. "1.8 s" is the time for all 16 environments to finish their running episode and restart on the drawn track. Completion on the drawn S-curve was already 100% before the fine-tune, so the "after 2 min" row mainly shows the fine-tune does no harm. Under low domain randomization the same track went from 8% to 100%. The browser run below swapped in 1.6 s.
- **Matrix rows.** The learned-policy matrix is 12 held-out stage-2 seeds by three surfaces by two randomization levels, 144 episodes. The pure-pursuit and oracle rows are upper bounds that check the car model and the tracks, not the policy.
- **Frame-rate rows.** They were measured in headless Chromium with SwiftShader software rendering on the same 4 cores as the learner. Low holds 60 fps with no long tasks. High reaches 22 fps against a 50 fps bar, and its long tasks are the compositor waiting on the software rasterizer, not JavaScript: JavaScript stays near 1 ms per frame. Nobody has measured either preset on a GPU yet.
- **Vehicle step.** The SPEC budget of 0.15 ms covers a full environment step including rays. The vehicle row is the physics alone. The vectorized-env rows cover the whole step (rays, track lookup, reward) per car, about 0.025 ms each at 39,000 car-steps/s. That is one process with no policy in the loop; with TF.js inference and PPO updates, training runs far slower. The SPEC's four-worker farm does not exist.
- **Hairpin loop.** The loop and its dry and ice laps are covered by `tests/track/hairpin.test.ts`, so the table is a regression test as well as a recording. The driver is the pure-pursuit baseline, not the learned policy. Ice here means track `mu` of 0.4 with the tire `mu0` scaled to match, so the controller's speed profile brakes earlier by design.
- **Live training.** The race table reads the leaderboard while 64 cars train in the worker on stage-0 tracks. An update only reports a lap time if a car finishes one inside that update's rollout. Over 182 updates the best lap did not improve: the median of the first five lap-finishing updates was 8.8 s and of the last five 9.1 s. The policy starts from the imitation recipe, and the in-browser PPO does not make it faster in four minutes. The spec's 40 s to 14 s story is not reproduced. Two recordings of this run printed identical numbers, because training is seeded.
- **Draw and train.** The adaptation numbers are the app's own panel. The cars drove the drawn hairpin at 100% before any update, and the best lap on it fell from 13.7 s at the first lap-finishing reading to 12.1 s over three minutes of mixed PPO and imitation. The editor verdict time was measured under software rendering on a busy machine, so the clean number for the 16 ms live-edit budget is the oval-fixture validation above.
