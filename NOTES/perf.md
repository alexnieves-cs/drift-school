# Perf

Measured on a Linux container (CPU only, software GL) with `npm run bench`, `npm run eval` and `npm run test:e2e`. Budgets that depend on wall-clock live in `tests/perf` (warm JIT, best of 7, `PERF_SLACK` multiplier) and are not part of `npm test`.

- One `Vehicle.step` at 50 Hz control (10 substeps) is under the 0.15 ms budget.
- A full `VecEnv` step at n=32 (substeps, 21 rays, track lookup, reward) takes about 0.076 ms per car warm, about 13k car-steps/s in one process. A cold first run is slower (JIT), which is why the budget is benched warm. There is no four-worker SharedArrayBuffer pool, so the SPEC's 20k steps/s is not claimed.
- Training with the PPO update in the loop (8 cars, horizon 48, TF.js CPU backend) runs at about 500 steps/s.
- `generateTrack` median 1.4 ms, p99 9.4 ms. Editor `validateTrack` is asserted under 16 ms.
- Software GL renders the ghost view at about 6-7 fps regardless of car count (1280x720). The SPEC's 50 fps with 64 ghosts at the high preset needs a GPU and was not measured. With the editor open the scene is drawn at a quarter rate so strokes stay responsive.
- In the browser the worker finishes an update every few seconds under software GL because rendering competes for the same CPU; Node runs the same update in under a second.
