# Perf

Measured on a Linux container (CPU only, software GL) with `npm run bench`, `npm run eval` and `npm run test:e2e`. Budgets that depend on wall-clock live in `tests/perf` (warm JIT, best of 7, `PERF_SLACK` multiplier) and are not part of `npm test`.

- One `Vehicle.step` at 50 Hz control (10 substeps) is under the 0.15 ms budget.
- A full `VecEnv` step at n=32 (substeps, 21 rays, track lookup, reward) takes about 0.076 ms per car warm, about 13k car-steps/s in one process. A cold first run is slower (JIT), which is why the budget is benched warm. The four-worker SharedArrayBuffer farm now exists (entry below), but the SPEC's 20k steps/s on 4 workers is not claimed as a speedup.
- Training with the PPO update in the loop (8 cars, horizon 48, TF.js CPU backend) runs at about 500 steps/s.
- `generateTrack` median 1.4 ms, p99 9.4 ms. Editor `validateTrack` is asserted under 16 ms.
- Software GL renders the ghost view at about 6-7 fps regardless of car count (1280x720). The SPEC's 50 fps with 64 ghosts at the high preset needs a GPU and was not measured. With the editor open the scene is drawn at a quarter rate so strokes stay responsive.
- In the browser the worker finishes an update every few seconds under software GL because rendering competes for the same CPU; Node runs the same update in under a second.

## Env farm (from-scratch-rl branch)

- `src/workers/envFarm.ts` splits `total` slots into contiguous shards (default 4). Each shard is a `VecEnv` in its own worker; actions, observations, rewards, flags, poses, step info and finished-episode records live in one SharedArrayBuffer (`src/workers/envFarmLayout.ts`). A step is one `Atomics.store` + `notify` on a sequence word and one wait per shard, with no messages.
- Determinism: `VecEnv` takes a `slotOffset`, so shard slot i seeds its noise stream as global slot `lo + i`. `tests/workers/envFarm.test.ts` checks 16 slots under high domain randomization for 700 steps: inline farm (1, 3, 4, 5 shards) and 4 Node worker threads (blocking and async steps) match one `VecEnv(16)` byte for byte, episode records included.
- Throughput, `npx vite-node scripts/farm-bench.ts`, fixed actions, Node worker threads, 2026-11-25 on a shared 4-core machine at load average 7.2 to 7.8 (two training runs and other agents): n=64, 4 workers 19.4k car-steps/s against 29.4k in one process; n=64, 2 workers 27.1k against 26.9k; n=256, 4 workers 33.1k against 33.2k. So the farm reaches the SPEC's 20k figure at n=256, but one process does too on this machine, and no speedup was shown. An idle machine or a browser run is needed before claiming one.
- Not done: the page trainer does not use the farm yet, and the farm has no per-episode sampler (a slot restarts on its own track and config).

