# Perf

Measured on this machine with Vitest, single thread.

- One `Vehicle.step` at 50 Hz control (10 substeps) is under the 0.15 ms budget. See `tests/sim/vehicle.test.ts` and `tests/perf/budget.bench.ts`.
- `VecEnv` at n=32 measured about 6800 car-steps/s here (asserted above 6k). That is one process, not four workers. A PPO update on 8 cars measured about 700 steps/s with the TF.js CPU backend.
- Headless Chromium, low preset, 64 ghost cars: about 34 fps. The spec's 50 fps high-preset number is for a mid-range laptop and was not measured here.
- A PPO update on the policy network is far slower than the env. Training throughput with the network in the loop stayed near 800 steps/s in earlier probes. The 2 s/update WebGL budget is not met on the CPU backend.
- Editor `validateTrack` on a stadium is asserted under 16 ms.
- Render fps is measured in Playwright (`tests/e2e/app.spec.ts`) on the ghost view. The spec asks for 50 fps with 64 ghosts at the high preset. The e2e floor checked in CI-less Playwright is 30 fps, because software GL on a headless box is not a mid-range laptop.
