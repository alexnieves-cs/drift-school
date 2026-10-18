# Physics, drift and simulation throughput (acc/phys)

## Bottom line

- The simulation is not what limits the learner. The pure-pursuit baseline completes 100% of laps in every
  surface x DR cell of the M5 matrix on stage 1, 2 and 3 (10 track seeds x 2 DR draws per cell, 20 laps per cell).
  The learner's 13% on high-DR cells is a policy problem, not a car-model or surface problem.
- `VecEnv.step` went from about 3.3k to 11k-22k car-steps/s (single process, n=32 and n=64), with trajectories
  bit-for-bit identical. See "Throughput".
- Drift is reachable and recoverable under high DR (100 of 100 draws) with `driftSetup()`, and the old drift test
  turned out to be a spin to a standstill (see "Drift").
- One sensor bug found and left as is (default behavior is what the policy trains on): `RayField` rays are wrong
  for 3-6% of rays. Opt-in fix: `SENSOR_FLAGS.nearestRay = true`. See "Ray bug".

## 1. Physics sanity on the M5 matrix

`npm run eval -- --only=physics` (`src/eval/physics.ts`, registered as `physics` in `scripts/eval.ts`) drives the
imitation-teacher controller (`VecEnv.baselineActions()`, i.e. `purePursuit` unchanged) through
`casesFor` / `withSurface` with the same per-case seeds and step cap as `evaluateCases`. Every episode end is
classified (lap, wall, off-track, wrong way, stuck, flip, timeout). `scripts/phys-matrix.ts` runs the same
matrix with knobs (`--stages --surfaces --drs --seeds --copies --corners --oracle --drift`), and
`scripts/phys-debug.ts` replays failing cases with a trace.

Baseline completion, teacher controller, 10 seeds x 2 DR draws per cell (20 laps each):

| stage | surface | DR low | DR high |
|---|---|---|---|
| 1, 2, 3 | asphalt | 100% | 100% |
| 1, 2, 3 | carpet | 100% | 100% |
| 1, 2, 3 | dirt | 100% | 100% |

(18 cells, 360 laps, 0 failures.) A wider sweep, 30 seeds x 2 draws, DR off and high, stages 0, 3, 4, on asphalt, dirt
and ice (ice is outside the matrix): 97%-100% everywhere, the misses being single laps (stage 0 ice/high 97%,
stage 3 dirt/high 98%, stage 4 dirt/high 98%, stage 4 ice/high 97%).

### Stress: the corners of the high-DR box

`sampleCarCorner` (new, `src/sim/dr.ts`) puts every randomized property at one end of its spec range by coin flip
(mass +-25%, CG +-30%, mu0 0.55 or 1.4, load sensitivity, B/C/E +-15%, spring/damper +-25%, motor +-20%, servo rate
+-20%, steering offset +-2 deg), with latency 2 ticks and noise x2. 6 draws x 16 seeds per cell, surface scaling on top:

| stage | surface | teacher controller | oracle controller |
|---|---|---|---|
| 1 | asphalt / dirt / ice | 100% / 96% / 75% | 100% / 100% / 99% |
| 3 | asphalt / dirt / ice | 100% / 93% / 80% | 98% / 98% / 97% |
| 4 | asphalt / dirt / ice | 98% / 82% / 72% | 96% / 97% / 96% |

Failures of the teacher at the corners were all "stuck" (pinned against a wall facing it, throttle held, no reverse)
and "wrong way" after a spin. Causes, in order:

1. The speed target uses the surface mu on the track samples but not the car's own mu0, so a 0.55-mu0 car on dirt
   drives as fast as a 1.4-mu0 one, understeers into a wall and sits there.
2. No recovery: `purePursuit` never reverses.

Both are controller issues. `purePursuit` now takes optional `BaselineOpts { gripAware, recover }` (plus a
`BaselineMemory` the VecEnv keeps per car). With them on (the "oracle": it reads the true mu0, which a learner cannot)
the box corners complete 96-100% in every cell, including ice. Nothing in the car model (no unrecoverable spin, flip,
NaN or tire singularity: 0 flips in all runs) needed a fix, so `vehicle.ts` physics is untouched. The default
`baselineActions()` is unchanged on purpose: `src/rl/trainer.ts` imitates it as the warm start, and a teacher that
depends on an input the policy cannot see would be a worse teacher.

The `physics` eval section reports both: the teacher on the real matrix (rows per stage) and the oracle at the box
corners (rows per stage), all passing at >= 70%.

### What is still weak

- Effective grip is the product of the DR mu0 draw and the surface: `car.surfaceMu = sample.mu / 1.05` multiplies
  `tire.mu0`. High DR on dirt therefore spans mu 0.37-0.93, and on ice 0.18-0.47, below the spec's 0.5-1.4 total
  range for mu0 (the spec has surfaces replace the draw, not scale it). The baseline copes; a learner has to
  infer a wider grip range than the spec intends. Not changed, because it would move every number the learner was
  measured against. If it matters, clamp the product in `sampleCar`/`VecEnv` to >= 0.5.
- `tire.mu0` is jittered by DR but the baseline's speed profile cannot see it (teacher), so the baseline is the
  upper bound only on a car it is allowed to know.

## 2. Drift (demo moment 4)

Finding: the scripted test in `tests/sim/vehicle.test.ts` ("drifts with low rear grip and recovers under
countersteer") is not discriminating. Its setting (`rearMuScale` 0.42, rear B 4.2) sends the car around in a
spin-out: rear slip peaks at 82 deg and the speed goes to 0.0 m/s. It passes because it checks only peak slip > 15 deg
and a recovery condition. It is left untouched (it must keep passing) and the new tests are stricter.

New (`tests/sim/drift.test.ts`, script in `tests/sim/driftScript.ts`): a closed-loop corner. Cruise at 4.5 m/s, turn
in and ramp steer with throttle pinned, hold the slide once deep with countersteer (front wheels along the velocity
vector) and throttle modulation, then straighten with countersteer and the throttle mostly lifted. Pass means:
rear slip above 15 deg for >= 0.5 s, never beyond 75 deg, speed never below 1.5 m/s while sliding (a drift, not a
spin to rest), then slip < 5 deg, yaw rate < 1 rad/s and speed > 1 m/s held 0.5 s.

Numbers (scripted drift on `driftSetup(defaultCarParams())`, no DR): peak rear slip 56 deg, 1.2 s sliding, minimum
speed 4.0 m/s while sliding, recovered in 0.82 s at 4.3 m/s. Under DR, 100 draws each:

| rearMuScale (RWD) | low DR | high DR |
|---|---|---|
| 1.0 | 100% | 98% |
| 0.95 | 100% | 100% |
| 0.90 (`driftSetup`) | 100% | 100% |
| 0.85 | 100% | 100% |
| 0.80 | 100% | 97% |
| 0.70 | 98% | 76% (spins out) |
| 0.42 (old test) | spins out | spins out |

`driftSetup(base)` (`src/sim/dr.ts`): RWD, `split` 0, `rearMuScale` 0.9. It sits in the middle of the 0.85-0.95
band that is 100% under high DR. Negative controls in the tests: a crawling corner on the stock 4WD tune never
reaches 15 deg (`grip`), and the old test's rear grip is classified `spun-out`. The scripted recovery succeeded in
every draw that reached a drift.

Cost: with `driftSetup` the teacher baseline completes less on dirt at stage 4 (low DR 75%, high DR 65%; oracle 75%
and 70%, off-track in the hairpins) and is 90-100% elsewhere. It is an opt-in setup for stage 4 and the demo car;
nothing uses it by default.

## 3. Throughput

Profile of `VecEnv.step` before (n=32, pure-pursuit driver, mixed stages): `RayField.cast` 54%, `wallPush` 14%,
vehicle physics about 10%, GC 3%. Both hot spots tested every wall segment of the whole track (about 750-1200
segments) for every car every step.

Changes (all in `src/sim/**`):

- `src/sim/wallGrid.ts` (new): one uniform grid (0.5 m cells, CSR layout) per wall set, cached in a `WeakMap` keyed on the
  `track.walls` array so `withSurface` copies and track-pool reuse share it. Build cost 0.2-1.4 ms and 90-140 KiB per
  track. Two per-cell lists sorted by segment id: segments touching the cell (rays walk the cells a ray crosses, with
  a visit stamp to de-duplicate) and segments within 0.72 m of the cell (wall contact tests only the car's cell).
- Exactness, not approximation: wall contact is a sequential push-out in wall order, so the cell list has to be a
  superset and in the same order. If a chain of pushes ever moves the car more than 0.54 m (dense swallow-tail
  geometry at hairpins; found by the equivalence test), the state is restored and the original full loop runs.
  Rays collect all hits in the cells, sort the few by segment id, and apply the original (buggy, see below) rule.
- Allocation: `Vehicle.substep` no longer allocates per substep (scratch arrays for steers/forces, `tireForcesInto`,
  `ackermannInto`, `restHeight`; the old `tireForces`/`ackermann`/`rideRest` still exist and are what the unit
  tests use). `wheelsOff` and the per-step action pair no longer allocate. Remaining per-step allocations: the
  `nearestSample` result objects, `sampleAtS`/`curvaturesAhead`, `scoreStep` result objects (all in `src/track`, or
  per control step, not per substep).
- `VecEnv.baselineActions(ctl?)` takes an optional `BaselineOpts` or a controller function. No argument is the old behavior.

Proof of identical behavior (`tests/sim/vecEnvGolden.test.ts`, fixture `tests/sim/fixtures/vecenv-golden.json`):
the fixture was recorded with the code before any change (commit 59c1d2e) for 5 scenarios (random wild driver on stage 0;
stage 3, high DR, dirt, 6 cars; pure-pursuit on stage 1 low-DR asphalt, stage 2 high-DR carpet, stage 3 no DR;
500-900 control steps each, noise and latency on, many wall hits and resets). Each control step is hashed over every car's
observation, reward, done, truncated and pose, and the final full-precision pose is compared exactly. It still
passes after every commit. `tests/sim/wallGrid.test.ts` additionally compares the ray distances (both rules) and the
wall push-out against copies of the original loops on 5 tracks (stage 0-4), thousands of poses, including poses
buried in walls.

Measured car-steps/s, single process, `scripts/bench-vec.ts` (pure-pursuit driver, mix of stage 0/2/3 tracks).
The machine had 3 other heavy workers running (load average 6-8 on 4 cores), so the absolute numbers are noisy.
Before and after were run interleaved, 3 timed repeats each, repeated 3-4 times:

| n | before (median of runs) | after (median of runs) |
|---|---|---|
| 32 | 3.1k-3.6k (3.3k) | 11.5k-21.6k (about 17k) |
| 64 | 3.4k-3.5k | 14.0k-20.0k |

Best single repeats after: 22k-25k. The SPEC floor (>= 6k per worker) is met with margin even under load; the
`tests/perf/budget.bench.ts` assertion (> 6000) is unchanged. I did not touch any timing threshold.
After the change the profile is: ray casting about 20% (cell walk 15%, fan setup 5%), vehicle substep plus tire/Pacejka about 32%,
`sampleAtS` 5%, GC 4%, the rest spread thin.
Further gains would come from `sampleAtS` (linear scan from index 0, in `src/track/sample.ts`, not mine) and from
collapsing the three `nearestSample` calls per step, which cannot be done without risking a different tie-break.

## 4. Ray bug (not fixed by default)

`RayField.cast` initializes `best = RAY_MAX` (metres) but compares `hit < best` where `hit` is a fraction t in
[0, 1], then stores `best = hit * RAY_MAX`. So a farther crossing visited later can overwrite a nearer one: the ray
sees through the first wall it crosses whenever a later segment in wall order is also hit. `scripts/ray-bug.ts`:
3-6% of rays (stage 0: 5.9%, stage 3: 3.2%) differ from the true nearest hit, by 0.3-0.46 of the 6 m range when wrong,
typically reporting open space where a wall is near.

The default is unchanged because the policy trains on these observations and my brief forbids observation changes.
`SENSOR_FLAGS.nearestRay = true` (in `src/sim/sensors.ts`) switches to the true nearest hit; both rules are covered by
the grid equivalence test. Recommend flipping it at the next retrain.

## Interface additions (nothing existing changed)

- `VecEnv.baselineActions(ctl?: BaselineOpts | BaselineController)`.
- `purePursuit(car, track, hint, opts?, mem?)`, `BaselineOpts`, `BaselineMemory`, `newBaselineMemory`, `BaselineController`.
- `src/sim/wallGrid.ts`: `getWallGrid`, `buildWallGrid`, `resolveWallContact`.
- `src/sim/dr.ts`: `sampleCarCorner`, `driftSetup`.
- `src/sim/sensors.ts`: `SENSOR_FLAGS`.
- `src/sim/car/*`: `tireForcesInto`, `ackermannInto`, `restHeight`.
- `src/eval/physics.ts`: `evaluateBaseline`, `baselineMatrix`, `baselineCorners`, `physics` section.
