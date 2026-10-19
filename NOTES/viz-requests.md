# Requests from viz to the worker and trainer owners

The viz side (`src/main.ts`, `src/viz`, `src/ui`) does not edit `src/workers/**` or `src/rl/**`.
These are the changes that would help the render path. Each one says what the viz side already does
so the change can land in either order.

## 1. `learner.worker.ts` sends the pose tape as a boxed array on every update

```ts
post({ type: 'metrics', stat, tape: tape ? Array.from(tape) : [] });
```

`Array.from(Float32Array)` allocates one JS number per float and the structured clone then copies
each of them again. Today the tape is 16 frames x 7 floats, so it costs little. It scales with the
horizon (3.5 floats per env step) and it becomes a real cost the moment the tape is a whole best lap
(a 40 s lap at 25 Hz is 1000 frames, 7000 numbers, per update).

Asked for:

- Send the tape as a `Float32Array` and put its buffer in the transfer list, or
- better, leave it out of `metrics` and answer `replayReq` only. The viz side already asks on a
  click (`worker.postMessage({ type: 'replayReq', gen })`) and plays the `replayRes` it gets back.
  `onMetrics` accepts a tape if one is present and `replayRes.poses` as either an array or a
  `Float32Array`, so no change on this side is needed. The leaderboard row click is the only consumer.

## 2. `ghosts` is published once per PPO update, not at 30 Hz

`loop()` awaits `trainer.update()` (the whole rollout plus the PPO epochs) and calls `publish()` after
it. With 64 envs that is one `ghosts` message every several seconds, so the pack on screen glides for
about a quarter of a second after each message and then holds still. SPEC section 1 says the ghost
stream is 30 Hz and interpolated; the interpolation is done (`src/viz/interp.ts`), the stream is not.

Asked for: call `publish()` from inside the rollout loop, throttled to about every 33 ms of wall
time (an `onStep` callback on `Trainer.update`, which is `src/rl`). `postMessage` from a synchronous
loop is fine; the main thread receives it while the worker is busy. The viz side already blends over
the observed message interval (clamped to 16 ms .. 250 ms), so it will follow whatever cadence arrives.

## 3. `publish()` copies the pose buffer twice

```ts
const poses = new Float32Array(trainer.env.poses());   // copy 1, needed: poseBuf is reused
post({ type: 'ghosts', poses, ... });                   // copy 2: structured clone
```

Asked for: `scope.postMessage(msg, [poses.buffer])`. At 64 cars it is 1.8 KB, so this is hygiene
rather than a measured cost.

## 4. The tape is not a lap

`trainer.replays.set(gen, ...)` records car 0's pose every second control step for the `T` steps of
one rollout, which is 0.64 s of driving at `horizon = 32`. SPEC section 1 says a leaderboard click
replays "that gen's best lap". The viz side plays whatever tape it is given at `TAPE_HZ = 25`
(`src/viz/replay.ts`), so recording the generation's best lap at one pose per two control steps
is all that is needed. If the recording rate differs, change `TAPE_HZ` in that file.

## 5. PPO update time with 64 envs

The fps eval waits for PPO updates from a 64-env learner. How long that takes is printed by
`npm run eval -- --only=fps` (`updates after N s`). It is a property of the trainer and of the CPU the
other workers are using, not of rendering. It matters because the leaderboard only gains a row per
update, so a slow update makes the "rows == updates" check slow to collect.
