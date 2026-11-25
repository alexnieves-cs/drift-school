# Contract notes

The lead added fields the SPEC snippet did not list, because the physics and the UI needed them:

- `CarParams.pitchInertia`, `rollInertia`, `tire.rearMuScale`
- `TrackSpec.gates`
- `EpisodeConfig.raysOnly`, `lapsTarget`, `seed`
- `StepInfo` extras: `speed`, `wrongWay`, `stuck`, `rewardRaw`
- `GenStat` extras: `entropy`, `kl`, `valueLoss`, `stepsPerSec`
- `VecEnv.snapshot`, `trackIds` (not on the interface; concrete class only)
- From-scratch-rl branch: `EpisodeConfig.drift` (`DriftConfig`: `handbrake`, `slipWeight`, `slipTargetDeg`), a fourth `handbrake` argument to `Vehicle.step` (default 0), `VecEnvOpts.slotOffset` and `actDim` (2 or 3), and `RewardInput.slip`, `slipWeight`, `slipTarget`. All are optional and leave existing behaviour unchanged when left out.

No other package owns `src/shared/`.
