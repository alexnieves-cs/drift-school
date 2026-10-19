# Contract notes

The lead added fields the SPEC snippet did not list, because the physics and the UI needed them:

- `CarParams.pitchInertia`, `rollInertia`, `tire.rearMuScale`
- `TrackSpec.gates`
- `EpisodeConfig.raysOnly`, `lapsTarget`, `seed`
- `StepInfo` extras: `speed`, `wrongWay`, `stuck`, `rewardRaw`
- `GenStat` extras: `entropy`, `kl`, `valueLoss`, `stepsPerSec`
- `VecEnv.snapshot`, `trackIds` (not on the interface; concrete class only)

No other package owns `src/shared/`.
