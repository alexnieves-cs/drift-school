# Integration notes

Streams were built in one branch (`cursor/drift-school-a9e2`) rather than four worktrees. The cloud agent opens a single pull request, so separate `feat/*` branches would have fought that constraint.

## What is real

- Deterministic vehicle, Pacejka-lite, suspension weight transfer, Rapier `castShape` for the keyboard car.
- Seeded tracks, validator, pure-pursuit baseline that finishes stage 0–2.
- PPO (toy env) plus a pure-pursuit imitation warm start and a critic-only warm-up. That warm start is what laps stage 0. From-scratch PPO at roughly 500 steps/s does not.
- Per-episode curriculum, track mix and domain randomization are wired through `VecEnv.sampler`; hot-swap puts every env on the drawn track at once and the 70/30 mix applies afterwards.
- Vectorized env wall response is a polyline impulse. Rapier is not stepped per ghost.
- `tRef` is an analytic speed-profile integral, not a simulated lap. `simulateBaseline` is the simulated check.
- Learner worker owns training. A separate env worker implements init/step/`setTrackPool`. There was no 4-worker SharedArrayBuffer farm at v1.0.0; on the from-scratch-rl branch it exists as a tested module that the page does not use yet (NOTES/perf.md), and the 20k steps/s budget is still not claimed.
- Renderer: instanced ghosts, AgX, shadows on high, WebGPU attempted when `navigator.gpu` exists, WebGL2 fallback. No GTAO, TAA, CSM, or bloom.
- Tire marks are an instanced box ribbon while rear slip is high, not a decal pass.

## Not done

- Elevation, self-play, pixel policy, ONNX, rays-only side-by-side UI.
- Handbrake (from-scratch-rl branch): drift mode exists in the vehicle, `VecEnv` (actDim 3) and the reward, off by default, but the policy, trainer and page still use 2-D actions, so nothing is trained with it.
- Training through stages 1-4: no run here got past stage 0, so the matrix and M4 numbers are for a stage-0 policy. The drawn S-curve is already driven zero-shot, so the two-minute fine-tune check shows no regression, not adaptation.
- M3 frame rate and long-task bars on a GPU; M5 camera-switch and HUD assertions.
- Drift is a sustained slide, not a tuned drift.
- Hot-swap does not wait for running episodes to finish; the replay tape is the first car only.

See the README for the measured acceptance table and the full known-gaps list.
