# Integration notes

Streams were built in one branch (`cursor/drift-school-a9e2`) rather than four worktrees. The cloud agent opens a single pull request, so separate `feat/*` branches would have fought that constraint.

## What is real

- Deterministic vehicle, Pacejka-lite, suspension weight transfer, Rapier `castShape` for the keyboard car.
- Seeded tracks, validator, pure-pursuit baseline that finishes stage 0–2.
- PPO (toy env) plus a pure-pursuit imitation warm start. That warm start is what laps stage 0. From-scratch PPO at roughly 800 steps/s does not.
- Vectorized env wall response is a polyline impulse. Rapier is not stepped per ghost.
- `tRef` is an analytic speed-profile integral, not a simulated lap. `simulateBaseline` is the simulated check.
- Learner worker owns training. A separate env worker implements init/step/`setTrackPool`. There is no 4-worker SharedArrayBuffer farm, so the 20k steps/s budget is not claimed.
- Renderer: instanced ghosts, AgX, shadows on high, WebGPU attempted when `navigator.gpu` exists, WebGL2 fallback. No GTAO, TAA, CSM, or bloom.
- Tire marks are an instanced box ribbon while rear slip is high, not a decal pass.

## Not done

- Handbrake, elevation, self-play, pixel policy, ONNX, rays-only side-by-side UI.
- M4's two-minute fine-tune "≥ 80% and no forgetting" was not run as an end-to-end clock. The swap keeps the weight hash and sets the 70/30 mix.
- M5's full surface × DR completion matrix at ≥ 70% was not demonstrated. Segment μ is applied. Completion across dirt and high DR is unproven.
