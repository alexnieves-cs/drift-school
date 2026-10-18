# Observation layout

`OBS_DIM = 48`. Values are written by `fillObservation` in `src/sim/sensors.ts`.

| Index | Name | Notes |
|---:|---|---|
| 0–20 | `ray0`…`ray20` | 21 distances in a forward 200° fan, divided by 6 m. Optional 1 cm noise and 1% dropout. |
| 21 | `v_long` | Body-frame longitudinal speed / 10 |
| 22 | `v_lat` | Body-frame lateral speed / 10 |
| 23 | `yaw_rate` | rad/s / 6 |
| 24 | `slip_front` | Mean front slip angle (rad) |
| 25 | `slip_rear` | Mean rear slip angle (rad) |
| 26 | `slip_ratio_rear` | Mean rear longitudinal slip ratio |
| 27 | `prev_steer` | Previous steer command |
| 28 | `prev_throttle` | Previous throttle command |
| 29 | `steer_actual` | Servo angle / max angle |
| 30 | `load_front` | Front axle share of normal load |
| 31 | `load_rear` | Rear axle share |
| 32 | `lat_offset` | Centerline offset / half-width (zeroed in rays-only mode) |
| 33 | `heading_sin` | sin(heading error) |
| 34 | `heading_cos` | cos(heading error) |
| 35–39 | `curv0`…`curv4` | Curvature at 0.5, 1, 2, 3, 4 m ahead, ×4 |
| 40 | `time_left` | 1 → 0 over the episode step cap |
| 41 | `roll` | rad |
| 42 | `pitch` | rad (positive nose-down) |
| 43 | `slip_ratio_front` | Mean front slip ratio |
| 44 | `speed` | Planar speed / 10 |
| 45 | `load_fl` | FL load / total |
| 46 | `load_fr` | FR load / total |
| 47 | `progress` | `s / length` |

Action is 2-D: `[steer, throttle]` in `[-1, 1]`. Negative throttle brakes, then reverses once the car is nearly stopped.
