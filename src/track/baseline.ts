import type { TrackSpec } from '../shared/types';
import { CONTROL_DT } from '../shared/constants';
import { nearestSample, sampleAtS } from './sample';
import { Vehicle } from '../sim/car/vehicle';
import { defaultCarParams } from '../sim/car/params';

export interface BaselineCommand {
  steer: number;
  throttle: number;
  hint: number;
  s: number;
}

export function purePursuit(car: Vehicle, track: TrackSpec, hint: number): BaselineCommand {
  const q = nearestSample(track, car.px, car.pz, hint);
  const look = 1.15 + Math.min(1.2, car.speed() * 0.15);
  const target = sampleAtS(track, q.sample.s + look);
  const dx = target.x - car.px;
  const dz = target.z - car.pz;
  const cy = Math.cos(car.yaw);
  const sy = Math.sin(car.yaw);
  const bx = cy * dx + sy * dz;
  const bz = -sy * dx + cy * dz;
  const steer = Math.max(-1, Math.min(1, Math.atan2(bz, Math.max(0.2, bx)) / car.p.steer.maxAngle));
  const k = Math.abs(sampleAtS(track, q.sample.s + 0.6).curvature);
  const mu = q.sample.mu || 1;
  const vLim = Math.sqrt((0.55 * mu * 9.81) / Math.max(k, 0.08));
  const vTarget = Math.min(5.8, Math.max(1.6, vLim));
  const throttle = Math.max(-1, Math.min(1, (vTarget - car.speed()) * 0.9));
  return { steer, throttle, hint: q.index, s: q.sample.s };
}

/** Run the scripted baseline. Returns whether it completed a lap and the lap time. */
export function simulateBaseline(track: TrackSpec, maxSeconds = 90): { finished: boolean; lapTime: number } {
  const car = new Vehicle(defaultCarParams());
  const s0 = track.samples[track.startIndex] ?? track.samples[0];
  if (!s0) return { finished: false, lapTime: 0 };
  car.reset({ x: s0.x, z: s0.z, yaw: Math.atan2(s0.tz, s0.tx), speed: 1.2 });
  let hint = track.startIndex;
  let prevS = s0.s;
  let time = 0;
  const steps = Math.round(maxSeconds / CONTROL_DT);
  for (let i = 0; i < steps; i++) {
    const cmd = purePursuit(car, track, hint);
    hint = cmd.hint;
    car.step(cmd.steer, cmd.throttle, CONTROL_DT);
    time += CONTROL_DT;
    const q = nearestSample(track, car.px, car.pz, hint);
    hint = q.index;
    if (q.sample.s < prevS - track.length * 0.5 && prevS > track.length * 0.4) {
      return { finished: true, lapTime: time };
    }
    prevS = q.sample.s;
    if (car.speed() < 0.15 && time > 8) break;
  }
  return { finished: false, lapTime: time };
}
