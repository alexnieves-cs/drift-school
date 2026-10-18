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

export interface BaselineOpts {
  /**
   * Scale the speed target by the car's real tire grip relative to the default tune
   * (`car.p.tire.mu0 / 1.15`), not just the surface μ stored on the track. This is a privileged
   * oracle (a learner cannot read μ0), so it is off for the imitation teacher and on for the
   * upper-bound reference.
   */
  gripAware?: boolean;
  /** Back out of a wall when pinned (needs a BaselineMemory). Also off for the teacher. */
  recover?: boolean;
}

/** Per-car controller state for `recover`. */
export interface BaselineMemory {
  pinned: number;
  reversing: number;
}

export const newBaselineMemory = (): BaselineMemory => ({ pinned: 0, reversing: 0 });

export type BaselineController = (car: Vehicle, track: TrackSpec, hint: number) => BaselineCommand;

const PINNED_TICKS = 20;
const REVERSE_TICKS = 35;

export function purePursuit(car: Vehicle, track: TrackSpec, hint: number, opts?: BaselineOpts, mem?: BaselineMemory): BaselineCommand {
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
  let mu = q.sample.mu || 1;
  if (opts?.gripAware) mu *= (car.p.tire.mu0 / 1.15) * Math.min(1, car.p.tire.rearMuScale);
  const vLim = Math.sqrt((0.55 * mu * 9.81) / Math.max(k, 0.08));
  const vTarget = Math.min(5.8, Math.max(1.6, vLim));
  const throttle = Math.max(-1, Math.min(1, (vTarget - car.speed()) * 0.9));
  if (opts?.recover && mem) {
    if (mem.reversing > 0) {
      mem.reversing--;
      // reverse with the wheels turned the other way so the nose swings toward the track direction
      return { steer: -steer, throttle: -0.8, hint: q.index, s: q.sample.s };
    }
    mem.pinned = car.speed() < 0.25 && throttle > 0.2 ? mem.pinned + 1 : 0;
    if (mem.pinned > PINNED_TICKS) {
      mem.pinned = 0;
      mem.reversing = REVERSE_TICKS;
    }
  }
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
