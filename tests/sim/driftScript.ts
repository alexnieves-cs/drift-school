import type { CarParams } from '../../src/shared/types';
import { Vehicle } from '../../src/sim/car/vehicle';

const DEG = 180 / Math.PI;
const DT = 0.02;

/** Point the front wheels along the velocity vector and damp yaw rate. */
export function countersteer(car: Vehicle, kb = 1.2, kr = 0.12): number {
  const beta = Math.atan2(car.vz, Math.max(0.3, Math.abs(car.vx)));
  return Math.max(-1, Math.min(1, (kb * beta) / car.p.steer.maxAngle - kr * car.wYaw));
}

export interface DriftRun {
  /** Peak mean rear slip angle during the corner, degrees. */
  peakDeg: number;
  /** Seconds spent above 15 deg while still moving at >= 1.5 m/s. */
  slidingSeconds: number;
  /** Lowest speed seen while above 15 deg. */
  minSpeedWhileSliding: number;
  /** Rear slip held above 15 deg for >= 0.5 s without the speed dropping below 1.5 m/s or the slip passing 75 deg. */
  reached: boolean;
  /** After the corner, countersteer settles slip < 5 deg and yaw rate < 1 rad/s at > 1 m/s for 0.5 s. */
  recovered: boolean;
  recoverSeconds: number;
  finalSpeed: number;
  note: string;
}

/**
 * The scripted drift maneuver of SPEC section 3: cruise at ~4.5 m/s, turn in with throttle
 * pinned for `cornerSeconds`, then countersteer with the throttle mostly lifted.
 */
export function runDrift(p: CarParams, opts: { entrySteer?: number; cornerSeconds?: number; v0?: number } = {}): DriftRun {
  const car = new Vehicle(p);
  const v0 = opts.v0 ?? 4.5;
  car.reset({ speed: v0 });
  for (let i = 0; i < 100; i++) car.step(0, Math.max(0, Math.min(1, 0.4 + (v0 - car.speed()) * 0.3)), DT);
  let peak = 0;
  let sliding = 0;
  let minV = Infinity;
  const corner = Math.round((opts.cornerSeconds ?? 2) / DT);
  for (let i = 0; i < corner; i++) {
    const rear = car.meanSlip('rear') * DEG;
    // Turn in, then lean on the wheel until the rear lets go; once it has, hold the slide with
    // countersteer and throttle modulation so it neither grips back up nor spins out.
    const entry = Math.min(1, (opts.entrySteer ?? 0.6) + 0.6 * (i / corner));
    const deep = Math.abs(rear) > 32;
    const steer = deep ? Math.max(-1, Math.min(1, countersteer(car, 1.0, 0.0) + Math.sign(car.wYaw) * 0.25)) : entry;
    const throttle = Math.abs(rear) > 45 ? 0.3 : 1;
    car.step(steer, throttle, DT);
    const s = Math.abs(car.meanSlip('rear')) * DEG;
    peak = Math.max(peak, s);
    if (s > 15) {
      minV = Math.min(minV, car.speed());
      if (car.speed() >= 1.5) sliding += DT;
    }
  }
  let calm = 0;
  let recoverSeconds = -1;
  for (let i = 0; i < 300; i++) {
    car.step(countersteer(car), 0.15, DT);
    const ok = Math.abs(car.meanSlip('rear')) * DEG < 5 && Math.abs(car.wYaw) < 1 && car.speed() > 1;
    calm = ok ? calm + 1 : 0;
    if (calm >= 25) {
      recoverSeconds = (i + 1) * DT;
      break;
    }
  }
  // A drift keeps its speed while sliding. Spinning to a stop (or peaking past 75 deg) is not one.
  const reached = peak > 15 && peak < 75 && sliding >= 0.5 && minV >= 1.5;
  const recovered = recoverSeconds >= 0;
  const note = !reached ? (peak > 15 ? 'spun-out' : 'grip') : recovered ? 'ok' : 'no-recovery';
  return { peakDeg: peak, slidingSeconds: sliding, minSpeedWhileSliding: minV, reached, recovered, recoverSeconds, finalSpeed: car.speed(), note };
}
