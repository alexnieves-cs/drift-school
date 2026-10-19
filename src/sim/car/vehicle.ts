import type { CarParams, CarState } from '../../shared/types';
import { G, WHEEL_RADIUS } from '../../shared/constants';
import { applyArb, restHeight, springForce } from './suspension';
import { aeroDrag, lagAlpha, motorTorque, rollingForce } from './drivetrain';
import { ackermannInto, clamp, servoStep } from './steering';
import { tireForcesInto, type TireOut } from './tire';
import type { WallWorld } from '../world';

export interface PoseInit {
  x?: number;
  y?: number;
  z?: number;
  yaw?: number;
  speed?: number;
}

const WHEEL_I = 8e-4;

export class Vehicle {
  readonly p: CarParams;
  px = 0;
  py = 0;
  pz = 0;
  yaw = 0;
  pitch = 0;
  roll = 0;
  /** Body frame: +x forward, +y up, +z right. */
  vx = 0;
  vy = 0;
  vz = 0;
  wYaw = 0;
  wPitch = 0;
  wRoll = 0;
  steer = 0;
  driveState = 0;
  readonly omega = new Float32Array(4);
  readonly load = new Float32Array(4);
  readonly slipA = new Float32Array(4);
  readonly slipK = new Float32Array(4);
  readonly comp = new Float32Array(4);
  s = 0;
  lap = 0;
  lapTime = 0;
  /** Per-segment surface grip, multiplied onto tire μ. 1 keeps CarParams.mu0. */
  surfaceMu = 1;
  walls: WallWorld | null = null;
  private readonly state: CarState;
  private readonly xs: number[];
  private readonly zs: number[];
  // per-substep scratch, kept off the heap
  private readonly ack = new Float64Array(2);
  private readonly steers = new Float64Array(4);
  private readonly forces = new Float64Array(4);
  private readonly tire: TireOut = { Fx: 0, Fy: 0, slipAngle: 0, slipRatio: 0 };

  constructor(p: CarParams) {
    this.p = p;
    const wb = p.wheelbase / 2;
    const tr = p.track / 2;
    this.xs = [wb, wb, -wb, -wb];
    this.zs = [-tr, tr, -tr, tr];
    this.state = {
      pos: [0, 0, 0],
      quat: [1, 0, 0, 0],
      vel: [0, 0, 0],
      yawRate: 0,
      steer: 0,
      wheelOmega: new Float32Array(4),
      wheelLoad: new Float32Array(4),
      slipAngle: new Float32Array(4),
      slipRatio: new Float32Array(4),
      s: 0,
      lap: 0,
      lapTime: 0,
    };
    this.reset();
  }

  reset(pose: PoseInit = {}): CarState {
    this.px = pose.x ?? 0;
    this.py = pose.y ?? this.p.cgHeight;
    this.pz = pose.z ?? 0;
    this.yaw = pose.yaw ?? 0;
    this.pitch = 0;
    this.roll = 0;
    this.vx = pose.speed ?? 0;
    this.vy = 0;
    this.vz = 0;
    this.wYaw = 0;
    this.wPitch = 0;
    this.wRoll = 0;
    this.steer = this.p.steer.offset;
    this.driveState = 0;
    this.omega.fill(this.vx / WHEEL_RADIUS);
    this.load.fill((this.p.mass * G) / 4);
    this.slipA.fill(0);
    this.slipK.fill(0);
    this.comp.fill(0);
    this.s = 0;
    this.lap = 0;
    this.lapTime = 0;
    return this.writeState();
  }

  /** World-space planar speed. */
  speed(): number {
    return Math.hypot(this.vx, this.vz);
  }

  frontLoadShare(): number {
    const f = this.load[0]! + this.load[1]!;
    const r = this.load[2]! + this.load[3]!;
    const t = f + r;
    return t > 1e-4 ? f / t : 0.5;
  }

  meanSlip(axle: 'front' | 'rear'): number {
    return axle === 'front' ? (this.slipA[0]! + this.slipA[1]!) / 2 : (this.slipA[2]! + this.slipA[3]!) / 2;
  }

  step(steerCmd: number, throttleCmd: number, dt: number): CarState {
    const sub = 1 / 500;
    const n = Math.max(1, Math.round(dt / sub));
    const h = dt / n;
    const steer = clamp(steerCmd, -1, 1);
    const throttle = clamp(throttleCmd, -1, 1);
    for (let i = 0; i < n; i++) this.substep(steer, throttle, h);
    return this.writeState();
  }

  private substep(steerCmd: number, throttle: number, dt: number): void {
    const p = this.p;
    const target = steerCmd * p.steer.maxAngle + p.steer.offset;
    this.steer = servoStep(this.steer, target, p.steer.rate, dt);
    ackermannInto(this.ack, this.steer, p.wheelbase, p.track);
    const steers = this.steers;
    steers[0] = this.ack[0]!;
    steers[1] = this.ack[1]!;
    steers[2] = 0;
    steers[3] = 0;

    const driveCmd = throttle > 0 ? throttle : 0;
    const brakeCmd = throttle < 0 ? -throttle : 0;
    this.driveState += (driveCmd - this.driveState) * lagAlpha(dt, p.motor.tau);

    const rest = restHeight(p.mass, p.susp.k, p.cgHeight, G);
    const forces = this.forces;
    for (let i = 0; i < 4; i++) {
      const yMount = this.py - this.pitch * this.xs[i]! + this.roll * this.zs[i]!;
      const cRaw = rest - yMount;
      const yDot = this.vy - this.wPitch * this.xs[i]! + this.wRoll * this.zs[i]!;
      const cDot = -yDot;
      this.comp[i] = Math.min(p.susp.travel, Math.max(0, cRaw));
      forces[i] = springForce(cRaw, cDot, p.susp);
    }
    applyArb(forces, this.comp, p.susp.arb);

    let sumFx = 0;
    let sumFz = 0;
    let yawMom = 0;
    let pitchTire = 0;
    let rollTire = 0;
    const reverse = brakeCmd > 0.15 && Math.abs(this.vx) < 0.3;

    for (let i = 0; i < 4; i++) {
      const vxi = this.vx - this.wYaw * this.zs[i]!;
      const vzi = this.vz + this.wYaw * this.xs[i]!;
      const delta = steers[i]!;
      const cδ = Math.cos(delta);
      const sδ = Math.sin(delta);
      const vLong = vxi * cδ + vzi * sδ;
      const vLat = -vxi * sδ + vzi * cδ;
      const front = i < 2;
      const driven = p.drive === '4wd' || !front;
      const share = !driven ? 0 : p.drive === 'rwd' ? 0.5 : front ? p.split * 0.5 : (1 - p.split) * 0.5;
      const muScale = front ? 1 : p.tire.rearMuScale;
      const pace = front ? p.tire.front : p.tire.rear;
      const fz = forces[i]!;

      let T = 0;
      if (share > 0) {
        if (reverse) T = -brakeCmd * p.motor.torque * share;
        else T = motorTorque(this.omega[i]!, this.driveState, p.motor) * share;
      }
      const tire = this.tire;
      tireForcesInto(
        tire,
        vLong,
        vLat,
        this.omega[i]!,
        fz,
        p.tire.mu0 * muScale * this.surfaceMu,
        p.tire.loadSens,
        pace.B,
        pace.C,
        pace.E,
      );
      const Tbrake = reverse ? 0 : brakeCmd * p.motor.torque * 2.2 * Math.sign(this.omega[i]! || vLong || 1);
      const dw = (T - Tbrake - tire.Fx * WHEEL_RADIUS) / WHEEL_I;
      const capped = clamp(dw * dt, -40, 40);
      this.omega[i] = clamp(this.omega[i]! + capped, -900, 900);

      const rr = rollingForce(fz, vLong);
      const FxT = tire.Fx + rr * cδ;
      const FyT = tire.Fy;
      const FxB = FxT * cδ - FyT * sδ;
      const FzB = FxT * sδ + FyT * cδ;
      sumFx += FxB;
      sumFz += FzB;
      yawMom += this.xs[i]! * FzB - this.zs[i]! * FxB;
      pitchTire += p.cgHeight * FxB;
      rollTire += -p.cgHeight * FzB;
      this.slipA[i] = tire.slipAngle;
      this.slipK[i] = tire.slipRatio;
      this.load[i] = fz;
    }

    const spd = Math.hypot(this.vx, this.vz);
    sumFx += aeroDrag(this.vx, spd);

    this.vx += (sumFx / p.mass + this.vz * this.wYaw) * dt;
    this.vz += (sumFz / p.mass - this.vx * this.wYaw) * dt;
    this.vy += ((0 + forces[0]! + forces[1]! + forces[2]! + forces[3]!) / p.mass - G) * dt;

    let pitchSusp = 0;
    let rollSusp = 0;
    for (let i = 0; i < 4; i++) {
      pitchSusp += -this.xs[i]! * forces[i]!;
      rollSusp += this.zs[i]! * forces[i]!;
    }
    this.wYaw += (yawMom / p.yawInertia) * dt;
    this.wPitch += ((pitchSusp - pitchTire) / p.pitchInertia) * dt;
    this.wRoll += ((rollSusp + rollTire) / p.rollInertia) * dt;
    this.wPitch *= 1 - Math.min(0.15, 2.5 * dt);
    this.wRoll *= 1 - Math.min(0.15, 2.5 * dt);

    this.yaw += this.wYaw * dt;
    this.pitch = clamp(this.pitch + this.wPitch * dt, -0.55, 0.55);
    this.roll = clamp(this.roll + this.wRoll * dt, -0.9, 0.9);
    this.py += this.vy * dt;
    if (this.py < 0.015) {
      this.py = 0.015;
      if (this.vy < 0) this.vy = 0;
    }
    if (this.py > 0.25) {
      this.py = 0.25;
      if (this.vy > 0) this.vy = 0;
    }

    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    let worldVx = cy * this.vx - sy * this.vz;
    let worldVz = sy * this.vx + cy * this.vz;
    this.px += worldVx * dt;
    this.pz += worldVz * dt;

    if (this.walls) {
      const resolved = this.walls.resolve(this.px, this.py, this.pz, this.yaw, worldVx, this.vy, worldVz, dt);
      this.px = resolved.px;
      this.py = resolved.py;
      this.pz = resolved.pz;
      worldVx = resolved.vx;
      this.vy = resolved.vy;
      worldVz = resolved.vz;
      this.vx = cy * worldVx + sy * worldVz;
      this.vz = -sy * worldVx + cy * worldVz;
      if (resolved.hit.hit && resolved.hit.impactSpeed > 0.4) this.wYaw *= 0.98;
    }

    this.vx = clamp(finite(this.vx), -45, 45);
    this.vy = clamp(finite(this.vy), -20, 20);
    this.vz = clamp(finite(this.vz), -45, 45);
    this.wYaw = clamp(finite(this.wYaw), -25, 25);
    this.wPitch = clamp(finite(this.wPitch), -12, 12);
    this.wRoll = clamp(finite(this.wRoll), -12, 12);
  }

  private writeState(): CarState {
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const s = this.state;
    s.pos[0] = this.px;
    s.pos[1] = this.py;
    s.pos[2] = this.pz;
    const q = ypr(this.yaw, this.pitch, this.roll);
    s.quat[0] = q[0];
    s.quat[1] = q[1];
    s.quat[2] = q[2];
    s.quat[3] = q[3];
    s.vel[0] = cy * this.vx - sy * this.vz;
    s.vel[1] = this.vy;
    s.vel[2] = sy * this.vx + cy * this.vz;
    s.yawRate = this.wYaw;
    s.steer = this.steer;
    s.wheelOmega.set(this.omega);
    s.wheelLoad.set(this.load);
    s.slipAngle.set(this.slipA);
    s.slipRatio.set(this.slipK);
    s.s = this.s;
    s.lap = this.lap;
    s.lapTime = this.lapTime;
    return s;
  }
}

function finite(n: number): number {
  return Number.isFinite(n) ? n : 0;
}

function ypr(yaw: number, pitch: number, roll: number): [number, number, number, number] {
  const cy = Math.cos(yaw / 2);
  const sy = Math.sin(yaw / 2);
  const cp = Math.cos(pitch / 2);
  const sp = Math.sin(pitch / 2);
  const cr = Math.cos(roll / 2);
  const sr = Math.sin(roll / 2);
  return [
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ];
}

export function stateFinite(s: CarState): boolean {
  const nums = [...s.pos, ...s.quat, ...s.vel, s.yawRate, s.steer, s.s, s.lap, s.lapTime];
  for (const n of nums) if (!Number.isFinite(n)) return false;
  for (const a of [s.wheelOmega, s.wheelLoad, s.slipAngle, s.slipRatio]) {
    for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i]!)) return false;
  }
  return true;
}
