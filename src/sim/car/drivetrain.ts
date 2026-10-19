import { CDA, CRR, G, RHO, WHEEL_RADIUS } from '../../shared/constants';
import type { CarParams } from '../../shared/types';

/** Torque available at the wheel after the linear RPM fade. `torque` is the total motor budget. */
export function motorTorque(omega: number, throttle: number, p: CarParams['motor']): number {
  const rpm = (Math.abs(omega) * 60) / (2 * Math.PI);
  const fade = Math.max(0, 1 - rpm / Math.max(1, p.maxRpm));
  return Math.max(0, throttle) * p.torque * fade;
}

export function lagAlpha(dt: number, tau: number): number {
  return 1 - Math.exp(-dt / Math.max(tau, 1e-4));
}

/**
 * Steady-state top speed on flat ground: total drive force equals aero + rolling.
 * Drive force = T(ω) / r with ω = v / r (no-slip equilibrium).
 */
export function analyticTopSpeed(p: CarParams): number {
  const r = WHEEL_RADIUS;
  const vFree = (p.motor.maxRpm / 60) * Math.PI * 2 * r;
  const drive0 = p.motor.torque / r;
  const a = 0.5 * RHO * CDA;
  const b = drive0 / vFree;
  const roll = CRR * p.mass * G;
  // a v^2 + b v + roll - drive0 = 0
  const c = roll - drive0;
  const disc = b * b - 4 * a * c;
  if (disc <= 0 || a < 1e-8) return Math.max(0, vFree * (1 - roll / Math.max(drive0, 1e-6)));
  return Math.max(0, (-b + Math.sqrt(disc)) / (2 * a));
}

export function aeroDrag(vx: number, speed: number): number {
  return -0.5 * RHO * CDA * vx * speed;
}

export function rollingForce(fz: number, vLong: number): number {
  const s = Math.tanh(vLong / 0.08);
  return -CRR * fz * s;
}
