import { WHEEL_RADIUS } from '../../shared/constants';

export interface TireInput {
  vLong: number;
  vLat: number;
  omega: number;
  fz: number;
  mu0: number;
  loadSens: number;
  B: number;
  C: number;
  E: number;
}

export interface TireOut {
  Fx: number;
  Fy: number;
  slipAngle: number;
  slipRatio: number;
}

const FZ0 = 4;

/** Pacejka magic formula, odd in x, peak D. */
export function pacejka(x: number, B: number, C: number, E: number, D: number): number {
  const Bx = B * x;
  return D * Math.sin(C * Math.atan(Bx - E * (Bx - Math.atan(Bx))));
}

export function tireForces(i: TireInput): TireOut {
  const eps = 0.35;
  const slipAngle = Math.atan2(i.vLat, Math.abs(i.vLong));
  const slipRatio = (i.omega * WHEEL_RADIUS - i.vLong) / Math.max(Math.abs(i.vLong), eps);
  const mu = i.mu0 * Math.max(0.25, 1 - i.loadSens * (i.fz / FZ0));
  const D = Math.max(0, mu * i.fz);
  const Fx0 = pacejka(slipRatio, i.B, i.C, i.E, D);
  const Fy0 = -pacejka(slipAngle, i.B, i.C, i.E, D);
  let Fx = Fx0;
  let Fy = Fy0;
  const denom = Math.max(D, 1e-6);
  const ell = Math.hypot(Fx / denom, Fy / denom);
  if (ell > 1) {
    Fx /= ell;
    Fy /= ell;
  }
  const speed = Math.hypot(i.vLong, i.vLat);
  const blend = speed >= 0.5 ? 1 : speed / 0.5;
  // Longitudinal force keeps the Pacejka value (drive/brake still works at rest).
  // Lateral force blends into a linear damper below 0.5 m/s to avoid the slip-angle singularity.
  const damp = 1.1 * Math.max(i.fz, 0);
  Fx += (1 - blend) * (-damp * i.vLong) * 0.15;
  Fy = blend * Fy + (1 - blend) * (-damp * i.vLat);
  return { Fx, Fy, slipAngle, slipRatio };
}
