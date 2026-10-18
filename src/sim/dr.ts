import type { CarParams } from '../shared/types';
import type { RNG } from '../shared/rng';
import { cloneCar } from './car/params';

export type DRLevel = 'off' | 'low' | 'high';

function jitter(rng: RNG, base: number, frac: number, level: DRLevel): number {
  if (level === 'off' || frac === 0) return base;
  const f = level === 'low' ? frac * 0.45 : frac;
  return base * (1 + rng.range(-f, f));
}

/** Deterministic per seed. High matches the spec ranges; low is a milder subset. */
export function sampleCar(base: CarParams, rng: RNG, level: DRLevel, gripMul = 1): CarParams {
  const c = cloneCar(base);
  if (level === 'off') {
    c.tire.mu0 *= gripMul;
    return c;
  }
  c.mass = jitter(rng, base.mass, 0.25, level);
  c.cgHeight = jitter(rng, base.cgHeight, 0.3, level);
  c.yawInertia *= c.mass / base.mass;
  c.pitchInertia *= c.mass / base.mass;
  c.rollInertia *= c.mass / base.mass;
  const muLo = level === 'high' ? 0.55 : 0.85;
  const muHi = level === 'high' ? 1.4 : 1.2;
  c.tire.mu0 = rng.range(muLo, muHi) * gripMul;
  c.tire.loadSens = jitter(rng, base.tire.loadSens, 0.15, level);
  for (const axle of [c.tire.front, c.tire.rear]) {
    axle.B = jitter(rng, axle.B, 0.15, level);
    axle.C = jitter(rng, axle.C, 0.15, level);
    axle.E = Math.min(1.1, jitter(rng, axle.E, 0.15, level));
  }
  c.susp.k = jitter(rng, base.susp.k, 0.25, level);
  c.susp.c = jitter(rng, base.susp.c, 0.25, level);
  c.motor.torque = jitter(rng, base.motor.torque, 0.2, level);
  c.steer.rate = jitter(rng, base.steer.rate, 0.2, level);
  const off = level === 'high' ? 2 : 0.8;
  c.steer.offset = (rng.range(-off, off) * Math.PI) / 180;
  return c;
}

export function latencyFor(rng: RNG, level: DRLevel): number {
  if (level === 'off') return 0;
  return level === 'low' ? (rng.next() < 0.5 ? 0 : 1) : rng.int(3);
}

export function noiseFor(rng: RNG, level: DRLevel): number {
  if (level === 'off') return 0;
  return level === 'low' ? rng.range(0.8, 1.2) : rng.range(0.5, 2);
}

export interface Surface {
  name: 'asphalt' | 'carpet' | 'dirt' | 'ice';
  mu: number;
}

export const SURFACES: Record<Surface['name'], number> = {
  asphalt: 1.15,
  carpet: 0.9,
  dirt: 0.7,
  ice: 0.35,
};
