/** Per-wheel spring-damper + anti-roll. Compression is along world up. */

export interface SuspParams {
  k: number;
  c: number;
  travel: number;
  arb: number;
}

export interface SuspOut {
  force: number;
  compression: number;
}

export function springForce(compression: number, compressionRate: number, p: SuspParams): number {
  const c = Math.min(p.travel, Math.max(0, compression));
  const rate = compression > 0 && compression < p.travel ? compressionRate : 0;
  return Math.max(0, p.k * c + p.c * rate);
}

export function antiRoll(compL: number, compR: number, arb: number): number {
  return arb * (compL - compR);
}

export function rideRest(mass: number, k: number, cgHeight: number, g = 9.81): { delta: number; rest: number } {
  const delta = (mass * g) / (4 * k);
  return { delta, rest: cgHeight + delta };
}

export function applyArb(forces: number[], comp: ArrayLike<number>, arb: number): void {
  const front = antiRoll(comp[0] ?? 0, comp[1] ?? 0, arb);
  const rear = antiRoll(comp[2] ?? 0, comp[3] ?? 0, arb);
  forces[0] = Math.max(0, (forces[0] ?? 0) + front);
  forces[1] = Math.max(0, (forces[1] ?? 0) - front);
  forces[2] = Math.max(0, (forces[2] ?? 0) + rear);
  forces[3] = Math.max(0, (forces[3] ?? 0) - rear);
}
