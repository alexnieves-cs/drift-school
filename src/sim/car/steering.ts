export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Rate-limit a servo toward target. Returns the new angle. */
export function servoStep(current: number, target: number, rate: number, dt: number): number {
  const max = rate * dt;
  return current + clamp(target - current, -max, max);
}

/** Ackermann angles (left, right), radians. Positive steers to +z (right). */
export function ackermann(delta: number, wheelbase: number, track: number): [number, number] {
  if (Math.abs(delta) < 1e-4) return [delta, delta];
  const tan = Math.tan(delta);
  const R = wheelbase / tan;
  const innerLimit = track * 0.6;
  const Rs = Math.abs(R) < innerLimit ? Math.sign(R || 1) * innerLimit : R;
  const left = Math.atan(wheelbase / (Rs + track / 2));
  const right = Math.atan(wheelbase / (Rs - track / 2));
  return [left, right];
}

/** Allocation-free `ackermann`: writes (left, right) into out[0], out[1]. */
export function ackermannInto(out: ArrayLike<number> & { [i: number]: number }, delta: number, wheelbase: number, track: number): void {
  if (Math.abs(delta) < 1e-4) {
    out[0] = delta;
    out[1] = delta;
    return;
  }
  const tan = Math.tan(delta);
  const R = wheelbase / tan;
  const innerLimit = track * 0.6;
  const Rs = Math.abs(R) < innerLimit ? Math.sign(R || 1) * innerLimit : R;
  out[0] = Math.atan(wheelbase / (Rs + track / 2));
  out[1] = Math.atan(wheelbase / (Rs - track / 2));
}
