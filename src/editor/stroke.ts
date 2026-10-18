export interface V2 {
  x: number;
  z: number;
}

function dist(a: V2, b: V2): number {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

/** Ramer–Douglas–Peucker. */
export function rdp(points: V2[], epsilon: number): V2[] {
  if (points.length < 3) return points.slice();
  let maxD = 0;
  let idx = 0;
  const a = points[0]!;
  const b = points[points.length - 1]!;
  const abx = b.x - a.x;
  const abz = b.z - a.z;
  const ab = abx * abx + abz * abz || 1;
  for (let i = 1; i < points.length - 1; i++) {
    const p = points[i]!;
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.z - a.z) * abz) / ab));
    const d = Math.hypot(p.x - (a.x + abx * t), p.z - (a.z + abz * t));
    if (d > maxD) {
      maxD = d;
      idx = i;
    }
  }
  if (maxD > epsilon) {
    const left = rdp(points.slice(0, idx + 1), epsilon);
    const right = rdp(points.slice(idx), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [a, b];
}

/** Chaikin corner cutting. One pass. */
export function chaikin(points: V2[], closed = false): V2[] {
  const out: V2[] = [];
  const n = points.length;
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    const a = points[i]!;
    const b = points[(i + 1) % n]!;
    out.push({ x: a.x * 0.75 + b.x * 0.25, z: a.z * 0.75 + b.z * 0.25 });
    out.push({ x: a.x * 0.25 + b.x * 0.75, z: a.z * 0.25 + b.z * 0.75 });
  }
  if (!closed) {
    out.unshift(points[0]!);
    out.push(points[n - 1]!);
  }
  return out;
}

export function smoothStroke(points: V2[], closed: boolean): V2[] {
  const simple = rdp(points, 0.25);
  let cur = simple;
  for (let i = 0; i < 2; i++) cur = chaikin(cur, closed);
  return cur;
}

/** Resample a polyline (open or closed) to a uniform arc-length spacing. */
export function resampleUniform(points: V2[], spacing: number, closed: boolean): V2[] {
  const n = points.length;
  if (n < 3) return points.slice();
  const pts = closed ? [...points, points[0]!] : points;
  const cum: number[] = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1]! + dist(pts[i - 1]!, pts[i]!));
  const total = cum[cum.length - 1]!;
  const count = Math.max(closed ? 8 : 2, Math.round(total / spacing));
  const out: V2[] = [];
  let seg = 0;
  const last = closed ? count : count + 1;
  for (let k = 0; k < last; k++) {
    const target = (total * k) / count;
    while (seg < pts.length - 2 && cum[seg + 1]! < target) seg++;
    const a = pts[seg]!;
    const b = pts[seg + 1]!;
    const len = cum[seg + 1]! - cum[seg]! || 1;
    const t = Math.max(0, Math.min(1, (target - cum[seg]!) / len));
    out.push({ x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t });
  }
  return out;
}

function turnCurvature(a: V2, b: V2, c: V2): number {
  const abx = b.x - a.x;
  const abz = b.z - a.z;
  const bcx = c.x - b.x;
  const bcz = c.z - b.z;
  const cross = abx * bcz - abz * bcx;
  const la = Math.hypot(abx, abz);
  const lb = Math.hypot(bcx, bcz);
  const lc = Math.hypot(c.x - a.x, c.z - a.z);
  const denom = la * lb * lc;
  return denom < 1e-9 ? 0 : (2 * Math.abs(cross)) / denom;
}

/**
 * Curvature limiter (SPEC editor section): resample a closed stroke uniformly, low-pass it, then
 * relax every point whose turn radius is below `minRadius` toward its neighbors until none
 * is, so Chaikin corners on an uneven RDP polygon cannot pinch the track edges.
 */
export function relaxCurvature(points: V2[], minRadius: number, spacing = 0.4, maxIter = 120): V2[] {
  let cur = resampleUniform(points, spacing, true);
  const n = cur.length;
  if (n < 8) return cur;
  // Low-pass first. A mouse reports whole pixels (4 cm here), and that tremor, sampled by a
  // 0.1 m spline, reads as curvature spikes the 0.4 m polyline test below cannot see.
  for (let pass = 0; pass < 4; pass++) {
    cur = cur.map((b, i) => {
      const a = cur[(i - 1 + n) % n]!;
      const c = cur[(i + 1) % n]!;
      return { x: b.x * 0.5 + (a.x + c.x) * 0.25, z: b.z * 0.5 + (a.z + c.z) * 0.25 };
    });
  }
  const limit = 1 / minRadius;
  for (let iter = 0; iter < maxIter; iter++) {
    const bad: number[] = [];
    for (let i = 0; i < n; i++) {
      if (turnCurvature(cur[(i - 1 + n) % n]!, cur[i]!, cur[(i + 1) % n]!) > limit) bad.push(i);
    }
    if (!bad.length) break;
    const next = cur.slice();
    const mark = new Set<number>();
    for (const i of bad) for (let d = -1; d <= 1; d++) mark.add((i + d + n) % n);
    for (const i of mark) {
      const a = cur[(i - 1 + n) % n]!;
      const b = cur[i]!;
      const c = cur[(i + 1) % n]!;
      next[i] = { x: b.x * 0.5 + (a.x + c.x) * 0.25, z: b.z * 0.5 + (a.z + c.z) * 0.25 };
    }
    cur = next;
  }
  return cur;
}

/**
 * Start the closed loop where it is straightest (the validator wants a straight start line,
 * and where the user happened to begin drawing says nothing about that). Input is resampled
 * uniformly first.
 */
export function rotateToStraightest(points: V2[], spacing = 0.4, windowMeters = 2): V2[] {
  const pts = resampleUniform(points, spacing, true);
  const n = pts.length;
  if (n < 12) return pts;
  const k = pts.map((p, i) => turnCurvature(pts[(i - 1 + n) % n]!, p, pts[(i + 1) % n]!));
  const w = Math.max(3, Math.round(windowMeters / spacing));
  let best = 0;
  let bestScore = Infinity;
  for (let i = 0; i < n; i++) {
    let score = 0;
    for (let j = 0; j < w; j++) score += k[(i + j) % n]!;
    if (score < bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return pts.slice(best).concat(pts.slice(0, best));
}

export function toFloat(points: V2[]): Float32Array {
  const out = new Float32Array(points.length * 2);
  points.forEach((p, i) => {
    out[i * 2] = p.x;
    out[i * 2 + 1] = p.z;
  });
  return out;
}

export function fromFloat(data: ArrayLike<number>): V2[] {
  const out: V2[] = [];
  for (let i = 0; i + 1 < data.length; i += 2) out.push({ x: data[i]!, z: data[i + 1]! });
  return out;
}

export { dist };
