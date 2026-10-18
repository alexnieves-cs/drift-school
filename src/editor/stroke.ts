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

/** Drop trailing points that sit within `gap` of the start, so a closed stroke has no duplicate seam. */
export function trimClosingTail(points: V2[], gap = 1): V2[] {
  const body = points.slice();
  while (body.length > 3 && dist(body[body.length - 1]!, body[0]!) < gap) body.pop();
  return body;
}

/** Even arc-length spacing. Uneven control points make the spline overshoot where spacing jumps. */
export function resamplePolyline(points: V2[], spacing: number, closed: boolean): V2[] {
  const pts = closed ? [...points, points[0]!] : points;
  let total = 0;
  for (let i = 1; i < pts.length; i++) total += dist(pts[i - 1]!, pts[i]!);
  const count = Math.max(closed ? 4 : 2, Math.round(total / spacing));
  const step = total / (closed ? count : count - 1);
  const out: V2[] = [];
  let seg = 1;
  let acc = 0;
  for (let k = 0; k < (closed ? count : count - 1); k++) {
    const target = k * step;
    while (seg < pts.length - 1 && acc + dist(pts[seg - 1]!, pts[seg]!) < target) {
      acc += dist(pts[seg - 1]!, pts[seg]!);
      seg++;
    }
    const a = pts[seg - 1]!;
    const b = pts[seg]!;
    const len = dist(a, b) || 1;
    const t = Math.min(1, Math.max(0, (target - acc) / len));
    out.push({ x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t });
  }
  if (!closed) out.push({ ...pts[pts.length - 1]! });
  return out;
}

function turnCurvature(a: V2, b: V2, c: V2): number {
  const ab = dist(a, b);
  const bc = dist(b, c);
  const ac = dist(a, c);
  const area2 = Math.abs((b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x));
  const den = ab * bc * ac;
  return den < 1e-9 ? 0 : (2 * area2) / den;
}

/**
 * Curvature limiter. Points whose turn is tighter than `maxCurvature` are pulled toward the
 * midpoint of their neighbours, repeatedly, until none are. Hand-drawn jitter otherwise leaves
 * local kinks that fold the inside edge over itself.
 */
export function limitCurvature(points: V2[], closed: boolean, maxCurvature: number, iterations = 60): V2[] {
  let cur = points.map((p) => ({ ...p }));
  const n = cur.length;
  if (n < 4) return cur;
  const lo = closed ? 0 : 1;
  const hi = closed ? n : n - 1;
  for (let it = 0; it < iterations; it++) {
    let moved = false;
    const next = cur.map((p) => ({ ...p }));
    for (let i = lo; i < hi; i++) {
      const a = cur[(i - 1 + n) % n]!;
      const b = cur[i]!;
      const c = cur[(i + 1) % n]!;
      if (turnCurvature(a, b, c) > maxCurvature) {
        next[i] = { x: b.x + 0.5 * ((a.x + c.x) / 2 - b.x), z: b.z + 0.5 * ((a.z + c.z) / 2 - b.z) };
        moved = true;
      }
    }
    cur = next;
    if (!moved) break;
  }
  return cur;
}

/**
 * RDP, then Chaikin x2, then the curvature limiter. A closed stroke is smoothed as a loop and
 * returned without a repeated first point.
 */
export function smoothStroke(points: V2[], closed: boolean, halfWidth = 1): V2[] {
  const body = closed ? trimClosingTail(points) : points;
  let cur = rdp(body, 0.12);
  for (let i = 0; i < 2; i++) cur = chaikin(cur, closed);
  cur = resamplePolyline(cur, 0.3, closed);
  return limitCurvature(cur, closed, 0.7 / halfWidth);
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
