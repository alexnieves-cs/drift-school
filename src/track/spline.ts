/** Centripetal Catmull-Rom (alpha = 0.5) and arc-length resampling. */

export interface V2 {
  x: number;
  z: number;
}

const ALPHA = 0.5;

function dist(a: V2, b: V2): number {
  return Math.hypot(b.x - a.x, b.z - a.z);
}

function lerp(a: V2, b: V2, t: number): V2 {
  return { x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t };
}

/** Point between p1 and p2. t is 0..1. Closed-loop callers wrap indices. */
export function catmullRom(p0: V2, p1: V2, p2: V2, p3: V2, t: number): V2 {
  const d01 = Math.max(dist(p0, p1), 1e-6) ** ALPHA;
  const d12 = Math.max(dist(p1, p2), 1e-6) ** ALPHA;
  const d23 = Math.max(dist(p2, p3), 1e-6) ** ALPHA;
  const t0 = 0;
  const t1 = t0 + d01;
  const t2 = t1 + d12;
  const t3 = t2 + d23;
  const u = t1 + (t2 - t1) * Math.min(1, Math.max(0, t));
  const A1 = lerp(p0, p1, (u - t0) / (t1 - t0));
  const A2 = lerp(p1, p2, (u - t1) / (t2 - t1));
  const A3 = lerp(p2, p3, (u - t2) / (t3 - t2));
  const B1 = lerp(A1, A2, (u - t0) / (t2 - t0));
  const B2 = lerp(A2, A3, (u - t1) / (t3 - t1));
  return lerp(B1, B2, (u - t1) / (t2 - t1));
}

export function sampleClosedSpline(pts: V2[], stepsPerSeg = 12): V2[] {
  const n = pts.length;
  const out: V2[] = [];
  if (n < 2) return pts.slice();
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n]!;
    const p1 = pts[i]!;
    const p2 = pts[(i + 1) % n]!;
    const p3 = pts[(i + 2) % n]!;
    for (let s = 0; s < stepsPerSeg; s++) {
      out.push(catmullRom(p0, p1, p2, p3, s / stepsPerSeg));
    }
  }
  return out;
}

/** Resample a closed polyline to uniform arc length. */
export function resampleClosed(pts: V2[], spacing: number): V2[] {
  if (pts.length < 2) return pts.slice();
  const seg: number[] = [];
  let total = 0;
  for (let i = 0; i < pts.length; i++) {
    const d = dist(pts[i]!, pts[(i + 1) % pts.length]!);
    seg.push(d);
    total += d;
  }
  if (total < spacing * 2) return pts.slice();
  const count = Math.max(8, Math.round(total / spacing));
  const step = total / count;
  const out: V2[] = [];
  let acc = 0;
  let idx = 0;
  let into = 0;
  for (let k = 0; k < count; k++) {
    const target = k * step;
    while (acc + (seg[idx]! - into) < target - 1e-9 && idx < pts.length) {
      acc += seg[idx]! - into;
      into = 0;
      idx++;
      if (idx >= pts.length) break;
    }
    const remain = target - acc;
    const p = pts[idx % pts.length]!;
    const q = pts[(idx + 1) % pts.length]!;
    const len = seg[idx % pts.length] || 1;
    const t = len > 1e-8 ? Math.min(1, (into + remain) / len) : 0;
    out.push(lerp(p, q, t));
  }
  return out;
}

export function curvatureAt(prev: V2, p: V2, next: V2): number {
  const h1 = Math.max(dist(prev, p), 1e-6);
  const h2 = Math.max(dist(p, next), 1e-6);
  const h = (h1 + h2) * 0.5;
  const xp = (next.x - prev.x) / (2 * h);
  const zp = (next.z - prev.z) / (2 * h);
  const xpp = (next.x - 2 * p.x + prev.x) / (h * h);
  const zpp = (next.z - 2 * p.z + prev.z) / (h * h);
  const denom = (xp * xp + zp * zp) ** 1.5;
  if (denom < 1e-8) return 0;
  return (xp * zpp - zp * xpp) / denom;
}

export function tangentNormal(prev: V2, next: V2): { tx: number; tz: number; nx: number; nz: number } {
  let tx = next.x - prev.x;
  let tz = next.z - prev.z;
  const l = Math.hypot(tx, tz) || 1;
  tx /= l;
  tz /= l;
  return { tx, tz, nx: -tz, nz: tx };
}

export function polygonLength(pts: V2[]): number {
  let s = 0;
  for (let i = 0; i < pts.length; i++) s += dist(pts[i]!, pts[(i + 1) % pts.length]!);
  return s;
}

export function pointsFromArray(data: ArrayLike<number>): V2[] {
  const out: V2[] = [];
  for (let i = 0; i + 1 < data.length; i += 2) out.push({ x: data[i]!, z: data[i + 1]! });
  return out;
}

export function closeIfNear(pts: V2[], maxGap: number): { pts: V2[]; closed: boolean } {
  if (pts.length < 3) return { pts, closed: false };
  const gap = dist(pts[0]!, pts[pts.length - 1]!);
  if (gap <= maxGap) {
    const copy = pts.slice(0, -1);
    if (copy.length >= 3 && dist(copy[0]!, copy[copy.length - 1]!) > 1e-4) return { pts: copy, closed: true };
    if (gap < 1e-3) return { pts: pts.slice(0, -1), closed: true };
    return { pts: copy.length >= 3 ? copy : pts, closed: copy.length >= 3 };
  }
  return { pts, closed: false };
}
