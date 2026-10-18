import type { InvalidReason, Validation, ValidationIssue } from '../shared/types';
import {
  BOUNDS,
  EDGE_GAP,
  MAX_TRACK_LENGTH,
  MIN_RADIUS,
  MIN_TRACK_LENGTH,
  START_STRAIGHT,
} from '../shared/constants';
import { closeIfNear, pointsFromArray, polygonLength, type V2 } from './spline';

export interface BuiltLoop {
  center: V2[];
  left: V2[];
  right: V2[];
  curvature: number[];
  halfWidth: number;
  length: number;
  closed: boolean;
}

function segIntersect(a: V2, b: V2, c: V2, d: V2): boolean {
  const o = (p: V2, q: V2, r: V2) => (q.z - p.z) * (r.x - q.x) - (q.x - p.x) * (r.z - q.z);
  const o1 = o(a, b, c);
  const o2 = o(a, b, d);
  const o3 = o(c, d, a);
  const o4 = o(c, d, b);
  return o1 * o2 < 0 && o3 * o4 < 0;
}

function selfIntersects(poly: V2[]): V2 | null {
  const n = poly.length;
  if (n < 4) return null;
  const cell = 4;
  const buckets = new Map<number, number[]>();
  const key = (ix: number, iz: number) => (ix + 4096) * 8192 + (iz + 4096);
  const minx = new Float64Array(n);
  const maxx = new Float64Array(n);
  const minz = new Float64Array(n);
  const maxz = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % n]!;
    const x0 = Math.min(a.x, b.x);
    const x1 = Math.max(a.x, b.x);
    const z0 = Math.min(a.z, b.z);
    const z1 = Math.max(a.z, b.z);
    minx[i] = x0;
    maxx[i] = x1;
    minz[i] = z0;
    maxz[i] = z1;
    const ix0 = Math.floor(x0 / cell);
    const ix1 = Math.floor(x1 / cell);
    const iz0 = Math.floor(z0 / cell);
    const iz1 = Math.floor(z1 / cell);
    for (let ix = ix0; ix <= ix1; ix++) {
      for (let iz = iz0; iz <= iz1; iz++) {
        const k = key(ix, iz);
        let arr = buckets.get(k);
        if (!arr) buckets.set(k, (arr = []));
        arr.push(i);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % n]!;
    const ix0 = Math.floor(minx[i]! / cell);
    const ix1 = Math.floor(maxx[i]! / cell);
    const iz0 = Math.floor(minz[i]! / cell);
    const iz1 = Math.floor(maxz[i]! / cell);
    for (let ix = ix0; ix <= ix1; ix++) {
      for (let iz = iz0; iz <= iz1; iz++) {
        const arr = buckets.get(key(ix, iz));
        if (!arr) continue;
        for (let p = 0; p < arr.length; p++) {
          const j = arr[p]!;
          if (j <= i) continue;
          const gap = Math.min(j - i, n - (j - i));
          if (gap <= 1) continue;
          if (maxx[i]! < minx[j]! || maxx[j]! < minx[i]! || maxz[i]! < minz[j]! || maxz[j]! < minz[i]!) continue;
          const c = poly[j]!;
          const d = poly[(j + 1) % n]!;
          if (segIntersect(a, b, c, d)) return { x: (a.x + c.x) / 2, z: (a.z + c.z) / 2 };
        }
      }
    }
  }
  return null;
}

export function validateLoop(loop: BuiltLoop): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const push = (reason: InvalidReason, at: V2) => issues.push({ reason, at: [at.x, at.z] });
  if (!loop.closed || loop.center.length < 4) {
    const p = loop.center[0] ?? { x: 0, z: 0 };
    push('open', p);
    return issues;
  }
  if (loop.length < MIN_TRACK_LENGTH) push('tooShort', loop.center[0]!);
  if (loop.length > MAX_TRACK_LENGTH) push('tooLong', loop.center[0]!);

  let worstK = 0;
  let worstAt = loop.center[0]!;
  for (let i = 0; i < loop.center.length; i++) {
    const k = Math.abs(loop.curvature[i] ?? 0);
    if (k > worstK) {
      worstK = k;
      worstAt = loop.center[i]!;
    }
    const p = loop.center[i]!;
    if (Math.abs(p.x) > BOUNDS || Math.abs(p.z) > BOUNDS) push('bounds', p);
  }
  if (worstK > 1 / MIN_RADIUS) push('minRadius', worstAt);

  const hitL = selfIntersects(loop.left);
  const hitR = selfIntersects(loop.right);
  const hitC = selfIntersects(loop.center);
  if (hitL) push('selfIntersect', hitL);
  else if (hitR) push('selfIntersect', hitR);
  else if (hitC) push('selfIntersect', hitC);

  const n = loop.center.length;
  const minSepIdx = Math.max(3, Math.round(n * 0.08));
  let overlapAt: V2 | null = null;
  for (let i = 0; i < n && !overlapAt; i++) {
    for (let j = i + minSepIdx; j < n; j++) {
      const wrap = Math.min(j - i, n - (j - i));
      if (wrap < minSepIdx) continue;
      const a = loop.center[i]!;
      const b = loop.center[j]!;
      const need = loop.halfWidth * 2 + EDGE_GAP;
      if (Math.hypot(a.x - b.x, a.z - b.z) < need) {
        overlapAt = { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 };
        break;
      }
    }
  }
  if (overlapAt) push('edgeOverlap', overlapAt);

  let straight = 0;
  let seam = 0;
  let sharp = false;
  let sharpAt = loop.center[0]!;
  for (let i = 0; i < n; i++) {
    const a = loop.center[i]!;
    const b = loop.center[(i + 1) % n]!;
    const ds = Math.hypot(b.x - a.x, b.z - a.z);
    if (Math.abs(loop.curvature[i] ?? 0) > 1 / 8) {
      // Ignore a single closure kink before the straight has begun.
      if (straight < 0.05 && seam < 0.3) {
        seam += ds;
        continue;
      }
      sharp = true;
      sharpAt = a;
      break;
    }
    straight += ds;
    if (straight >= START_STRAIGHT) break;
  }
  if (sharp || straight < START_STRAIGHT) push('startStraight', sharpAt);

  return issues;
}

export function validationFromIssues(issues: ValidationIssue[]): Validation {
  return { ok: issues.length === 0, issues };
}

/** Quick geometric checks used by tests on raw control points (before/after build). */
export function controlClosed(controlPts: ArrayLike<number>): boolean {
  const pts = pointsFromArray(controlPts);
  return closeIfNear(pts, 1).closed || (pts.length > 2 && polygonLength(pts) > 0 && distFirstLast(pts) <= 1);
}

function distFirstLast(pts: V2[]): number {
  if (pts.length < 2) return Infinity;
  return Math.hypot(pts[0]!.x - pts[pts.length - 1]!.x, pts[0]!.z - pts[pts.length - 1]!.z);
}
