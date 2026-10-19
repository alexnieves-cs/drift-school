import type { TrackSample, TrackSpec } from '../shared/types';
import { BOUNDS, MAX_TRACK_LENGTH, SAMPLE_SPACING, WALL_MARGIN } from '../shared/constants';
import { fnv1a } from '../shared/hash';
import {
  closeIfNear,
  curvatureAt,
  pointsFromArray,
  polygonLength,
  resampleClosed,
  sampleClosedSpline,
  tangentNormal,
  type V2,
} from './spline';
import { validateLoop, validationFromIssues, type BuiltLoop } from './validate';
import type { Validation } from '../shared/types';

export interface BuildOpts {
  halfWidth: number;
  mu?: number;
  mus?: number[];
  seed?: number;
  source?: 'gen' | 'user';
  id?: string;
  /** Rescale loops that fall outside 18-140 m into 28-120 m. The generator uses it; user tracks are built as drawn so what was validated is what trains. */
  autoScale?: boolean;
}

function scaleTo(pts: V2[], origin: V2, s: number): V2[] {
  return pts.map((p) => ({ x: origin.x + (p.x - origin.x) * s, z: origin.z + (p.z - origin.z) * s }));
}

function centroid(pts: V2[]): V2 {
  let x = 0;
  let z = 0;
  for (const p of pts) {
    x += p.x;
    z += p.z;
  }
  const n = pts.length || 1;
  return { x: x / n, z: z / n };
}

export function loopFromControl(control: V2[], halfWidth: number, closed: boolean): BuiltLoop {
  const dense = control.length >= 3 ? sampleClosedSpline(control, 10) : control.slice();
  const center = resampleClosed(dense, SAMPLE_SPACING);
  const curvature: number[] = [];
  const left: V2[] = [];
  const right: V2[] = [];
  const n = center.length;
  for (let i = 0; i < n; i++) {
    const prev = center[(i - 1 + n) % n]!;
    const p = center[i]!;
    const next = center[(i + 1) % n]!;
    const k = curvatureAt(prev, p, next);
    curvature.push(k);
    const tn = tangentNormal(prev, next);
    const widen = 1 + 0.12 * Math.min(1, Math.abs(k) * 3);
    const hw = halfWidth * widen;
    left.push({ x: p.x + tn.nx * hw, z: p.z + tn.nz * hw });
    right.push({ x: p.x - tn.nx * hw, z: p.z - tn.nz * hw });
  }
  return { center, left, right, curvature, halfWidth, length: polygonLength(center), closed };
}

/** Rotate the loop so the start lies on the straightest region. */
export function rotateToStraight(loop: BuiltLoop): BuiltLoop {
  const n = loop.center.length;
  if (n < 8) return loop;
  let best = 0;
  let bestScore = Infinity;
  const window = Math.max(4, Math.round(1.6 / SAMPLE_SPACING));
  for (let i = 0; i < n; i++) {
    let score = 0;
    for (let k = 0; k < window; k++) score += Math.abs(loop.curvature[(i + k) % n] ?? 0);
    if (score < bestScore) {
      bestScore = score;
      best = i;
    }
  }
  const rot = <T>(arr: T[]) => arr.slice(best).concat(arr.slice(0, best));
  return {
    ...loop,
    center: rot(loop.center),
    left: rot(loop.left),
    right: rot(loop.right),
    curvature: rot(loop.curvature),
  };
}

export function referenceTime(center: V2[], curvature: number[], mu: number): number {
  let t = 0;
  const n = center.length;
  for (let i = 0; i < n; i++) {
    const a = center[i]!;
    const b = center[(i + 1) % n]!;
    const ds = Math.hypot(b.x - a.x, b.z - a.z);
    const k = Math.abs(curvature[i] ?? 0);
    const vLim = Math.sqrt((0.62 * mu * 9.81) / Math.max(k, 0.04));
    const v = Math.min(6.2, Math.max(1.8, vLim));
    t += ds / v;
  }
  return t;
}

export function loopToSpec(loop: BuiltLoop, opts: BuildOpts): TrackSpec {
  const n = loop.center.length;
  const samples: TrackSample[] = [];
  const left = new Float32Array(n * 2);
  const right = new Float32Array(n * 2);
  // Closed loops repeat their first point so the last wall segment meets the first.
  const wallPts = loop.closed ? n + 1 : n;
  const wallL = new Float32Array(wallPts * 2);
  const wallR = new Float32Array(wallPts * 2);
  let s = 0;
  const baseMu = opts.mu ?? 1.05;
  for (let i = 0; i < n; i++) {
    const prev = loop.center[(i - 1 + n) % n]!;
    const p = loop.center[i]!;
    const next = loop.center[(i + 1) % n]!;
    const tn = tangentNormal(prev, next);
    const hw = Math.hypot(loop.left[i]!.x - p.x, loop.left[i]!.z - p.z) || opts.halfWidth;
    const mu = opts.mus?.[i] ?? baseMu;
    samples.push({
      x: p.x,
      z: p.z,
      tx: tn.tx,
      tz: tn.tz,
      nx: tn.nx,
      nz: tn.nz,
      curvature: loop.curvature[i] ?? 0,
      s,
      halfWidth: hw,
      mu,
    });
    left[i * 2] = loop.left[i]!.x;
    left[i * 2 + 1] = loop.left[i]!.z;
    right[i * 2] = loop.right[i]!.x;
    right[i * 2 + 1] = loop.right[i]!.z;
    wallL[i * 2] = p.x + tn.nx * (hw + WALL_MARGIN);
    wallL[i * 2 + 1] = p.z + tn.nz * (hw + WALL_MARGIN);
    wallR[i * 2] = p.x - tn.nx * (hw + WALL_MARGIN);
    wallR[i * 2 + 1] = p.z - tn.nz * (hw + WALL_MARGIN);
    const b = loop.center[(i + 1) % n]!;
    s += Math.hypot(b.x - p.x, b.z - p.z);
  }
  if (loop.closed) {
    wallL[n * 2] = wallL[0]!;
    wallL[n * 2 + 1] = wallL[1]!;
    wallR[n * 2] = wallR[0]!;
    wallR[n * 2 + 1] = wallR[1]!;
  }
  const gates: number[] = [];
  for (let g = 0; g < s; g += 2) {
    let best = 0;
    let bd = Infinity;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(samples[i]!.s - g);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    gates.push(best);
  }
  const id =
    opts.id ??
    `${opts.source ?? 'user'}-${fnv1a(
      samples.map((p) => `${p.x.toFixed(3)},${p.z.toFixed(3)}`).join('|'),
    )}`;
  return {
    id,
    seed: opts.seed,
    source: opts.source ?? 'user',
    samples,
    length: s,
    leftEdge: left,
    rightEdge: right,
    walls: [wallL, wallR],
    startIndex: 0,
    tRef: referenceTime(loop.center, loop.curvature, baseMu),
    gates,
  };
}

const MAX_CONTROL_POINTS = 2000;
const HALF_WIDTH_RANGE = [0.1, 5] as const;

/** Cheap rejection of input that would make the spline or overlap checks run away (hostile share URLs, NaN). */
function guardControl(raw: V2[], halfWidth: number): Validation | null {
  if (!Number.isFinite(halfWidth) || halfWidth < HALF_WIDTH_RANGE[0] || halfWidth > HALF_WIDTH_RANGE[1]) {
    return validationFromIssues([{ reason: 'bounds', at: [0, 0] }]);
  }
  if (raw.length > MAX_CONTROL_POINTS) {
    return validationFromIssues([{ reason: 'tooLong', at: [raw[0]!.x, raw[0]!.z] }]);
  }
  let polyLen = 0;
  for (let i = 0; i < raw.length; i++) {
    const p = raw[i]!;
    if (!Number.isFinite(p.x) || !Number.isFinite(p.z) || Math.abs(p.x) > 2 * BOUNDS || Math.abs(p.z) > 2 * BOUNDS) {
      return validationFromIssues([{ reason: 'bounds', at: [Number.isFinite(p.x) ? p.x : 0, Number.isFinite(p.z) ? p.z : 0] }]);
    }
    const q = raw[(i + 1) % raw.length]!;
    polyLen += Math.hypot(q.x - p.x, q.z - p.z);
  }
  if (polyLen > 4 * MAX_TRACK_LENGTH) return validationFromIssues([{ reason: 'tooLong', at: [raw[0]!.x, raw[0]!.z] }]);
  return null;
}

export function buildTrackFromPoints(controlPts: Float32Array | ArrayLike<number>, opts: BuildOpts): TrackSpec {
  const raw = pointsFromArray(controlPts);
  if (guardControl(raw, opts.halfWidth)) throw new Error('control points rejected');
  const closedInfo = closeIfNear(raw, 1);
  let pts = closedInfo.pts;
  if (opts.autoScale && pts.length >= 3) {
    const len = polygonLength(sampleClosedSpline(pts, 8));
    if (len > 1 && (len < 18 || len > 140)) {
      const c = centroid(pts);
      const target = Math.min(120, Math.max(28, len));
      pts = scaleTo(pts, c, target / len);
    }
  }
  let loop = loopFromControl(pts, opts.halfWidth, closedInfo.closed);
  loop = { ...loop, closed: closedInfo.closed && loop.center.length >= 4 };
  if (loop.closed) loop = rotateToStraight(loop);
  return loopToSpec(loop, opts);
}

export function validateTrack(controlPts: Float32Array | ArrayLike<number>, halfWidth: number): Validation {
  const raw = pointsFromArray(controlPts);
  if (raw.length < 3) {
    return { ok: false, issues: [{ reason: 'open', at: [raw[0]?.x ?? 0, raw[0]?.z ?? 0] }] };
  }
  const rejected = guardControl(raw, halfWidth);
  if (rejected) return rejected;
  const closedInfo = closeIfNear(raw, 1);
  if (!closedInfo.closed) {
    const a = raw[0]!;
    return validationFromIssues([{ reason: 'open', at: [a.x, a.z] }]);
  }
  // Same loop the build produces: rotated onto its straightest stretch, so `startStraight` does not depend on where the stroke began.
  let loop = loopFromControl(closedInfo.pts, halfWidth, true);
  loop = rotateToStraight({ ...loop, closed: true });
  return validationFromIssues(validateLoop(loop));
}

export function smoothClosed(pts: V2[], amount = 0.25): V2[] {
  const n = pts.length;
  return pts.map((p, i) => {
    const a = pts[(i - 1 + n) % n]!;
    const b = pts[(i + 1) % n]!;
    return {
      x: p.x * (1 - amount) + ((a.x + b.x) * 0.5) * amount,
      z: p.z * (1 - amount) + ((a.z + b.z) * 0.5) * amount,
    };
  });
}

/** Run the editor's validity checks on a finished spec (for example a generated track). */
export function validateSpec(spec: TrackSpec): Validation {
  const n = spec.samples.length;
  const center: V2[] = [];
  const left: V2[] = [];
  const right: V2[] = [];
  const curvature: number[] = [];
  let halfWidth = Infinity;
  for (let i = 0; i < n; i++) {
    const s = spec.samples[i]!;
    center.push({ x: s.x, z: s.z });
    left.push({ x: spec.leftEdge[i * 2]!, z: spec.leftEdge[i * 2 + 1]! });
    right.push({ x: spec.rightEdge[i * 2]!, z: spec.rightEdge[i * 2 + 1]! });
    curvature.push(s.curvature);
    halfWidth = Math.min(halfWidth, s.halfWidth);
  }
  return validationFromIssues(validateLoop({ center, left, right, curvature, halfWidth, length: spec.length, closed: true }));
}
