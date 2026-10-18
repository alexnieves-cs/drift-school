import type { TrackSpec, Validation } from '../shared/types';
import { buildTrackFromPoints, validateTrack } from '../track/build';
import { MIN_RADIUS } from '../shared/constants';
import { dist, relaxCurvature, rotateToStraightest, smoothStroke, toFloat, type V2 } from './stroke';

/** A stroke ends "closed" when the pointer comes back within this many meters of its start. */
export const CLOSE_DISTANCE = 1;

/** Append a near-duplicate of the first point so the closed spline has no seam. */
export function closeEnds(points: V2[], closed: boolean): V2[] {
  if (!closed || points.length < 3) return points;
  const s = points[0]!;
  const e = points[points.length - 1]!;
  if (dist(s, e) <= CLOSE_DISTANCE) return points;
  return [...points, { x: s.x + 0.02, z: s.z }];
}

export interface StrokeResult {
  /** Control points after RDP + Chaikin x2 (or the raw stroke if smoothing broke validity). */
  points: V2[];
  validation: Validation;
  track: TrackSpec | null;
  closed: boolean;
}

/**
 * The editor's stroke pipeline without a canvas: freehand stroke -> RDP -> Chaikin x2 ->
 * (closed loop) -> centripetal Catmull-Rom + arc-length resample inside buildTrackFromPoints.
 * The smoothed candidate is preferred; the raw stroke is the fallback when smoothing
 * turns a valid drawing invalid. `TrackEditor.onUp` and the eval both call this.
 */
export function processStroke(raw: V2[], halfWidth: number, mu: number): StrokeResult {
  const first = raw[0];
  const last = raw[raw.length - 1];
  const closed = !!first && !!last && dist(first, last) < CLOSE_DISTANCE;
  const smooth = smoothStroke(raw, closed);
  // Preferred: smoothed and curvature-limited. Fallbacks: smoothed only, then the raw stroke.
  const candidates = closed
    ? [
        rotateToStraightest(relaxCurvature(smooth, Math.max(MIN_RADIUS, halfWidth * 1.3))),
        rotateToStraightest(smooth),
        rotateToStraightest(raw),
      ]
    : [smooth, raw.slice()];
  let points = candidates[0]!;
  for (const candidate of candidates) {
    const closedPts = closeEnds(candidate, closed);
    points = closedPts;
    if (validateTrack(toFloat(closedPts), halfWidth).ok) break;
  }
  const flat = toFloat(points);
  const validation = validateTrack(flat, halfWidth);
  let track: TrackSpec | null = null;
  if (points.length >= 4) {
    try {
      track = buildTrackFromPoints(flat, { halfWidth, mu, source: 'user' });
    } catch {
      track = null;
    }
  }
  return { points, validation, track, closed };
}

/**
 * The S-curve the M4 acceptance test draws: a closed loop whose top side is an S
 * (offset amp * sin(2 pi u) * sin(pi u), tangent-continuous with the ends) between two
 * half-circle ends. World meters, centered on the origin, about 55 m around, minimum
 * corner radius about 3 m. The path is sampled the way a mouse reports it (steps of about
 * 0.2 m) and ends 0.3 m from where it began, so the editor closes it.
 */
export function sCurveStroke(): V2[] {
  const half = 7; // half length of the straights
  const r = 4; // end radius
  const amp = 1.6; // S amplitude
  const step = 0.2;
  const out: V2[] = [];
  const push = (x: number, z: number) => out.push({ x, z });
  // Bottom side, left to right (z = +r).
  for (let x = -half; x < half; x += step) push(x, r);
  // Right end, bottom to top.
  const arc = Math.PI * r;
  for (let s = 0; s < arc; s += step) {
    const phi = s / r;
    push(half + r * Math.sin(phi), r * Math.cos(phi));
  }
  // Top side, right to left, one S period.
  for (let x = half; x > -half; x -= step) {
    const u = (half - x) / (2 * half);
    push(x, -r + amp * Math.sin(2 * Math.PI * u) * Math.sin(Math.PI * u));
  }
  // Left end, top to bottom; stop 0.3 m short of the start so the loop closes by proximity.
  for (let s = 0; s < arc - 0.3; s += step) {
    const phi = s / r;
    push(-half - r * Math.sin(phi), -r * Math.cos(phi));
  }
  return out;
}

/**
 * A second hand-drawn shape for held-out checks: a rounded triangle, r = R (1 + 0.1 cos 3 theta),
 * about 57 m around, sampled like a mouse (about 0.25 m apart) and ending 0.4 m before the
 * start. Never used for fine-tuning in the eval.
 */
export function triangleStroke(): V2[] {
  const out: V2[] = [];
  const R = 9;
  const total = 230;
  for (let k = 0; k < total; k++) {
    const t = (k / total) * Math.PI * 2;
    const r = R * (1 + 0.1 * Math.cos(3 * t));
    out.push({ x: r * Math.cos(t), z: r * Math.sin(t) });
  }
  return out;
}
