import type { TrackGenParams, TrackSpec } from '../shared/types';
import { makeRng, type RNG } from '../shared/rng';
import { STAGE_MIN_RADIUS, STAGE_WIDTH } from '../shared/constants';
import { buildTrackFromPoints, loopFromControl, loopToSpec, rotateToStraight, smoothClosed } from './build';
import { validateLoop } from './validate';
import { polygonLength, type V2 } from './spline';

function convexHull(pts: V2[]): V2[] {
  const p = pts.slice().sort((a, b) => (a.x === b.x ? a.z - b.z : a.x - b.x));
  if (p.length <= 2) return p;
  const cross = (o: V2, a: V2, b: V2) => (a.x - o.x) * (b.z - o.z) - (a.z - o.z) * (b.x - o.x);
  const lower: V2[] = [];
  for (const pt of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, pt) <= 0) lower.pop();
    lower.push(pt);
  }
  const upper: V2[] = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const pt = p[i]!;
    while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, pt) <= 0) upper.pop();
    upper.push(pt);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function displace(pts: V2[], rng: RNG, amp: number): V2[] {
  const out: V2[] = [];
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const a = pts[i]!;
    const b = pts[(i + 1) % n]!;
    out.push(a);
    const mx = (a.x + b.x) / 2;
    const mz = (a.z + b.z) / 2;
    let nx = -(b.z - a.z);
    let nz = b.x - a.x;
    const l = Math.hypot(nx, nz) || 1;
    nx /= l;
    nz /= l;
    const mag = rng.range(-amp, amp);
    out.push({ x: mx + nx * mag, z: mz + nz * mag });
  }
  return out;
}

function pushApart(pts: V2[], minD: number): V2[] {
  const p = pts.map((q) => ({ ...q }));
  for (let iter = 0; iter < 4; iter++) {
    for (let i = 0; i < p.length; i++) {
      for (let j = i + 2; j < p.length; j++) {
        if (i === 0 && j === p.length - 1) continue;
        const dx = p[j]!.x - p[i]!.x;
        const dz = p[j]!.z - p[i]!.z;
        const d = Math.hypot(dx, dz) || 1e-4;
        if (d < minD) {
          const push = ((minD - d) / d) * 0.5;
          p[i]!.x -= dx * push;
          p[i]!.z -= dz * push;
          p[j]!.x += dx * push;
          p[j]!.z += dz * push;
        }
      }
    }
  }
  return p;
}

function limitTurns(pts: V2[], minRadius: number): V2[] {
  const p = pts.map((q) => ({ ...q }));
  const maxTurn = Math.min(2.2, 2.4 / Math.sqrt(minRadius));
  for (let iter = 0; iter < 6; iter++) {
    for (let i = 0; i < p.length; i++) {
      const a = p[(i - 1 + p.length) % p.length]!;
      const b = p[i]!;
      const c = p[(i + 1) % p.length]!;
      const a1 = Math.atan2(b.z - a.z, b.x - a.x);
      const a2 = Math.atan2(c.z - b.z, c.x - b.x);
      let d = a2 - a1;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      if (Math.abs(d) > maxTurn) {
        b.x = b.x * 0.7 + ((a.x + c.x) * 0.5) * 0.3;
        b.z = b.z * 0.7 + ((a.z + c.z) * 0.5) * 0.3;
      }
    }
  }
  return p;
}

function stadium(rng: RNG, stage: number): V2[] {
  const radius = STAGE_MIN_RADIUS[stage] ?? 4;
  const straight = 6 + stage * 2 + rng.range(0, 4);
  const pts: V2[] = [];
  const cap = 6;
  for (let i = 0; i <= cap; i++) {
    const a = -Math.PI / 2 + (i / cap) * Math.PI;
    pts.push({ x: straight / 2 + Math.cos(a) * radius, z: Math.sin(a) * radius });
  }
  for (let i = 1; i <= cap; i++) {
    const a = Math.PI / 2 + (i / cap) * Math.PI;
    pts.push({ x: -straight / 2 + Math.cos(a) * radius, z: Math.sin(a) * radius });
  }
  return pts;
}

function tryOnce(seed: number, stage: number, ampScale: number): TrackSpec | null {
  const rng = makeRng(seed, `track/s${stage}/a${ampScale.toFixed(2)}`);
  const K = 8 + rng.int(13);
  const radius = (STAGE_MIN_RADIUS[stage] ?? 4) + 6 + rng.range(0, 5);
  const raw: V2[] = [];
  for (let i = 0; i < K; i++) {
    const ang = (i / K) * Math.PI * 2 + rng.range(-0.25, 0.25);
    const r = radius * (0.75 + rng.range(0, 0.5));
    raw.push({ x: Math.cos(ang) * r, z: Math.sin(ang) * r });
  }
  let pts = convexHull(raw);
  if (pts.length < 4) pts = stadium(rng, stage);
  const amp = [0.15, 0.45, 0.9, 1.5, 1.8][stage]! * ampScale;
  if (amp > 0.05) pts = displace(pts, rng, amp);
  pts = pushApart(pts, (STAGE_MIN_RADIUS[stage] ?? 2) * 0.8);
  pts = limitTurns(pts, STAGE_MIN_RADIUS[stage] ?? 2);
  for (let s = 0; s < 2; s++) pts = smoothClosed(pts, 0.35);
  const len = polygonLength(pts);
  const target = 32 + stage * 6 + rng.range(0, 10);
  if (len > 1) {
    const sc = target / len;
    pts = pts.map((p) => ({ x: p.x * sc, z: p.z * sc }));
  }
  const half = (STAGE_WIDTH[stage] ?? 1) / 2;
  let loop = loopFromControl(pts, half, true);
  loop.closed = true;
  loop = rotateToStraight(loop);
  const mus = surfaceMus(makeRng(seed, `mu/${stage}`), stage, loop.center.length);
  const issues = validateLoop(loop);
  if (issues.length) return null;
  const spec = loopToSpec(loop, { halfWidth: half, mu: mus[0], mus, seed, source: 'gen', id: `gen-${seed}-s${stage}` });
  spec.seed = seed;
  return spec;
}

function surfaceMus(rng: RNG, stage: number, n: number): number[] {
  const mus = new Array<number>(n);
  if (stage < 3) {
    mus.fill(stage === 0 ? 1.2 : 1.05);
    return mus;
  }
  const zones = stage === 4 ? [1.15, 0.95, 0.72, 0.55] : [1.15, 0.95, 0.75];
  let i = 0;
  while (i < n) {
    const len = 8 + rng.int(20);
    const mu = zones[rng.int(zones.length)]!;
    for (let k = 0; k < len && i < n; k++, i++) mus[i] = mu;
  }
  return mus;
}

/** Counts how often the generator gave up on random tracks and used the stadium fallback. Tests assert it stays 0. */
export const generatorStats = { fallbacks: 0 };

function fallbackOval(seed: number, stage: number): TrackSpec {
  generatorStats.fallbacks++;
  const rng = makeRng(seed, `fallback/${stage}`);
  const pts = stadium(rng, Math.max(0, stage - 1));
  const len = polygonLength(pts);
  const target = 36 + stage * 4;
  const sc = target / len;
  const scaled = pts.map((p) => ({ x: p.x * sc, z: p.z * sc }));
  const flat = new Float32Array(scaled.length * 2);
  scaled.forEach((p, i) => {
    flat[i * 2] = p.x;
    flat[i * 2 + 1] = p.z;
  });
  const half = (STAGE_WIDTH[stage] ?? 1) / 2;
  const spec = buildTrackFromPoints(flat, {
    autoScale: true,
    halfWidth: half,
    mu: stage >= 4 ? 0.85 : 1.1,
    seed,
    source: 'gen',
    id: `gen-${seed}-s${stage}`,
  });
  spec.seed = seed;
  return spec;
}

export function generateTrack(p: TrackGenParams): TrackSpec {
  const stage = Math.max(0, Math.min(4, p.stage | 0));
  let seed = p.seed >>> 0;
  for (let attempt = 0; attempt < 24; attempt++) {
    const amp = attempt < 8 ? 1 : attempt < 16 ? 0.45 : 0.15;
    const spec = tryOnce(seed, stage, amp);
    if (spec && spec.samples.length > 10 && spec.length >= 15 && spec.length <= 150) {
      spec.id = `gen-${p.seed}-s${stage}`;
      spec.seed = p.seed;
      return spec;
    }
    seed = (seed + 1) >>> 0;
  }
  const spec = fallbackOval(p.seed >>> 0, stage);
  spec.id = `gen-${p.seed}-s${stage}`;
  spec.seed = p.seed;
  return spec;
}

export function trackFingerprint(t: TrackSpec): string {
  const parts: string[] = [t.length.toFixed(5), String(t.samples.length)];
  for (const s of t.samples) {
    parts.push(
      [s.x, s.z, s.tx, s.tz, s.nx, s.nz, s.curvature, s.s, s.halfWidth, s.mu].map((n) => n.toFixed(5)).join(','),
    );
  }
  return parts.join(';');
}
