import type { TrackSample, TrackSpec } from '../shared/types';

export interface SampleQuery {
  index: number;
  lateral: number;
  heading: number;
  sample: TrackSample;
}

/** Walk from a cached index. Returns the nearest centerline sample. */
export function nearestSample(track: TrackSpec, x: number, z: number, hint = 0): SampleQuery {
  const n = track.samples.length;
  if (n === 0) {
    const dummy: TrackSample = { x: 0, z: 0, tx: 1, tz: 0, nx: 0, nz: 1, curvature: 0, s: 0, halfWidth: 1, mu: 1 };
    return { index: 0, lateral: 0, heading: 0, sample: dummy };
  }
  const idx = ((hint % n) + n) % n;
  let best = idx;
  let bestD = Infinity;
  const span = Math.min(n, 24);
  for (let pass = 0; pass < 2; pass++) {
    const from = pass === 0 ? idx - span : 0;
    const to = pass === 0 ? idx + span : n - 1;
    let improved = false;
    for (let i = from; i <= to; i++) {
      const j = ((i % n) + n) % n;
      const s = track.samples[j]!;
      const d = (s.x - x) ** 2 + (s.z - z) ** 2;
      if (d < bestD) {
        bestD = d;
        best = j;
        improved = true;
      }
    }
    if (pass === 0 && improved && bestD < 4) break;
  }
  const s = track.samples[best]!;
  const lateral = (x - s.x) * s.nx + (z - s.z) * s.nz;
  return { index: best, lateral, heading: 0, sample: s };
}

export function sampleAtS(track: TrackSpec, s: number): TrackSample {
  const n = track.samples.length;
  if (n === 0) {
    return { x: 0, z: 0, tx: 1, tz: 0, nx: 0, nz: 1, curvature: 0, s: 0, halfWidth: 1, mu: 1 };
  }
  let ss = s % track.length;
  if (ss < 0) ss += track.length;
  let i = 0;
  while (i < n - 1 && track.samples[i + 1]!.s < ss) i++;
  const a = track.samples[i]!;
  const b = track.samples[(i + 1) % n]!;
  const span = Math.max(1e-4, (b.s > a.s ? b.s - a.s : track.length - a.s));
  const t = Math.min(1, Math.max(0, (ss - a.s) / span));
  const lerp = (u: number, v: number) => u + (v - u) * t;
  return {
    x: lerp(a.x, b.x),
    z: lerp(a.z, b.z),
    tx: lerp(a.tx, b.tx),
    tz: lerp(a.tz, b.tz),
    nx: lerp(a.nx, b.nx),
    nz: lerp(a.nz, b.nz),
    curvature: lerp(a.curvature, b.curvature),
    s: ss,
    halfWidth: lerp(a.halfWidth, b.halfWidth),
    mu: lerp(a.mu, b.mu),
  };
}

export function curvaturesAhead(track: TrackSpec, s: number, distances: readonly number[]): number[] {
  return distances.map((d) => sampleAtS(track, s + d).curvature);
}
