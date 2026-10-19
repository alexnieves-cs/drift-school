import { describe, expect, it } from 'vitest';
import { makeRng } from '../../src/shared/rng';
import { RAY_ARC, RAY_COUNT, RAY_MAX } from '../../src/shared/constants';
import type { TrackSpec } from '../../src/shared/types';
import { RayField, SENSOR_FLAGS } from '../../src/sim/sensors';
import { getWallGrid, resolveWallContact, type WallBody, type WallResult } from '../../src/sim/wallGrid';
import { generateTrack } from '../../src/track/generate';

/** The pre-optimization loops, copied verbatim, as the reference. */

function segHit(ax: number, az: number, bx: number, bz: number, cx: number, cz: number, dx: number, dz: number): number {
  const rxs = bx - ax;
  const rzs = bz - az;
  const sxs = dx - cx;
  const szs = dz - cz;
  const den = rxs * szs - rzs * sxs;
  if (Math.abs(den) < 1e-8) return -1;
  const qpx = cx - ax;
  const qpz = cz - az;
  const t = (qpx * szs - qpz * sxs) / den;
  const u = (qpx * rzs - qpz * rxs) / den;
  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) return t;
  return -1;
}

function bruteRays(track: TrackSpec, x: number, z: number, yaw: number, nearest: boolean): number[] {
  const out: number[] = [];
  for (let r = 0; r < RAY_COUNT; r++) {
    const ang = yaw + -RAY_ARC / 2 + (r / (RAY_COUNT - 1)) * RAY_ARC;
    const dx = Math.cos(ang);
    const dz = Math.sin(ang);
    let best = RAY_MAX;
    const x2 = x + dx * RAY_MAX;
    const z2 = z + dz * RAY_MAX;
    for (const s of track.walls) {
      for (let i = 0; i + 3 < s.length; i += 2) {
        const hit = segHit(x, z, x2, z2, s[i]!, s[i + 1]!, s[i + 2]!, s[i + 3]!);
        if (nearest) {
          if (hit >= 0) best = Math.min(best, hit * RAY_MAX);
        } else if (hit >= 0 && hit < best) best = hit * RAY_MAX;
      }
    }
    out.push(Math.fround(best / RAY_MAX));
  }
  return out;
}

function bruteWallPush(
  walls: Float32Array[],
  car: WallBody,
): { contact: boolean; impact: number } {
  const margin = 0.18;
  let hit = false;
  let impact = 0;
  for (let w = 0; w < walls.length; w++) {
    const poly = walls[w]!;
    for (let i = 0; i + 3 < poly.length; i += 2) {
      const x1 = poly[i]!;
      const z1 = poly[i + 1]!;
      const x2 = poly[i + 2]!;
      const z2 = poly[i + 3]!;
      const dx = x2 - x1;
      const dz = z2 - z1;
      const len2 = dx * dx + dz * dz || 1;
      let t = ((car.px - x1) * dx + (car.pz - z1) * dz) / len2;
      t = Math.max(0, Math.min(1, t));
      const cx = x1 + dx * t;
      const cz = z1 + dz * t;
      const ox = car.px - cx;
      const oz = car.pz - cz;
      const dist = Math.hypot(ox, oz);
      if (dist < margin) {
        hit = true;
        const nx = ox / (dist || 1);
        const nz = oz / (dist || 1);
        car.px += nx * (margin - dist);
        car.pz += nz * (margin - dist);
        const cy = Math.cos(car.yaw);
        const sy = Math.sin(car.yaw);
        const worldVx = cy * car.vx - sy * car.vz;
        const worldVz = sy * car.vx + cy * car.vz;
        const vn = worldVx * nx + worldVz * nz;
        if (vn < 0) {
          const e = 0.3;
          const wx = worldVx - (1 + e) * vn * nx;
          const wz = worldVz - (1 + e) * vn * nz;
          car.vx = cy * wx + sy * wz;
          car.vz = -sy * wx + cy * wz;
          impact = Math.max(impact, -vn);
        }
      }
    }
  }
  return { contact: hit, impact };
}

const TRACKS: TrackSpec[] = [
  generateTrack({ seed: 3, stage: 0 }),
  generateTrack({ seed: 4, stage: 2 }),
  generateTrack({ seed: 9, stage: 3 }),
  generateTrack({ seed: 12, stage: 4 }),
  generateTrack({ seed: 31, stage: 4 }),
];

/** Poses on the centerline with lateral offsets that reach past the walls, plus off-track junk. */
function poses(track: TrackSpec, n: number, seed: number) {
  const rng = makeRng(seed, 'wallgrid-poses');
  const out: { x: number; z: number; yaw: number }[] = [];
  for (let i = 0; i < n; i++) {
    const s = track.samples[rng.int(track.samples.length)]!;
    const lat = rng.range(-1, 1) * (s.halfWidth + 0.5);
    out.push({ x: s.x + s.nx * lat, z: s.z + s.nz * lat, yaw: rng.range(-Math.PI, Math.PI) });
  }
  for (let i = 0; i < n / 10; i++) out.push({ x: rng.range(-90, 90), z: rng.range(-90, 90), yaw: rng.range(-3, 3) });
  return out;
}

describe('wall grid equals the brute-force loops', () => {
  for (const nearest of [false, true]) {
    it(`ray distances are identical (${nearest ? 'nearest' : 'legacy'} rule)`, () => {
      const saved = SENSOR_FLAGS.nearestRay;
      SENSOR_FLAGS.nearestRay = nearest;
      try {
        let checked = 0;
        for (const track of TRACKS) {
          const rf = new RayField();
          rf.setTrack(track);
          const out = new Float32Array(RAY_COUNT);
          for (const p of poses(track, 400, 5)) {
            rf.cast(p.x, p.z, p.yaw, out, 0, null, 0);
            const want = bruteRays(track, p.x, p.z, p.yaw, nearest);
            for (let r = 0; r < RAY_COUNT; r++) expect(out[r], `${track.id} ray ${r} @ ${p.x},${p.z}`).toBe(want[r]);
            checked++;
          }
        }
        expect(checked).toBeGreaterThan(2000);
      } finally {
        SENSOR_FLAGS.nearestRay = saved;
      }
    });
  }

  it('wall contact leaves the car in exactly the same state', () => {
    const rng = makeRng(8, 'wallgrid-contact');
    let contacts = 0;
    let impacts = 0;
    const res: WallResult = { contact: false, hit: false, impact: 0 };
    for (const track of TRACKS) {
      const g = getWallGrid(track.walls);
      for (const p of poses(track, 3000, 6)) {
        // park the car right on a wall half the time so contacts (and corner double contacts) happen
        const mk = (): WallBody => ({ px: p.x, pz: p.z, yaw: p.yaw, vx: 0, vz: 0 });
        const a = mk();
        if (rng.next() < 0.5) {
          const seg = rng.int(g.nSeg) * 4;
          const t = rng.next();
          a.px = g.seg[seg]! + (g.seg[seg + 2]! - g.seg[seg]!) * t + rng.range(-0.25, 0.25);
          a.pz = g.seg[seg + 1]! + (g.seg[seg + 3]! - g.seg[seg + 1]!) * t + rng.range(-0.25, 0.25);
        }
        a.vx = rng.range(-6, 6);
        a.vz = rng.range(-2, 2);
        const b = { ...a };
        const want = bruteWallPush(track.walls, b);
        resolveWallContact(a, g, res);
        expect(a).toEqual(b);
        expect(res.contact).toBe(want.contact);
        expect(res.impact).toBe(want.impact);
        expect(res.hit).toBe(want.impact > 0.4);
        if (want.contact) contacts++;
        if (want.impact > 0) impacts++;
      }
    }
    expect(contacts).toBeGreaterThan(1000);
    expect(impacts).toBeGreaterThan(300);
  });

  it('keeps per-cell work small however long the track is (the point of the grid)', () => {
    for (const track of TRACKS) {
      const g = getWallGrid(track.walls);
      let maxNear = 0;
      let maxRay = 0;
      for (let c = 0; c + 1 < g.nearOff.length; c++) maxNear = Math.max(maxNear, g.nearOff[c + 1]! - g.nearOff[c]!);
      for (let c = 0; c + 1 < g.rayOff.length; c++) maxRay = Math.max(maxRay, g.rayOff[c + 1]! - g.rayOff[c]!);
      // brute force touches every segment (thousands); a cell holds a few dozen at the very most
      expect(g.nSeg).toBeGreaterThan(400);
      expect(maxNear, `${track.id} near list`).toBeLessThan(g.nSeg / 4);
      expect(maxRay, `${track.id} ray list`).toBeLessThan(g.nSeg / 8);
    }
  });

  it('is shared across TrackSpec copies that reuse the walls array', () => {
    const t = TRACKS[0]!;
    expect(getWallGrid(t.walls)).toBe(getWallGrid({ ...t, id: 'copy' }.walls));
  });
});
