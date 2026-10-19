import type { TrackSpec } from '../shared/types';
import { LOOKAHEADS, OBS, RAY_ARC, RAY_COUNT, RAY_MAX, V_SCALE, YAW_SCALE } from '../shared/constants';
import type { RNG } from '../shared/rng';
import { randn } from '../shared/rng';
import { curvaturesAhead, nearestSample } from '../track/sample';
import type { Vehicle } from './car/vehicle';
import { getWallGrid, type WallGrid } from './wallGrid';

/**
 * Flags that change what the sensors report. See NOTES/phys.md ("ray bug").
 */
export const SENSOR_FLAGS = {
  /**
   * true (default): the real nearest hit. false: the original buggy rule, kept for the
   * equivalence tests. Every segment the ray crosses is visited in wall order and `hit < best`
   * compared a fraction t in [0,1] against `best` in metres, so a farther crossing visited later
   * could overwrite a nearer one.
   */
  nearestRay: true,
};

export class RayField {
  private grid: WallGrid | null = null;
  private hitId = new Int32Array(64);
  private hitT = new Float64Array(64);

  setTrack(track: TrackSpec): void {
    this.grid = getWallGrid(track.walls);
  }

  cast(x: number, z: number, yaw: number, out: Float32Array, offset: number, rng: RNG | null, noise: number): void {
    const g = this.grid;
    for (let r = 0; r < RAY_COUNT; r++) {
      const ang = yaw + -RAY_ARC / 2 + (r / (RAY_COUNT - 1)) * RAY_ARC;
      const dx = Math.cos(ang);
      const dz = Math.sin(ang);
      let best = g ? this.castOne(g, x, z, x + dx * RAY_MAX, z + dz * RAY_MAX, dx, dz) : RAY_MAX;
      if (rng && noise > 0) {
        if (rng.next() < 0.01 * Math.max(noise, 0.5)) best = RAY_MAX;
        else best = Math.min(RAY_MAX, Math.max(0, best + randn(rng) * 0.01 * noise));
      }
      out[offset + r] = best / RAY_MAX;
    }
  }

  /** Distance in metres along one ray, walking only the grid cells the ray crosses. */
  private castOne(g: WallGrid, x: number, z: number, x2: number, z2: number, dx: number, dz: number): number {
    const inv = g.inv;
    const fx = (x - g.ox) * inv;
    const fz = (z - g.oz) * inv;
    let ix = Math.floor(fx);
    let iz = Math.floor(fz);
    const adx = Math.abs(dx);
    const adz = Math.abs(dz);
    const stepX = dx > 0 ? 1 : -1;
    const stepZ = dz > 0 ? 1 : -1;
    const tDx = adx > 1e-12 ? 1 / adx : Infinity;
    const tDz = adz > 1e-12 ? 1 / adz : Infinity;
    let tMx = adx > 1e-12 ? (dx > 0 ? ix + 1 - fx : fx - ix) * tDx : Infinity;
    let tMz = adz > 1e-12 ? (dz > 0 ? iz + 1 - fz : fz - iz) * tDz : Infinity;
    const L = RAY_MAX * inv + 1e-6;
    const { nx, nz, rayOff, rayIds, seg, stamp } = g;
    let tick = ++g.tick;
    if (tick > 2_000_000_000) {
      stamp.fill(0);
      g.tick = tick = 1;
    }
    let nHits = 0;
    for (;;) {
      if (ix >= 0 && ix < nx && iz >= 0 && iz < nz) {
        const c = iz * nx + ix;
        for (let p = rayOff[c]!, e = rayOff[c + 1]!; p < e; p++) {
          const id = rayIds[p]!;
          if (stamp[id] === tick) continue;
          stamp[id] = tick;
          const k = id * 4;
          const hit = segHit(x, z, x2, z2, seg[k]!, seg[k + 1]!, seg[k + 2]!, seg[k + 3]!);
          if (hit >= 0) {
            if (nHits === this.hitId.length) this.growHits();
            this.hitId[nHits] = id;
            this.hitT[nHits] = hit;
            nHits++;
          }
        }
      }
      if (tMx < tMz) {
        if (tMx > L) break;
        ix += stepX;
        tMx += tDx;
      } else {
        if (tMz > L) break;
        iz += stepZ;
        tMz += tDz;
      }
    }
    let best = RAY_MAX;
    if (nHits === 0) return best;
    if (SENSOR_FLAGS.nearestRay) {
      for (let k = 0; k < nHits; k++) best = Math.min(best, this.hitT[k]! * RAY_MAX);
      return best;
    }
    // Legacy rule needs the hits in wall order. nHits is tiny, so insertion sort by id.
    const ids = this.hitId;
    const ts = this.hitT;
    for (let a = 1; a < nHits; a++) {
      const id = ids[a]!;
      const t = ts[a]!;
      let b = a - 1;
      while (b >= 0 && ids[b]! > id) {
        ids[b + 1] = ids[b]!;
        ts[b + 1] = ts[b]!;
        b--;
      }
      ids[b + 1] = id;
      ts[b + 1] = t;
    }
    for (let k = 0; k < nHits; k++) {
      const hit = ts[k]!;
      if (hit < best) best = hit * RAY_MAX;
    }
    return best;
  }

  private growHits(): void {
    const id = new Int32Array(this.hitId.length * 2);
    const t = new Float64Array(this.hitT.length * 2);
    id.set(this.hitId);
    t.set(this.hitT);
    this.hitId = id;
    this.hitT = t;
  }
}

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

export function fillObservation(
  out: Float32Array,
  base: number,
  track: TrackSpec,
  car: Vehicle,
  rays: RayField,
  hint: number,
  prevSteer: number,
  prevThrottle: number,
  timeLeft: number,
  raysOnly: boolean,
  rng: RNG | null,
  noise: number,
): number {
  rays.cast(car.px, car.pz, car.yaw, out, base + OBS.RAY0, rng, noise);
  const q = nearestSample(track, car.px, car.pz, hint);
  const cy = Math.cos(car.yaw);
  const sy = Math.sin(car.yaw);
  const forward = cy * q.sample.tx + sy * q.sample.tz;
  const right = -sy * q.sample.tx + cy * q.sample.tz;
  const head = Math.atan2(right, forward);
  out[base + OBS.V_LONG] = car.vx / V_SCALE;
  out[base + OBS.V_LAT] = car.vz / V_SCALE;
  out[base + OBS.YAW_RATE] = car.wYaw / YAW_SCALE;
  out[base + OBS.SLIP_FRONT] = car.meanSlip('front');
  out[base + OBS.SLIP_REAR] = car.meanSlip('rear');
  out[base + OBS.SLIP_RATIO_REAR] = (car.slipK[2]! + car.slipK[3]!) / 2;
  out[base + OBS.PREV_STEER] = prevSteer;
  out[base + OBS.PREV_THROTTLE] = prevThrottle;
  out[base + OBS.STEER_ACTUAL] = car.steer / Math.max(0.2, car.p.steer.maxAngle);
  const share = car.frontLoadShare();
  out[base + OBS.LOAD_FRONT] = share;
  out[base + OBS.LOAD_REAR] = 1 - share;
  const lat = raysOnly ? 0 : q.lateral / Math.max(0.2, q.sample.halfWidth);
  out[base + OBS.LAT_OFFSET] = lat;
  out[base + OBS.HEAD_SIN] = raysOnly ? 0 : Math.sin(head);
  out[base + OBS.HEAD_COS] = raysOnly ? 0 : Math.cos(head);
  const curv = raysOnly ? [0, 0, 0, 0, 0] : curvaturesAhead(track, q.sample.s, LOOKAHEADS);
  for (let i = 0; i < 5; i++) out[base + OBS.CURV0 + i] = (curv[i] ?? 0) * 4;
  out[base + OBS.TIME_LEFT] = timeLeft;
  out[base + OBS.ROLL] = car.roll;
  out[base + OBS.PITCH] = car.pitch;
  out[base + OBS.SLIP_RATIO_FRONT] = (car.slipK[0]! + car.slipK[1]!) / 2;
  out[base + OBS.SPEED] = car.speed() / V_SCALE;
  const tot = car.load[0]! + car.load[1]! + car.load[2]! + car.load[3]! + 1e-4;
  out[base + OBS.LOAD_FL] = car.load[0]! / tot;
  out[base + OBS.LOAD_FR] = car.load[1]! / tot;
  out[base + OBS.PROGRESS] = track.length > 0 ? q.sample.s / track.length : 0;
  return q.index;
}

export function wheelsOff(track: TrackSpec, car: Vehicle, hint: number): number {
  const q = nearestSample(track, car.px, car.pz, hint);
  const s = q.sample;
  let off = 0;
  const cy = Math.cos(car.yaw);
  const sy = Math.sin(car.yaw);
  const wb = car.p.wheelbase / 2;
  const tr = car.p.track / 2;
  // wheel order FL, FR, RL, RR; no per-call arrays
  for (let i = 0; i < 4; i++) {
    const lx = i < 2 ? wb : -wb;
    const lz = (i & 1) === 0 ? -tr : tr;
    const wx = car.px + cy * lx - sy * lz;
    const wz = car.pz + sy * lx + cy * lz;
    const lat = (wx - s.x) * s.nx + (wz - s.z) * s.nz;
    if (Math.abs(lat) > s.halfWidth) off++;
  }
  return off;
}
