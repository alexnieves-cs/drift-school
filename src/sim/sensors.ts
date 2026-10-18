import type { TrackSpec } from '../shared/types';
import { LOOKAHEADS, OBS, RAY_ARC, RAY_COUNT, RAY_MAX, V_SCALE, YAW_SCALE } from '../shared/constants';
import type { RNG } from '../shared/rng';
import { randn } from '../shared/rng';
import { curvaturesAhead, nearestSample } from '../track/sample';
import type { Vehicle } from './car/vehicle';

interface Seg {
  x1: number;
  z1: number;
  x2: number;
  z2: number;
}

const CELL = 4;

export class RayField {
  segs: Seg[] = [];
  /** Wall segments by 4 m grid cell, so a ray cast only tests walls near the car. */
  private grid = new Map<number, number[]>();
  private stamp = new Int32Array(0);
  private stampId = 0;
  private near = new Int32Array(0);
  nearCount = 0;

  setTrack(track: TrackSpec): void {
    this.segs.length = 0;
    this.grid.clear();
    for (const wall of track.walls) {
      for (let i = 0; i + 3 < wall.length; i += 2) {
        const seg = { x1: wall[i]!, z1: wall[i + 1]!, x2: wall[i + 2]!, z2: wall[i + 3]! };
        const id = this.segs.length;
        this.segs.push(seg);
        const x0 = Math.floor(Math.min(seg.x1, seg.x2) / CELL);
        const x1 = Math.floor(Math.max(seg.x1, seg.x2) / CELL);
        const z0 = Math.floor(Math.min(seg.z1, seg.z2) / CELL);
        const z1 = Math.floor(Math.max(seg.z1, seg.z2) / CELL);
        for (let ix = x0; ix <= x1; ix++) {
          for (let iz = z0; iz <= z1; iz++) {
            const k = cellKey(ix, iz);
            let arr = this.grid.get(k);
            if (!arr) this.grid.set(k, (arr = []));
            arr.push(id);
          }
        }
      }
    }
    this.stamp = new Int32Array(this.segs.length);
    this.stampId = 0;
    this.near = new Int32Array(this.segs.length);
  }

  /** Fill `near` with the segments whose grid cells overlap a square of half-size `r` around (x, z). */
  gather(x: number, z: number, r: number): Int32Array {
    this.stampId++;
    let count = 0;
    const x0 = Math.floor((x - r) / CELL);
    const x1 = Math.floor((x + r) / CELL);
    const z0 = Math.floor((z - r) / CELL);
    const z1 = Math.floor((z + r) / CELL);
    for (let ix = x0; ix <= x1; ix++) {
      for (let iz = z0; iz <= z1; iz++) {
        const arr = this.grid.get(cellKey(ix, iz));
        if (!arr) continue;
        for (let k = 0; k < arr.length; k++) {
          const id = arr[k]!;
          if (this.stamp[id] === this.stampId) continue;
          this.stamp[id] = this.stampId;
          this.near[count++] = id;
        }
      }
    }
    this.nearCount = count;
    return this.near;
  }

  cast(x: number, z: number, yaw: number, out: Float32Array, offset: number, rng: RNG | null, noise: number): void {
    const near = this.gather(x, z, RAY_MAX);
    const n = this.nearCount;
    for (let r = 0; r < RAY_COUNT; r++) {
      const ang = yaw + -RAY_ARC / 2 + (r / (RAY_COUNT - 1)) * RAY_ARC;
      const dx = Math.cos(ang);
      const dz = Math.sin(ang);
      let bestT = 1;
      const x2 = x + dx * RAY_MAX;
      const z2 = z + dz * RAY_MAX;
      for (let i = 0; i < n; i++) {
        const s = this.segs[near[i]!]!;
        const hit = segHit(x, z, x2, z2, s.x1, s.z1, s.x2, s.z2);
        if (hit >= 0 && hit < bestT) bestT = hit;
      }
      let best = bestT * RAY_MAX;
      if (rng && noise > 0) {
        if (rng.next() < 0.01 * Math.max(noise, 0.5)) best = RAY_MAX;
        else best = Math.min(RAY_MAX, Math.max(0, best + randn(rng) * 0.01 * noise));
      }
      out[offset + r] = best / RAY_MAX;
    }
  }
}

function cellKey(ix: number, iz: number): number {
  return (ix + 4096) * 8192 + (iz + 4096);
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
  const locals: [number, number][] = [
    [wb, -tr],
    [wb, tr],
    [-wb, -tr],
    [-wb, tr],
  ];
  for (const [lx, lz] of locals) {
    const wx = car.px + cy * lx - sy * lz;
    const wz = car.pz + sy * lx + cy * lz;
    const lat = (wx - s.x) * s.nx + (wz - s.z) * s.nz;
    if (Math.abs(lat) > s.halfWidth) off++;
  }
  return off;
}
