/**
 * Uniform spatial grid over a track's wall segments, built once per wall set and shared by
 * every car on that track (the cache is keyed on the `walls` array, which `withSurface` and
 * the other TrackSpec copies reuse by reference).
 *
 * Two cell lists, both in CSR layout and both sorted by ascending segment id, so that code
 * that walks them visits segments in exactly the order the brute-force loops did:
 *
 *  - `ray`: segments whose (slightly padded) bounding box touches the cell. Ray casting walks
 *    the cells a ray crosses.
 *  - `near`: segments within NEAR_R of the cell rectangle. Wall contact only has to look at the
 *    one cell the car is in. NEAR_R is the contact margin plus room for the car to be pushed
 *    out by earlier segments of the same step before a later segment is tested.
 */

export const GRID_CELL = 0.5;
export const WALL_MARGIN_SIM = 0.18;
/** margin + three full-penetration pushes */
export const NEAR_R = WALL_MARGIN_SIM * 4;
const RAY_PAD = 1e-3;

export interface WallGrid {
  nSeg: number;
  /** x1, z1, x2, z2 per segment, doubles (Float32 wall data widened exactly). */
  seg: Float64Array;
  ox: number;
  oz: number;
  inv: number;
  nx: number;
  nz: number;
  rayOff: Int32Array;
  rayIds: Int32Array;
  nearOff: Int32Array;
  nearIds: Int32Array;
  /** Per-segment visit stamp for de-duplicating a ray's cell walk. Scratch, single-threaded use. */
  stamp: Int32Array;
  tick: number;
}

const cache = new WeakMap<object, WallGrid>();

export function getWallGrid(walls: ArrayLike<number>[]): WallGrid {
  let g = cache.get(walls);
  if (!g) {
    g = buildWallGrid(walls);
    cache.set(walls, g);
  }
  return g;
}

function buildList(
  seg: Float64Array,
  nSeg: number,
  ox: number,
  oz: number,
  nx: number,
  nz: number,
  pad: number,
): { off: Int32Array; ids: Int32Array } {
  const cells = nx * nz;
  const off = new Int32Array(cells + 1);
  const range = (s: number): [number, number, number, number] => {
    const x1 = seg[s * 4]!;
    const z1 = seg[s * 4 + 1]!;
    const x2 = seg[s * 4 + 2]!;
    const z2 = seg[s * 4 + 3]!;
    const i0 = Math.max(0, Math.floor((Math.min(x1, x2) - pad - ox) / GRID_CELL));
    const i1 = Math.min(nx - 1, Math.floor((Math.max(x1, x2) + pad - ox) / GRID_CELL));
    const j0 = Math.max(0, Math.floor((Math.min(z1, z2) - pad - oz) / GRID_CELL));
    const j1 = Math.min(nz - 1, Math.floor((Math.max(z1, z2) + pad - oz) / GRID_CELL));
    return [i0, i1, j0, j1];
  };
  for (let pass = 0; pass < 2; pass++) {
    const cursor = pass === 0 ? null : off.slice(0, cells);
    const ids = pass === 0 ? null : new Int32Array(off[cells]!);
    for (let s = 0; s < nSeg; s++) {
      const [i0, i1, j0, j1] = range(s);
      for (let i = i0; i <= i1; i++) {
        for (let j = j0; j <= j1; j++) {
          const c = j * nx + i;
          if (pass === 0) off[c + 1]!++;
          else ids![cursor![c]!++] = s;
        }
      }
    }
    if (pass === 0) {
      for (let c = 0; c < cells; c++) off[c + 1]! += off[c]!;
    } else {
      return { off, ids: ids! };
    }
  }
  throw new Error('unreachable');
}

export function buildWallGrid(walls: ArrayLike<number>[]): WallGrid {
  let nSeg = 0;
  for (const poly of walls) nSeg += Math.max(0, Math.floor((poly.length - 2) / 2));
  const seg = new Float64Array(nSeg * 4);
  let k = 0;
  let minX = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxZ = -Infinity;
  for (const poly of walls) {
    for (let i = 0; i + 3 < poly.length; i += 2) {
      for (let q = 0; q < 4; q++) seg[k * 4 + q] = poly[i + q]!;
      minX = Math.min(minX, seg[k * 4]!, seg[k * 4 + 2]!);
      maxX = Math.max(maxX, seg[k * 4]!, seg[k * 4 + 2]!);
      minZ = Math.min(minZ, seg[k * 4 + 1]!, seg[k * 4 + 3]!);
      maxZ = Math.max(maxZ, seg[k * 4 + 1]!, seg[k * 4 + 3]!);
      k++;
    }
  }
  if (nSeg === 0) {
    minX = minZ = 0;
    maxX = maxZ = 0;
  }
  const ox = minX - NEAR_R - GRID_CELL;
  const oz = minZ - NEAR_R - GRID_CELL;
  const nx = Math.ceil((maxX + NEAR_R + GRID_CELL - ox) / GRID_CELL) + 1;
  const nz = Math.ceil((maxZ + NEAR_R + GRID_CELL - oz) / GRID_CELL) + 1;
  const ray = buildList(seg, nSeg, ox, oz, nx, nz, RAY_PAD);
  const near = buildList(seg, nSeg, ox, oz, nx, nz, NEAR_R);
  return { nSeg, seg, ox, oz, inv: 1 / GRID_CELL, nx, nz, rayOff: ray.off, rayIds: ray.ids, nearOff: near.off, nearIds: near.ids, stamp: new Int32Array(nSeg), tick: 0 };
}

export interface WallBody {
  px: number;
  pz: number;
  yaw: number;
  vx: number;
  vz: number;
}

export interface WallResult {
  /** Any segment within the margin this step. */
  contact: boolean;
  /** Impact speed above the 0.4 m/s threshold. */
  hit: boolean;
  impact: number;
}

const MARGIN = WALL_MARGIN_SIM;
const MARGIN_TEST = MARGIN * MARGIN * 1.01;

/** Scratch for resolveWallContact (single-threaded, not re-entrant). */
const st = { contact: false, impact: 0, px0: 0, pz0: 0, maxD2: 0 };

/** One segment of the push-out loop. Mutates the body and `st`. */
function pushFromSegment(car: WallBody, seg: Float64Array, k: number): void {
  const x1 = seg[k]!;
  const z1 = seg[k + 1]!;
  const dx = seg[k + 2]! - x1;
  const dz = seg[k + 3]! - z1;
  const len2 = dx * dx + dz * dz || 1;
  let t = ((car.px - x1) * dx + (car.pz - z1) * dz) / len2;
  t = Math.max(0, Math.min(1, t));
  const ox = car.px - (x1 + dx * t);
  const oz = car.pz - (z1 + dz * t);
  // cheap reject; the exact hypot test below decides
  if (ox * ox + oz * oz > MARGIN_TEST) return;
  const dist = Math.hypot(ox, oz);
  if (dist < MARGIN) {
    st.contact = true;
    const nx = ox / (dist || 1);
    const nz = oz / (dist || 1);
    car.px += nx * (MARGIN - dist);
    car.pz += nz * (MARGIN - dist);
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
      st.impact = Math.max(st.impact, -vn);
    }
    const mx = car.px - st.px0;
    const mz = car.pz - st.pz0;
    const d2 = mx * mx + mz * mz;
    if (d2 > st.maxD2) st.maxD2 = d2;
  }
}

/**
 * Polyline wall resolution: push the body out of every segment closer than MARGIN and reflect
 * the normal velocity with restitution 0.3. Segments are visited in ascending id (the original
 * wall order) and each push moves the body before the next test, exactly like the loop over all
 * walls this replaces; only segments in the body's grid cell are visited.
 *
 * Exactness: the cell list holds every segment within NEAR_R of the body's starting position.
 * A skipped segment could only matter if the body got within MARGIN of it, i.e. moved more than
 * NEAR_R - MARGIN. If a chain of pushes ever moves it that far (dense swallow-tail geometry at a
 * hairpin), the state is restored and the full brute-force loop runs instead.
 */
export function resolveWallContact(car: WallBody, g: WallGrid, out: WallResult): void {
  st.contact = false;
  st.impact = 0;
  const px0 = car.px;
  const pz0 = car.pz;
  const vx0 = car.vx;
  const vz0 = car.vz;
  st.px0 = px0;
  st.pz0 = pz0;
  st.maxD2 = 0;
  const ci = Math.floor((px0 - g.ox) * g.inv);
  const cj = Math.floor((pz0 - g.oz) * g.inv);
  if (ci >= 0 && ci < g.nx && cj >= 0 && cj < g.nz) {
    const cell = cj * g.nx + ci;
    const seg = g.seg;
    for (let p = g.nearOff[cell]!, e = g.nearOff[cell + 1]!; p < e; p++) pushFromSegment(car, seg, g.nearIds[p]! * 4);
    const reach = NEAR_R - MARGIN;
    if (st.maxD2 >= reach * reach) {
      car.px = px0;
      car.pz = pz0;
      car.vx = vx0;
      car.vz = vz0;
      st.contact = false;
      st.impact = 0;
      for (let id = 0; id < g.nSeg; id++) pushFromSegment(car, seg, id * 4);
    }
  }
  // else: outside the padded grid, so farther than NEAR_R from every wall and nothing can touch.
  out.contact = st.contact;
  out.impact = st.impact;
  out.hit = st.impact > 0.4;
}
