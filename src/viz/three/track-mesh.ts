import * as THREE from 'three';
import type { TrackSpec } from '../../shared/types';

/** Vertex color for a surface sample. Lit colors are linear floats, unlit ones are display values. */
function surfaceColor(mu: number, curb: boolean, stripe: boolean, unlit: boolean): [number, number, number] {
  if (unlit) {
    const hex = curb ? (stripe ? 0xb8433a : 0xe6e2da) : mu < 0.8 ? 0x7a6347 : mu < 0.98 ? 0x6a5a55 : 0x5a5f6b;
    const c = new THREE.Color().setHex(hex, THREE.LinearSRGBColorSpace);
    return [c.r, c.g, c.b];
  }
  if (curb) return stripe ? [0.82, 0.16, 0.14] : [0.92, 0.9, 0.86];
  return mu < 0.8 ? [0.42, 0.3, 0.16] : mu < 0.98 ? [0.32, 0.22, 0.2] : [0.2, 0.22, 0.27];
}

/** Surface and wall materials. `lit` uses Lambert shading, otherwise flat unlit colors (Low preset). */
export function trackMaterials(lit: boolean): { surface: THREE.Material; wall: THREE.Material } {
  if (lit) {
    return {
      surface: new THREE.MeshLambertMaterial({ vertexColors: true }),
      wall: new THREE.MeshLambertMaterial({ color: 0xd5dceb }),
    };
  }
  return {
    surface: new THREE.MeshBasicMaterial({ vertexColors: true }),
    wall: new THREE.MeshBasicMaterial({ color: new THREE.Color().setHex(0xc4cbdb, THREE.LinearSRGBColorSpace) }),
  };
}

export function buildTrackMesh(track: TrackSpec, lit = true): THREE.Group {
  const group = new THREE.Group();
  const n = track.samples.length;
  if (n < 3) return group;
  const pos: number[] = [];
  const col: number[] = [];
  const idx: number[] = [];
  const wall: number[] = [];
  const wallIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    const s = track.samples[i]!;
    const hw = s.halfWidth;
    pos.push(s.x + s.nx * hw, 0.02, s.z + s.nz * hw);
    pos.push(s.x - s.nx * hw, 0.02, s.z - s.nz * hw);
    const curb = Math.abs(s.curvature) > 0.18;
    // Alternate every half metre of arc length. Samples are 0.1 m apart, so alternating per sample
    // painted a 0.2 m hatch that aliased into moire at any zoom.
    const stripe = Math.floor(s.s / 0.5) % 2 === 0;
    const mu = s.mu;
    const c = surfaceColor(mu, curb, stripe, !lit);
    col.push(c[0]!, c[1]!, c[2]!, c[0]!, c[1]!, c[2]!);
    const lift = 0.18;
    wall.push(s.x + s.nx * (hw + 0.12), 0.02, s.z + s.nz * (hw + 0.12));
    wall.push(s.x + s.nx * (hw + 0.12), lift, s.z + s.nz * (hw + 0.12));
    wall.push(s.x - s.nx * (hw + 0.12), 0.02, s.z - s.nz * (hw + 0.12));
    wall.push(s.x - s.nx * (hw + 0.12), lift, s.z - s.nz * (hw + 0.12));
  }
  for (let i = 0; i < n; i++) {
    const a = i * 2;
    const b = a + 1;
    const c = ((i + 1) % n) * 2;
    const d = c + 1;
    idx.push(a, c, b, b, c, d);
    const w = i * 4;
    const nw = ((i + 1) % n) * 4;
    wallIdx.push(w, nw, w + 1, w + 1, nw, nw + 1);
    wallIdx.push(w + 2, w + 3, nw + 2, w + 3, nw + 3, nw + 2);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  const mats = trackMaterials(lit);
  group.add(new THREE.Mesh(geo, mats.surface));
  const wgeo = new THREE.BufferGeometry();
  wgeo.setAttribute('position', new THREE.Float32BufferAttribute(wall, 3));
  wgeo.setIndex(wallIdx);
  wgeo.computeVertexNormals();
  group.add(new THREE.Mesh(wgeo, mats.wall));
  return group;
}

export function trackCenter(track: TrackSpec): THREE.Vector3 {
  let x = 0;
  let z = 0;
  for (const s of track.samples) {
    x += s.x;
    z += s.z;
  }
  const n = Math.max(1, track.samples.length);
  return new THREE.Vector3(x / n, 0, z / n);
}

export function trackSpan(track: TrackSpec): number {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const s of track.samples) {
    minX = Math.min(minX, s.x);
    maxX = Math.max(maxX, s.x);
    minZ = Math.min(minZ, s.z);
    maxZ = Math.max(maxZ, s.z);
  }
  return Math.max(8, Math.max(maxX - minX, maxZ - minZ) + 8);
}
