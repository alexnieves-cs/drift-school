import * as THREE from 'three';
import type { TrackSpec } from '../../shared/types';

export function buildTrackMesh(track: TrackSpec): THREE.Group {
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
    const stripe = i % 2 === 0;
    const mu = s.mu;
    const base = mu < 0.8 ? [0.42, 0.3, 0.16] : mu < 0.98 ? [0.32, 0.22, 0.2] : [0.2, 0.22, 0.27];
    const c = curb ? (stripe ? [0.82, 0.16, 0.14] : [0.92, 0.9, 0.86]) : base;
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
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.86, metalness: 0.04 });
  group.add(new THREE.Mesh(geo, mat));
  const wgeo = new THREE.BufferGeometry();
  wgeo.setAttribute('position', new THREE.Float32BufferAttribute(wall, 3));
  wgeo.setIndex(wallIdx);
  wgeo.computeVertexNormals();
  group.add(new THREE.Mesh(wgeo, new THREE.MeshStandardMaterial({ color: 0xd5dceb, roughness: 0.7 })));
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
