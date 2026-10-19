import * as THREE from 'three';
import type { TrackSpec } from '../../shared/types';

/** Spring-damped chase cam. The boom shortens when a wall ray hits. */
export class ChaseCamera {
  readonly camera = new THREE.PerspectiveCamera(55, 1, 0.05, 200);
  private pos = new THREE.Vector3(0, 2, -4);
  private look = new THREE.Vector3();
  private vel = new THREE.Vector3();

  resize(w: number, h: number): void {
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  update(hero: { x: number; y: number; z: number; yaw: number; speed: number }, track: TrackSpec | null, dt: number): void {
    const back = 1.6;
    const height = 0.7;
    const fx = Math.cos(hero.yaw);
    const fz = Math.sin(hero.yaw);
    let boom = back;
    if (track) {
      const hit = rayWalls(track, hero.x, hero.z, -fx, -fz, back + 0.3);
      if (hit > 0) boom = Math.max(0.45, hit - 0.25);
    }
    const desired = new THREE.Vector3(hero.x - fx * boom, hero.y + height, hero.z - fz * boom);
    const k = 8;
    const c = 6;
    const acc = desired.clone().sub(this.pos).multiplyScalar(k).sub(this.vel.clone().multiplyScalar(c));
    this.vel.addScaledVector(acc, dt);
    this.pos.addScaledVector(this.vel, dt);
    const lookAhead = Math.min(2.2, 0.4 + hero.speed * 0.15);
    this.look.set(hero.x + fx * lookAhead, hero.y + 0.15, hero.z + fz * lookAhead);
    this.camera.position.copy(this.pos);
    this.camera.lookAt(this.look);
    if (track) {
      for (let i = 0; i < 8 && this.insideWall(track); i++) {
        this.pos.x += (hero.x - this.pos.x) * 0.4;
        this.pos.z += (hero.z - this.pos.z) * 0.4;
        this.camera.position.copy(this.pos);
      }
    }
  }

  /** True when the camera sits inside the wall margin. */
  insideWall(track: TrackSpec): boolean {
    return pointInWall(track, this.camera.position.x, this.camera.position.z);
  }
}

function rayWalls(track: TrackSpec, x: number, z: number, dx: number, dz: number, max: number): number {
  const len = Math.hypot(dx, dz) || 1;
  dx /= len;
  dz /= len;
  let best = max;
  for (const wall of track.walls) {
    for (let i = 0; i + 3 < wall.length; i += 2) {
      const t = segT(x, z, x + dx * max, z + dz * max, wall[i]!, wall[i + 1]!, wall[i + 2]!, wall[i + 3]!);
      if (t >= 0 && t < best) best = t * max;
    }
  }
  return best;
}

function segT(ax: number, az: number, bx: number, bz: number, cx: number, cz: number, dx: number, dz: number): number {
  const rx = bx - ax;
  const rz = bz - az;
  const sx = dx - cx;
  const sz = dz - cz;
  const den = rx * sz - rz * sx;
  if (Math.abs(den) < 1e-8) return -1;
  const t = ((cx - ax) * sz - (cz - az) * sx) / den;
  const u = ((cx - ax) * rz - (cz - az) * rx) / den;
  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) return t;
  return -1;
}

function pointInWall(track: TrackSpec, x: number, z: number): boolean {
  for (const wall of track.walls) {
    for (let i = 0; i + 3 < wall.length; i += 2) {
      const x1 = wall[i]!;
      const z1 = wall[i + 1]!;
      const x2 = wall[i + 2]!;
      const z2 = wall[i + 3]!;
      const vx = x2 - x1;
      const vz = z2 - z1;
      const l2 = vx * vx + vz * vz || 1;
      let t = ((x - x1) * vx + (z - z1) * vz) / l2;
      t = Math.max(0, Math.min(1, t));
      const d = Math.hypot(x - (x1 + vx * t), z - (z1 + vz * t));
      if (d < 0.08) return true;
    }
  }
  return false;
}
