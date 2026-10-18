import RAPIER from '@dimforge/rapier3d-compat';

let initPromise: Promise<void> | null = null;

export function initRapier(): Promise<void> {
  if (!initPromise) initPromise = RAPIER.init();
  return initPromise;
}

export interface ContactHit {
  hit: boolean;
  impactSpeed: number;
  nx: number;
  ny: number;
  nz: number;
}

/**
 * Chassis convex (cuboid) against extruded wall boxes.
 * Collision detection and contact normals come from Rapier; the vehicle
 * integrator applies a restitution-0.3 impulse so the custom tire model
 * stays deterministic.
 */
export class WallWorld {
  readonly world: RAPIER.World;
  readonly chassis: RAPIER.RigidBody;
  readonly chassisCol: RAPIER.Collider;
  private walls: RAPIER.RigidBody[] = [];
  private shape: RAPIER.Cuboid;
  private scratch: ContactHit = { hit: false, impactSpeed: 0, nx: 0, ny: 1, nz: 0 };

  constructor() {
    this.world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    this.world.timestep = 1 / 60;
    this.chassis = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(0, 0.05, 0),
    );
    this.chassisCol = this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(0.16, 0.045, 0.1).setRestitution(0.3).setFriction(0.4),
      this.chassis,
    );
    this.shape = new RAPIER.Cuboid(0.16, 0.045, 0.1);
  }

  clearWalls(): void {
    for (const b of this.walls) this.world.removeRigidBody(b);
    this.walls.length = 0;
  }

  setWalls(polylines: ArrayLike<number>[]): void {
    this.clearWalls();
    for (const poly of polylines) {
      for (let i = 0; i + 3 < poly.length; i += 2) {
        const x1 = poly[i]!;
        const z1 = poly[i + 1]!;
        const x2 = poly[i + 2]!;
        const z2 = poly[i + 3]!;
        const dx = x2 - x1;
        const dz = z2 - z1;
        const len = Math.hypot(dx, dz);
        if (len < 1e-3) continue;
        const yaw = Math.atan2(dx, dz);
        const body = this.world.createRigidBody(
          RAPIER.RigidBodyDesc.fixed()
            .setTranslation((x1 + x2) / 2, 0.12, (z1 + z2) / 2)
            .setRotation({ x: 0, y: Math.sin(yaw / 2), z: 0, w: Math.cos(yaw / 2) }),
        );
        this.world.createCollider(RAPIER.ColliderDesc.cuboid(0.08, 0.2, len / 2).setRestitution(0.3), body);
        this.walls.push(body);
      }
    }
    this.world.step();
  }

  /**
   * Move the chassis, step Rapier, and write a corrected pose/velocity.
   * `vel` is world-space linear velocity.
   */
  resolve(
    px: number,
    py: number,
    pz: number,
    yaw: number,
    vx: number,
    vy: number,
    vz: number,
    dt = 0.002,
  ): { px: number; py: number; pz: number; vx: number; vy: number; vz: number; hit: ContactHit } {
    const hit = this.scratch;
    hit.hit = false;
    hit.impactSpeed = 0;
    const rot = { x: 0, y: Math.sin(yaw / 2), z: 0, w: Math.cos(yaw / 2) };
    const ox = px - vx * dt;
    const oz = pz - vz * dt;
    const speed = Math.hypot(vx, vz);
    const toi = this.world.castShape(
      { x: ox, y: py, z: oz },
      rot,
      { x: vx, y: 0, z: vz },
      this.shape,
      0.04,
      Math.max(dt * 2, 0.004),
      true,
      undefined,
      undefined,
      this.chassisCol,
    );
    if (toi && speed > 0.05 && toi.time_of_impact <= dt + 1e-4) {
      const n = toi.normal1;
      let nx = n.x;
      let ny = n.y;
      let nz = n.z;
      if (vx * nx + vy * ny + vz * nz > 0) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
      }
      const approach = -(vx * nx + vy * ny + vz * nz);
      if (approach > 0.02) {
        hit.hit = true;
        hit.impactSpeed = approach;
        hit.nx = nx;
        hit.ny = ny;
        hit.nz = nz;
        const tHit = Math.max(0, toi.time_of_impact - 0.0004);
        px = ox + vx * tHit;
        pz = oz + vz * tHit;
        const e = 0.3;
        const bounce = (1 + e) * approach;
        vx += bounce * nx;
        vy += bounce * ny;
        vz += bounce * nz;
      }
    }
    this.chassis.setNextKinematicTranslation({ x: px, y: py, z: pz });
    this.chassis.setNextKinematicRotation(rot);
    return { px, py, pz, vx, vy, vz, hit };
  }

  free(): void {
    this.world.free();
  }
}
