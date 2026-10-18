import * as THREE from 'three';

/** Bumper camera, 90° FOV, with a little suspension bob. */
export class PovCamera {
  readonly camera = new THREE.PerspectiveCamera(90, 1, 0.03, 200);
  shake = 0;

  resize(w: number, h: number): void {
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  update(hero: { x: number; y: number; z: number; yaw: number; pitch: number; roll: number; speed: number }): void {
    const fx = Math.cos(hero.yaw);
    const fz = Math.sin(hero.yaw);
    const bob = Math.sin(performance.now() * 0.02) * 0.004 * Math.min(1, hero.speed / 4) + hero.pitch * 0.02;
    this.camera.position.set(hero.x + fx * 0.12, hero.y + 0.08 + bob, hero.z + fz * 0.12);
    this.camera.up.set(Math.sin(hero.roll) * 0.2, 1, 0);
    this.camera.lookAt(hero.x + fx * 3, hero.y + 0.05 - hero.pitch * 0.2, hero.z + fz * 3);
  }
}
