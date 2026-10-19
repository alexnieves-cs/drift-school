import * as THREE from 'three';

export class TopDownCamera {
  readonly camera: THREE.OrthographicCamera;
  target = new THREE.Vector3();
  zoom = 1;
  panX = 0;
  panZ = 0;
  private span = 40;

  constructor() {
    this.camera = new THREE.OrthographicCamera(-20, 20, 20, -20, 0.1, 200);
    this.camera.position.set(0, 40, 0);
    this.camera.up.set(0, 0, -1);
    this.camera.lookAt(0, 0, 0);
  }

  setSpan(span: number): void {
    this.span = span;
  }

  resize(w: number, h: number): void {
    const aspect = w / Math.max(1, h);
    const half = (this.span * 0.55) / this.zoom;
    this.camera.left = -half * aspect;
    this.camera.right = half * aspect;
    this.camera.top = half;
    this.camera.bottom = -half;
    this.camera.updateProjectionMatrix();
  }

  update(): void {
    this.camera.position.set(this.target.x + this.panX, 50, this.target.z + this.panZ);
    this.camera.lookAt(this.target.x + this.panX, 0, this.target.z + this.panZ);
  }
}
