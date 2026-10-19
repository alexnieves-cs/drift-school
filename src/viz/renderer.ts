import * as THREE from 'three';
import type { CameraMode, QualityPreset, Renderer, TrackSpec } from '../shared/types';
import { buildTrackMesh, trackCenter, trackSpan } from './three/track-mesh';
import { TopDownCamera } from './cameras/topdown';
import { ChaseCamera } from './cameras/chase';
import { PovCamera } from './cameras/pov';
import { PoseInterp } from './interp';

const PALETTE = [0x5b8def, 0x3dd68c, 0xff5d73, 0xc084fc, 0x22d3ee, 0xf97316, 0xeab308, 0x94a3b8];

export interface HeroState {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  speed: number;
  steer?: number;
  throttle?: number;
  slipRear?: number;
  load?: number[];
  lap?: number;
  lapTime?: number;
  bestLap?: number | null;
}

export class ThreeRenderer implements Renderer {
  readonly canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private trackGroup: THREE.Group | null = null;
  private ghosts: THREE.InstancedMesh | null = null;
  private hero = new THREE.Group();
  private trail: THREE.Line;
  private trailPts: THREE.Vector3[] = [];
  private top = new TopDownCamera();
  private chase = new ChaseCamera();
  private pov = new PovCamera();
  private mode: CameraMode = 'topdown';
  private quality: QualityPreset = 'low';
  private track: TrackSpec | null = null;
  private interp = new PoseInterp();
  private scratch = new Float32Array(64 * 7);
  private dummy = new THREE.Object3D();
  private color = new THREE.Color();
  private disposed = false;
  private webgpu = false;
  private sun: THREE.DirectionalLight;
  private lastPoses: Float32Array | null = null;
  private marks: THREE.InstancedMesh;
  private markCursor = 0;
  private pip = true;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.scene.background = new THREE.Color(0x10161c);
    this.scene.fog = new THREE.Fog(0x10161c, 30, 90);
    const hemi = new THREE.HemisphereLight(0xc5d4ee, 0x1d3a2e, 0.85);
    this.scene.add(hemi);
    this.sun = new THREE.DirectionalLight(0xfff4e0, 1.35);
    this.sun.position.set(12, 18, 8);
    this.scene.add(this.sun);
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(80, 48),
      new THREE.MeshStandardMaterial({ color: 0x163528, roughness: 1 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.02;
    this.scene.add(ground);
    this.buildHero();
    this.scene.add(this.hero);
    const trailGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    this.trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffb020 }));
    this.scene.add(this.trail);
    this.ghosts = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.34, 0.07, 0.2),
      new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.45, roughness: 0.45, metalness: 0.2 }),
      64,
    );
    this.ghosts.count = 0;
    this.ghosts.frustumCulled = false;
    this.scene.add(this.ghosts);
    this.marks = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.08, 0.005, 0.04),
      new THREE.MeshBasicMaterial({ color: 0x141414 }),
      240,
    );
    this.marks.count = 0;
    this.marks.frustumCulled = false;
    this.scene.add(this.marks);
  }

  async init(quality: QualityPreset = 'low'): Promise<void> {
    this.quality = quality;
    if (quality === 'high' && typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const mod = (await import('three/webgpu')) as unknown as {
          WebGPURenderer: new (opts: { canvas: HTMLCanvasElement; antialias: boolean }) => THREE.WebGLRenderer & { init: () => Promise<void> };
        };
        const gpu = new mod.WebGPURenderer({ canvas: this.canvas, antialias: true });
        await gpu.init();
        this.renderer = gpu;
        this.webgpu = true;
      } catch {
        this.webgpu = false;
      }
    }
    if (!this.renderer) {
      this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false, preserveDrawingBuffer: true });
    }
    this.applyQuality();
    this.resize();
  }

  get usingWebGPU(): boolean {
    return this.webgpu;
  }

  private applyQuality(): void {
    if (!this.renderer) return;
    const high = this.quality === 'high';
    this.renderer.setPixelRatio(Math.min(high ? 1.5 : 1, window.devicePixelRatio || 1));
    this.renderer.toneMapping = THREE.AgXToneMapping;
    this.renderer.toneMappingExposure = high ? 1.05 : 1.15;
    this.renderer.shadowMap.enabled = high;
    this.sun.castShadow = high;
    if (high) {
      this.sun.shadow.mapSize.set(1024, 1024);
      this.sun.shadow.camera.near = 1;
      this.sun.shadow.camera.far = 60;
    }
  }

  setQuality(q: QualityPreset): void {
    this.quality = q;
    this.applyQuality();
  }

  setCamera(m: CameraMode): void {
    this.mode = m;
  }

  get cameraMode(): CameraMode {
    return this.mode;
  }

  resize(): void {
    if (!this.renderer) return;
    const w = this.canvas.clientWidth || 800;
    const h = this.canvas.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.top.resize(w, h);
    this.chase.resize(w, h);
    this.pov.resize(w, h);
  }

  loadTrack(t: TrackSpec): void {
    if (this.trackGroup) {
      this.scene.remove(this.trackGroup);
      this.trackGroup.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        mesh.geometry?.dispose();
        const mat = mesh.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat?.dispose();
      });
    }
    this.track = t;
    this.trackGroup = buildTrackMesh(t);
    this.scene.add(this.trackGroup);
    const center = trackCenter(t);
    this.top.target.copy(center);
    this.top.setSpan(trackSpan(t));
    this.resize();
    this.trailPts = [];
  }

  updateCars(poses: Float32Array, n: number, highlight: number): void {
    this.lastPoses = poses;
    this.interp.push(poses, n);
    this.paint(n, highlight, false);
  }

  /** Render interpolated ghosts. Call once per animation frame. */
  frame(n: number, highlight: number, hero: HeroState | null, dt = 0.016): void {
    this.paint(n, highlight, true);
    if (hero) {
      this.hero.visible = true;
      this.hero.position.set(hero.x, hero.y, hero.z);
      this.hero.rotation.set(-hero.pitch, -hero.yaw, hero.roll, 'YXZ');
      this.trailPts.push(new THREE.Vector3(hero.x, 0.05, hero.z));
      if ((hero.slipRear ?? 0) > 0.12) this.dropMark(hero.x, hero.z, hero.yaw);
      if (this.trailPts.length > 80) this.trailPts.shift();
      this.trail.geometry.dispose();
      this.trail.geometry = new THREE.BufferGeometry().setFromPoints(this.trailPts.length > 1 ? this.trailPts : [new THREE.Vector3(), new THREE.Vector3()]);
      this.chase.update({ x: hero.x, y: hero.y, z: hero.z, yaw: hero.yaw, speed: hero.speed }, this.track, dt);
      this.pov.update(hero);
    }
    this.top.update();
    this.render();
  }

  private paint(n: number, highlight: number, interpolated = false): void {
    if (!this.ghosts) return;
    if (interpolated) this.interp.sample(this.scratch, n);
    const src = interpolated ? this.scratch : this.lastPoses;
    const count = Math.min(64, n);
    this.ghosts.count = Math.max(0, count - (highlight >= 0 ? 1 : 0));
    let inst = 0;
    for (let i = 0; i < count; i++) {
      const base = i * 7;
      const x = src ? src[base]! : 0;
      const y = src ? src[base + 1]! : 0;
      const z = src ? src[base + 2]! : 0;
      const qw = src ? src[base + 3]! : 1;
      const qy = src ? src[base + 5]! : 0;
      if (i === highlight) continue;
      this.dummy.position.set(x, Math.max(0.04, y), z);
      this.dummy.quaternion.set(0, -qy, 0, qw);
      this.dummy.scale.setScalar(1);
      this.dummy.updateMatrix();
      this.ghosts.setMatrixAt(inst, this.dummy.matrix);
      this.color.setHex(PALETTE[i % PALETTE.length]!);
      this.ghosts.setColorAt(inst, this.color);
      inst++;
    }
    this.ghosts.count = inst;
    this.ghosts.instanceMatrix.needsUpdate = true;
    if (this.ghosts.instanceColor) this.ghosts.instanceColor.needsUpdate = true;
  }

  /** Direct pose upload used by tests that do not need interpolation. */
  setPoses(poses: Float32Array, n: number, highlight: number): void {
    if (!this.ghosts) return;
    let inst = 0;
    const count = Math.min(64, n);
    for (let i = 0; i < count; i++) {
      if (i === highlight) {
        const base = i * 7;
        this.hero.visible = true;
        this.hero.position.set(poses[base]!, poses[base + 1]!, poses[base + 2]!);
        this.hero.quaternion.set(-(poses[base + 4] ?? 0), -(poses[base + 5] ?? 0), -(poses[base + 6] ?? 0), poses[base + 3] ?? 1);
        continue;
      }
      const base = i * 7;
      this.dummy.position.set(poses[base]!, poses[base + 1] ?? 0.05, poses[base + 2]!);
      this.dummy.quaternion.set(-(poses[base + 4] ?? 0), -(poses[base + 5] ?? 0), -(poses[base + 6] ?? 0), poses[base + 3] ?? 1);
      this.dummy.updateMatrix();
      this.ghosts.setMatrixAt(inst, this.dummy.matrix);
      this.color.setHex(PALETTE[i % PALETTE.length]!);
      this.ghosts.setColorAt(inst, this.color);
      inst++;
    }
    this.ghosts.count = inst;
    this.ghosts.instanceMatrix.needsUpdate = true;
    if (this.ghosts.instanceColor) this.ghosts.instanceColor.needsUpdate = true;
  }

  render(): void {
    if (!this.renderer || this.disposed) return;
    const cam = this.mode === 'pov' ? this.pov.camera : this.mode === 'chase' ? this.chase.camera : this.top.camera;
    const w = this.canvas.clientWidth || this.canvas.width || 800;
    const h = this.canvas.clientHeight || this.canvas.height || 600;
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, w, h);
    this.renderer.autoClear = true;
    this.renderer.render(this.scene, cam);
    if (this.pip && this.mode !== 'pov') {
      const pw = Math.round(Math.min(240, w * 0.22));
      const ph = Math.round(pw * 0.62);
      this.renderer.autoClear = false;
      this.renderer.setScissorTest(true);
      this.renderer.setViewport(w - pw - 16, 16, pw, ph);
      this.renderer.setScissor(w - pw - 16, 16, pw, ph);
      this.renderer.render(this.scene, this.pov.camera);
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, w, h);
      this.renderer.autoClear = true;
    }
  }

  private dropMark(x: number, z: number, yaw: number): void {
    this.dummy.position.set(x, 0.025, z);
    this.dummy.rotation.set(0, -yaw, 0);
    this.dummy.scale.set(1, 1, 1);
    this.dummy.updateMatrix();
    this.marks.setMatrixAt(this.markCursor % 240, this.dummy.matrix);
    this.markCursor++;
    this.marks.count = Math.min(240, this.markCursor);
    this.marks.instanceMatrix.needsUpdate = true;
  }

  activeCamera(): THREE.Camera {
    if (this.mode === 'pov') return this.pov.camera;
    if (this.mode === 'chase') return this.chase.camera;
    return this.top.camera;
  }

  chaseInsideWall(): boolean {
    return this.track ? this.chase.insideWall(this.track) : false;
  }

  dispose(): void {
    this.disposed = true;
    this.renderer?.dispose();
    this.renderer = null;
  }

  private buildHero(): void {
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.36, 0.06, 0.2),
      new THREE.MeshStandardMaterial({ color: 0xffb020, roughness: 0.35, metalness: 0.35 }),
    );
    body.position.y = 0.05;
    const cabin = new THREE.Mesh(
      new THREE.BoxGeometry(0.14, 0.05, 0.14),
      new THREE.MeshStandardMaterial({ color: 0x1a1e27, roughness: 0.2, metalness: 0.5 }),
    );
    cabin.position.set(-0.02, 0.1, 0);
    this.hero.add(body, cabin);
    const wheelGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.03, 10);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.8 });
    for (const [x, z] of [[0.1, 0.09], [0.1, -0.09], [-0.1, 0.09], [-0.1, -0.09]] as const) {
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.rotation.x = Math.PI / 2;
      w.position.set(x, 0.035, z);
      this.hero.add(w);
    }
  }
}
