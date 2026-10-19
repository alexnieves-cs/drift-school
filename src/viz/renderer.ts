import * as THREE from 'three';
import type { CameraMode, QualityPreset, Renderer, TrackSpec } from '../shared/types';
import { buildTrackMesh, trackCenter, trackSpan } from './three/track-mesh';
import { PALETTE, SKY_HEX, GROUND_LIT_HEX, GROUND_UNLIT_HEX, carGeometry, makeLook, paint, type LookMaterials } from './three/look';
import { TopDownCamera } from './cameras/topdown';
import { ChaseCamera } from './cameras/chase';
import { PovCamera } from './cameras/pov';
import { PoseInterp } from './interp';
import type { ReplayPose } from './replay';

export const MAX_CARS = 64;
const MARK_CAP = 240;
const TRAIL_PTS = 80;

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

export interface QualitySettings {
  /** Upper bound on devicePixelRatio. */
  maxPixelRatio: number;
  /** Upper bound on drawing-buffer pixels. The pixel ratio is lowered to fit, never the view. */
  maxPixels: number;
  /** MSAA. Fixed per WebGL context, so changing it rebuilds the context. */
  antialias: boolean;
  /** Lit look (Lambert, sun, AgX, sRGB output) or the flat unlit look; see three/look.ts. */
  lit: boolean;
  shadows: boolean;
  shadowMapSize: number;
  /** PCF filters 9 taps per lit pixel; Basic takes one tap and has harder edges. */
  shadowType: THREE.ShadowMapType;
  /** Re-render the shadow map every Nth frame. Cars move a few cm per frame, so 2 is invisible. */
  shadowEvery: number;
  exposure: number;
}

export const QUALITY: Record<QualityPreset, QualitySettings> = {
  low: {
    maxPixelRatio: 1,
    maxPixels: 1_000_000,
    antialias: false,
    lit: false,
    shadows: false,
    shadowMapSize: 0,
    shadowType: THREE.BasicShadowMap,
    shadowEvery: 0,
    exposure: 1,
  },
  high: {
    maxPixelRatio: 1.5,
    maxPixels: 2_600_000,
    antialias: true,
    lit: true,
    shadows: true,
    shadowMapSize: 1024,
    shadowType: THREE.PCFShadowMap,
    shadowEvery: 2,
    exposure: 1.05,
  },
};

/** Pixel ratio that honours both the DPR cap and the drawing-buffer budget. */
export function pixelRatioFor(cssW: number, cssH: number, dpr: number, q: QualitySettings): number {
  const area = Math.max(1, cssW * cssH);
  const budget = Math.sqrt(q.maxPixels / area);
  return Math.max(0.5, Math.min(q.maxPixelRatio, dpr || 1, budget));
}

export interface RenderTiming {
  updateMs: number;
  renderMs: number;
  calls: number;
  triangles: number;
}

export interface SceneCounts extends RenderTiming {
  geometries: number;
  textures: number;
  programs: number;
  drawingBufferWidth: number;
  drawingBufferHeight: number;
  pixelRatio: number;
  webgpu: boolean;
}

/** x, y, z, yaw of the drawn replay ghost, as the scene holds it. */
export interface ReplayDrawn {
  x: number;
  y: number;
  z: number;
  yaw: number;
}

export class ThreeRenderer implements Renderer {
  private canvas_: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private trackGroup: THREE.Group | null = null;
  private ghosts: THREE.InstancedMesh;
  private ground: THREE.Mesh;
  private look: LookMaterials;
  private unlit = false;
  private hero: THREE.Mesh;
  private replayCar: THREE.Mesh;
  private trail: THREE.Line;
  private trailPos = new Float32Array(TRAIL_PTS * 3);
  private trailLen = 0;
  private top = new TopDownCamera();
  private chase = new ChaseCamera();
  private pov = new PovCamera();
  private mode: CameraMode = 'topdown';
  private quality: QualityPreset = 'low';
  private settings: QualitySettings = QUALITY.low;
  private track: TrackSpec | null = null;
  private interp = new PoseInterp();
  private disposed = false;
  private webgpu = false;
  private sun: THREE.DirectionalLight;
  private marks: THREE.InstancedMesh;
  private markCursor = 0;
  private lastMark = { x: Infinity, z: Infinity };
  private pip = true;
  /** Picture-in-picture box in CSS px: distance from the left and bottom edges, and width. */
  private pipBox = { left: 16, bottom: 16, width: 240 };
  private frameNo = 0;
  private chaseWarm = 0;
  private cssW = 800;
  private cssH = 600;
  private pixelRatio = 1;
  private heroPrev: HeroState | null = null;
  private heroNext: HeroState | null = null;
  private heroStamp = 0;
  private heroDrawn: HeroState | null = null;
  private heroScratch: HeroState = { x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0, speed: 0 };
  private replayYaw = 0;
  private fog = new THREE.Fog(SKY_HEX, 30, 90);
  private skyColor = new THREE.Color(SKY_HEX);
  /** Timing of the most recent frame(). */
  readonly last: RenderTiming = { updateMs: 0, renderMs: 0, calls: 0, triangles: 0 };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas_ = canvas;
    this.look = makeLook(false);
    this.scene.add(new THREE.HemisphereLight(0xc5d4ee, 0x1d3a2e, 0.85));
    this.sun = new THREE.DirectionalLight(0xfff4e0, 1.35);
    this.sun.position.set(12, 18, 8);
    this.scene.add(this.sun, this.sun.target);
    this.ground = new THREE.Mesh(new THREE.CircleGeometry(80, 48), this.look.ground);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.y = -0.02;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);

    this.hero = new THREE.Mesh(carGeometry(false), this.look.hero);
    this.hero.castShadow = true;
    this.scene.add(this.hero);
    this.replayCar = new THREE.Mesh(this.hero.geometry, this.look.replay);
    this.replayCar.visible = false;
    this.scene.add(this.replayCar);

    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(this.trailPos, 3).setUsage(THREE.DynamicDrawUsage));
    trailGeo.setDrawRange(0, 0);
    this.trail = new THREE.Line(trailGeo, this.look.trail);
    this.trail.frustumCulled = false;
    this.scene.add(this.trail);

    this.ghosts = new THREE.InstancedMesh(new THREE.BoxGeometry(0.34, 0.07, 0.2), this.look.ghost, MAX_CARS);
    this.ghosts.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.paintGhosts();
    this.ghosts.count = 0;
    this.ghosts.frustumCulled = false;
    this.ghosts.castShadow = true;
    this.scene.add(this.ghosts);

    this.marks = new THREE.InstancedMesh(new THREE.BoxGeometry(0.08, 0.005, 0.04), this.look.marks, MARK_CAP);
    this.marks.count = 0;
    this.marks.frustumCulled = false;
    this.scene.add(this.marks);
    this.syncEnvironment();
  }

  get canvas(): HTMLCanvasElement {
    return this.canvas_;
  }

  async init(quality: QualityPreset = 'low', opts: { webgpu?: boolean } = {}): Promise<void> {
    this.quality = quality;
    this.settings = QUALITY[quality];
    // WebGPU is opt-in (?webgpu=1): it could not be exercised on the headless test box, and a
    // renderer that silently differs from the measured one would make the fps numbers meaningless.
    if (opts.webgpu && quality === 'high' && typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const mod = (await import('three/webgpu')) as unknown as {
          WebGPURenderer: new (o: { canvas: HTMLCanvasElement; antialias: boolean }) => THREE.WebGLRenderer & { init: () => Promise<void> };
        };
        const gpu = new mod.WebGPURenderer({ canvas: this.canvas_, antialias: true });
        await gpu.init();
        this.renderer = gpu;
        this.webgpu = true;
      } catch {
        this.webgpu = false;
      }
    }
    if (!this.renderer) this.createGl();
    this.applyQuality();
    this.resize();
  }

  get usingWebGPU(): boolean {
    return this.webgpu;
  }

  /** Underlying renderer, exposed for diagnostics and experiments only. */
  get gl(): THREE.WebGLRenderer | null {
    return this.renderer;
  }

  get sceneGraph(): THREE.Scene {
    return this.scene;
  }

  private createGl(): void {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas_,
      antialias: this.settings.antialias,
      alpha: false,
      powerPreference: 'high-performance',
      // The page never reads the canvas back, so the drawing buffer can be discarded after present.
      preserveDrawingBuffer: false,
    });
    this.renderer.info.autoReset = false;
  }

  /** A new context is the only way to change MSAA, so swap the canvas element. */
  private rebuildContext(): void {
    this.renderer?.dispose();
    this.renderer?.forceContextLoss();
    this.renderer = null;
    const next = this.canvas_.cloneNode(false) as HTMLCanvasElement;
    this.canvas_.replaceWith(next);
    this.canvas_ = next;
    this.createGl();
  }

  private applyQuality(): void {
    const r = this.renderer;
    if (!r) return;
    const s = this.settings;
    r.toneMapping = s.lit ? THREE.AgXToneMapping : THREE.NoToneMapping;
    r.toneMappingExposure = s.exposure;
    r.outputColorSpace = s.lit ? THREE.SRGBColorSpace : THREE.LinearSRGBColorSpace;
    r.shadowMap.enabled = s.shadows;
    r.shadowMap.type = s.shadowType;
    // The shadow map is refreshed by hand (see render) so the PIP pass never re-renders it.
    r.shadowMap.autoUpdate = false;
    r.shadowMap.needsUpdate = true;
    this.sun.castShadow = s.shadows;
    if (s.shadows) {
      this.sun.shadow.mapSize.set(s.shadowMapSize, s.shadowMapSize);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null;
      this.sun.shadow.bias = -0.0004;
      this.sun.shadow.normalBias = 0.02;
      this.fitShadow();
    }
    this.applyLook();
    this.scene.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (m) (Array.isArray(m) ? m : [m]).forEach((mm) => (mm.needsUpdate = true));
    });
  }

  /** Switch between the lit and unlit looks: materials, car geometry colors, track colors. */
  private applyLook(): void {
    const unlit = !this.settings.lit;
    if (unlit === this.unlit) return;
    this.unlit = unlit;
    const old = this.look;
    this.look = makeLook(unlit);
    this.ground.material = this.look.ground;
    this.ghosts.material = this.look.ghost;
    this.hero.material = this.look.hero;
    this.replayCar.material = this.look.replay;
    this.trail.material = this.look.trail;
    this.marks.material = this.look.marks;
    Object.values(old).forEach((m) => m.dispose());
    const oldGeo = this.hero.geometry;
    this.hero.geometry = carGeometry(unlit);
    this.replayCar.geometry = this.hero.geometry;
    oldGeo.dispose();
    this.paintGhosts();
    if (this.track) this.setTrackMesh(this.track);
    this.syncEnvironment();
  }

  private paintGhosts(): void {
    const unlit = this.unlit;
    for (let i = 0; i < MAX_CARS; i++) this.ghosts.setColorAt(i, paint(PALETTE[i % PALETTE.length]!, unlit));
    if (this.ghosts.instanceColor) this.ghosts.instanceColor.needsUpdate = true;
  }

  /**
   * Background, ground and fog follow the camera.
   * The top-down view is orthographic and looks straight down, so the ground is one flat color:
   * it is cleared to that color instead of drawing a full-screen mesh, which is the single most
   * expensive thing in the scene for a rasterizer that shades every pixel. Fog is a uniform wash
   * there (depth is constant), so it is dropped. Perspective views keep the ground and the fog.
   */
  private syncEnvironment(): void {
    const top = this.mode === 'topdown';
    const had = this.scene.fog !== null;
    this.scene.fog = top ? null : this.fog;
    this.skyColor = paint(SKY_HEX, this.unlit);
    this.fog.color.copy(this.skyColor);
    this.ground.visible = !top;
    const bg = top ? (this.unlit ? GROUND_UNLIT_HEX : GROUND_LIT_HEX) : SKY_HEX;
    this.scene.background = top ? paint(bg, this.unlit) : this.skyColor;
    if (had !== (this.scene.fog !== null)) {
      this.scene.traverse((o) => {
        const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
        if (m) (Array.isArray(m) ? m : [m]).forEach((mm) => (mm.needsUpdate = true));
      });
    }
  }

  setQuality(q: QualityPreset): void {
    if (q === this.quality && this.renderer) return;
    const needsContext = QUALITY[q].antialias !== this.settings.antialias && !this.webgpu;
    this.quality = q;
    this.settings = QUALITY[q];
    if (needsContext && this.renderer) this.rebuildContext();
    this.applyQuality();
    this.resize();
  }

  get preset(): QualityPreset {
    return this.quality;
  }

  setCamera(m: CameraMode): void {
    if (m === 'chase' && this.mode !== 'chase') this.chaseWarm = 40;
    this.mode = m;
    this.syncEnvironment();
  }

  get cameraMode(): CameraMode {
    return this.mode;
  }

  /** Where the POV picture-in-picture sits, in CSS px from the canvas's left and bottom edges. */
  setPip(box: { left: number; bottom: number; width: number } | null): void {
    this.pip = box !== null;
    if (box) this.pipBox = box;
  }

  resize(): void {
    if (!this.renderer) return;
    const w = this.canvas_.clientWidth || 800;
    const h = this.canvas_.clientHeight || 600;
    this.cssW = w;
    this.cssH = h;
    this.pixelRatio = pixelRatioFor(w, h, typeof window === 'undefined' ? 1 : window.devicePixelRatio, this.settings);
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setSize(w, h, false);
    this.top.resize(w, h);
    this.chase.resize(w, h);
    this.pov.resize(w, h);
  }

  private setTrackMesh(t: TrackSpec): void {
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
    this.trackGroup = buildTrackMesh(t, !this.unlit);
    this.trackGroup.children.forEach((c, i) => {
      c.receiveShadow = true;
      c.castShadow = i === 1;
    });
    this.scene.add(this.trackGroup);
  }

  loadTrack(t: TrackSpec): void {
    this.track = t;
    this.setTrackMesh(t);
    const center = trackCenter(t);
    this.top.target.copy(center);
    this.top.setSpan(trackSpan(t));
    this.fitShadow();
    this.resize();
    this.trailLen = 0;
    this.trail.geometry.setDrawRange(0, 0);
    this.markCursor = 0;
    this.marks.count = 0;
    this.lastMark.x = Infinity;
  }

  /** Aim the sun's shadow frustum at the track so the whole map is covered, not a 10 m patch. */
  private fitShadow(): void {
    if (!this.track) return;
    const c = trackCenter(this.track);
    const half = trackSpan(this.track) * 0.62;
    this.sun.target.position.copy(c);
    this.sun.position.set(c.x + half * 0.5, 40, c.z + half * 0.3);
    const cam = this.sun.shadow.camera;
    cam.left = -half;
    cam.right = half;
    cam.top = half;
    cam.bottom = -half;
    cam.near = 5;
    cam.far = 120;
    cam.updateProjectionMatrix();
    this.sun.target.updateMatrixWorld();
    if (this.renderer) this.renderer.shadowMap.needsUpdate = true;
  }

  /** Push a new 30 Hz snapshot. Cheap: copies into persistent buffers, touches no GPU state. */
  updateCars(poses: Float32Array, n: number, highlight: number): void {
    void highlight;
    this.interp.push(poses, Math.min(MAX_CARS, n));
  }

  /** Push a new hero snapshot. It is blended over the same interval as the ghost poses. */
  pushHero(h: HeroState): void {
    this.heroPrev = this.heroNext ?? h;
    this.heroNext = h;
    this.heroStamp = performance.now();
    this.noteTrail(h);
  }

  /** Replay ghost pose from a ReplayPlayer, or null to hide it. */
  setReplay(p: ReplayPose | null): void {
    if (!p) {
      this.replayCar.visible = false;
      return;
    }
    this.replayCar.visible = true;
    this.replayCar.position.set(p.x, p.y, p.z);
    this.replayCar.rotation.set(0, -p.yaw, 0);
    this.replayYaw = p.yaw;
  }

  replayDrawn(): ReplayDrawn | null {
    if (!this.replayCar.visible) return null;
    const p = this.replayCar.position;
    return { x: p.x, y: p.y, z: p.z, yaw: this.replayYaw };
  }

  /**
   * Render one animation frame. With `hero` given the hero is placed directly (physics-rate
   * callers such as the drive page); with null the pushed, interpolated hero is used.
   */
  frame(n: number, highlight: number, hero: HeroState | null, dt = 0.016): RenderTiming {
    const t0 = performance.now();
    const now = t0;
    const count = Math.min(MAX_CARS, n);
    const inst = this.ghosts.instanceMatrix;
    this.ghosts.count = this.interp.writeMatrices(inst.array as Float32Array, now, highlight, 0.04);
    if (this.ghosts.count > count) this.ghosts.count = count;
    inst.needsUpdate = this.ghosts.count > 0;

    let drawn: HeroState | null = hero;
    if (hero) {
      this.noteTrail(hero);
    } else if (this.heroNext) {
      drawn = this.blendHero(now);
    }
    this.heroDrawn = drawn;
    this.hero.visible = drawn !== null;
    if (drawn) {
      this.hero.position.set(drawn.x, drawn.y, drawn.z);
      this.hero.rotation.set(-drawn.pitch, -drawn.yaw, drawn.roll, 'YXZ');
      if (this.mode === 'chase') {
        if (this.chaseWarm > 0) {
          for (; this.chaseWarm > 0; this.chaseWarm--) this.chase.update(drawn, this.track, 1 / 60);
        }
        this.chase.update(drawn, this.track, dt);
      }
      this.pov.update(drawn);
    }
    this.top.update();
    const t1 = performance.now();
    this.render();
    const t2 = performance.now();
    const info = this.renderer?.info.render;
    this.last.updateMs = t1 - t0;
    this.last.renderMs = t2 - t1;
    this.last.calls = info?.calls ?? 0;
    this.last.triangles = info?.triangles ?? 0;
    return this.last;
  }

  private blendHero(now: number): HeroState {
    const a = this.heroPrev!;
    const b = this.heroNext!;
    const t = Math.min(1, Math.max(0, (now - this.heroStamp) / this.interp.interval));
    let dyaw = b.yaw - a.yaw;
    dyaw -= Math.round(dyaw / (2 * Math.PI)) * 2 * Math.PI;
    const jump = (b.x - a.x) ** 2 + (b.z - a.z) ** 2 > 0.75 * 0.75;
    const k = jump ? 1 : t;
    const o = Object.assign(this.heroScratch, b);
    o.x = a.x + (b.x - a.x) * k;
    o.y = a.y + (b.y - a.y) * k;
    o.z = a.z + (b.z - a.z) * k;
    o.yaw = a.yaw + dyaw * k;
    o.pitch = a.pitch + (b.pitch - a.pitch) * k;
    o.roll = a.roll + (b.roll - a.roll) * k;
    return o;
  }

  /** Trail and tire marks follow the hero's real updates, not the display refresh. */
  private noteTrail(h: HeroState): void {
    const len = this.trailLen;
    const px = this.trailPos;
    const d2 = len > 0 ? (px[(len - 1) * 3]! - h.x) ** 2 + (px[(len - 1) * 3 + 2]! - h.z) ** 2 : Infinity;
    if (d2 < 1e-4) return;
    // A reset or a move to another pool track teleports the car; start a new trail instead of a line across the map.
    if (len > 0 && d2 > 0.75 * 0.75) this.trailLen = 0;
    if (this.trailLen >= TRAIL_PTS) {
      px.copyWithin(0, 3, TRAIL_PTS * 3);
      this.trailLen = TRAIL_PTS - 1;
    }
    const o = this.trailLen * 3;
    px[o] = h.x;
    px[o + 1] = 0.05;
    px[o + 2] = h.z;
    this.trailLen++;
    const attr = this.trail.geometry.getAttribute('position') as THREE.BufferAttribute;
    attr.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, this.trailLen);
    if ((h.slipRear ?? 0) > 0.12) this.dropMark(h.x, h.z, h.yaw);
  }

  /** Direct pose upload used by tests that do not need interpolation. */
  setPoses(poses: Float32Array, n: number, highlight: number): void {
    this.interp.push(poses, Math.min(MAX_CARS, n), -1e9);
    this.ghosts.count = this.interp.writeMatrices(this.ghosts.instanceMatrix.array as Float32Array, 0, highlight, 0.04);
    this.ghosts.instanceMatrix.needsUpdate = true;
    if (highlight >= 0 && highlight < n) {
      const base = highlight * 7;
      this.hero.visible = true;
      this.hero.position.set(poses[base]!, poses[base + 1]!, poses[base + 2]!);
      this.hero.rotation.set(0, -2 * Math.atan2(poses[base + 5] ?? 0, poses[base + 3] ?? 1), 0);
    }
  }

  render(): void {
    const r = this.renderer;
    if (!r || this.disposed) return;
    const cam = this.activeCamera();
    const w = this.cssW;
    const h = this.cssH;
    const s = this.settings;
    r.info.reset();
    if (s.shadows && s.shadowEvery > 0 && this.frameNo % s.shadowEvery === 0) r.shadowMap.needsUpdate = true;
    this.frameNo++;
    r.setScissorTest(false);
    r.setViewport(0, 0, w, h);
    r.autoClear = true;
    r.render(this.scene, cam);
    if (this.pip && this.mode !== 'pov') {
      const pw = Math.round(Math.min(this.pipBox.width, w * 0.3));
      const ph = Math.round(pw * 0.62);
      const px = Math.max(0, Math.min(w - pw, this.pipBox.left));
      const py = Math.max(0, this.pipBox.bottom);
      // The POV pass needs the ground and sky that the top-down view skips; the scissored Color
      // background clears just the PIP box.
      const bg = this.scene.background;
      const fog = this.scene.fog;
      this.scene.background = this.skyColor;
      this.scene.fog = this.fog;
      const groundWas = this.ground.visible;
      this.ground.visible = true;
      r.setScissorTest(true);
      r.setViewport(px, py, pw, ph);
      r.setScissor(px, py, pw, ph);
      r.autoClear = false;
      r.render(this.scene, this.pov.camera);
      r.setScissorTest(false);
      r.setViewport(0, 0, w, h);
      r.autoClear = true;
      this.scene.background = bg;
      this.scene.fog = fog;
      this.ground.visible = groundWas;
    }
  }

  /** Counters for the last frame plus context sizes. Cheap enough to poll. */
  counts(): SceneCounts {
    const r = this.renderer;
    const size = r ? r.getDrawingBufferSize(new THREE.Vector2()) : new THREE.Vector2();
    return {
      ...this.last,
      geometries: r?.info.memory.geometries ?? 0,
      textures: r?.info.memory.textures ?? 0,
      programs: r?.info.programs?.length ?? 0,
      drawingBufferWidth: size.x,
      drawingBufferHeight: size.y,
      pixelRatio: this.pixelRatio,
      webgpu: this.webgpu,
    };
  }

  /** What the main view is showing for the hero (after interpolation). */
  heroPose(): HeroState | null {
    return this.heroDrawn;
  }

  private dropMark(x: number, z: number, yaw: number): void {
    const dx = x - this.lastMark.x;
    const dz = z - this.lastMark.z;
    if (dx * dx + dz * dz < 0.0016) return;
    this.lastMark.x = x;
    this.lastMark.z = z;
    const o = (this.markCursor % MARK_CAP) * 16;
    const c = Math.cos(-yaw);
    const s = Math.sin(-yaw);
    const a = this.marks.instanceMatrix.array as Float32Array;
    a.fill(0, o, o + 16);
    a[o] = c;
    a[o + 2] = -s;
    a[o + 5] = 1;
    a[o + 8] = s;
    a[o + 10] = c;
    a[o + 12] = x;
    a[o + 13] = 0.025;
    a[o + 14] = z;
    a[o + 15] = 1;
    this.markCursor++;
    this.marks.count = Math.min(MARK_CAP, this.markCursor);
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
}
