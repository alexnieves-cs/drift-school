/** Shared contracts. Lead-owned. See SPEC.md §4 and sim/OBS_LAYOUT.md. */

export interface TrackSample {
  x: number;
  z: number;
  tx: number;
  tz: number;
  nx: number;
  nz: number;
  curvature: number;
  s: number;
  halfWidth: number;
  mu: number;
}

export interface TrackSpec {
  id: string;
  seed?: number;
  source: 'gen' | 'user';
  samples: TrackSample[];
  length: number;
  leftEdge: Float32Array;
  rightEdge: Float32Array;
  /** Wall polylines, each a flat [x,z,x,z,...] loop. */
  walls: Float32Array[];
  startIndex: number;
  tRef: number;
  /** Sample indices of checkpoint gates (every ~2 m). */
  gates: number[];
}

export interface TrackGenParams {
  seed: number;
  stage: number;
}

export type InvalidReason =
  | 'open'
  | 'selfIntersect'
  | 'minRadius'
  | 'tooShort'
  | 'tooLong'
  | 'edgeOverlap'
  | 'bounds'
  | 'startStraight';

export interface ValidationIssue {
  reason: InvalidReason;
  at: [number, number];
}

export interface Validation {
  ok: boolean;
  issues: ValidationIssue[];
}

export interface PacejkaLite {
  B: number;
  C: number;
  E: number;
}

export interface CarParams {
  mass: number;
  cgHeight: number;
  wheelbase: number;
  track: number;
  yawInertia: number;
  pitchInertia: number;
  rollInertia: number;
  tire: {
    front: PacejkaLite;
    rear: PacejkaLite;
    mu0: number;
    /** Extra rear grip scale. 1 = same peak μ as the front. <1 drifts. */
    rearMuScale: number;
    loadSens: number;
  };
  susp: { k: number; c: number; travel: number; arb: number };
  motor: { torque: number; maxRpm: number; tau: number };
  steer: { maxAngle: number; rate: number; offset: number };
  drive: '4wd' | 'rwd';
  /** Front torque fraction for 4WD. Ignored for RWD (all torque to the rear). */
  split: number;
}

export interface CarState {
  pos: [number, number, number];
  quat: [number, number, number, number];
  vel: [number, number, number];
  yawRate: number;
  steer: number;
  wheelOmega: Float32Array;
  wheelLoad: Float32Array;
  slipAngle: Float32Array;
  slipRatio: Float32Array;
  s: number;
  lap: number;
  lapTime: number;
}

export const OBS_DIM = 48;
export const ACT_DIM = 2;

export interface EpisodeConfig {
  stage: number;
  car: CarParams;
  noiseScale: number;
  latencyTicks: number;
  trackId: string;
  /** Zero out track-relative features (rays-only ablation). */
  raysOnly?: boolean;
  lapsTarget?: number;
  seed?: number;
}

export interface StepInfo {
  progress: number;
  lapDone: boolean;
  lapTime?: number;
  offTrack: boolean;
  wallHit: boolean;
  speed: number;
  wrongWay: boolean;
  stuck: boolean;
  rewardRaw: number;
}

export interface StepResult {
  obs: Float32Array;
  reward: number;
  done: boolean;
  truncated: boolean;
  info: StepInfo;
}

export interface VecEnv {
  readonly n: number;
  reset(tracks: TrackSpec[], cfgs: EpisodeConfig[]): Float32Array;
  step(actions: Float32Array): {
    obs: Float32Array;
    rew: Float32Array;
    done: Uint8Array;
    trunc: Uint8Array;
    infos: StepInfo[];
  };
  poses(): Float32Array;
}

export interface Policy {
  act(
    obs: Float32Array,
    n: number,
  ): { action: Float32Array; logp: Float32Array; value: Float32Array };
  getWeights(): ArrayBuffer[];
  setWeights(w: ArrayBuffer[]): void;
}

export interface GenStat {
  gen: number;
  bestLap: number | null;
  meanLap: number | null;
  /** Episodes that finished during this update. When 0, completion and crashRate are 0 and carry no information. */
  episodes: number;
  completion: number;
  crashRate: number;
  topSpeed: number;
  meanReturn: number;
  trackId: string;
  entropy?: number;
  kl?: number;
  valueLoss?: number;
  stepsPerSec?: number;
}

export type CameraMode = 'pov' | 'chase' | 'topdown';
export type QualityPreset = 'low' | 'high';

export interface Renderer {
  loadTrack(t: TrackSpec): void;
  updateCars(poses: Float32Array, n: number, highlight: number): void;
  setCamera(m: CameraMode): void;
  setQuality(q: QualityPreset): void;
  dispose(): void;
}
