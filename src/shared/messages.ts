/**
 * Worker wire protocol, version 2. Lead-owned contract (CONTRACT_REQUESTS.md).
 *
 * Three channels, one discriminated union per direction:
 *
 *   page -> learner    LearnerIn     page <- learner    LearnerOut
 *   page -> env        EnvIn         page <- env        EnvOut       (standalone env worker)
 *   learner -> farm    FarmIn        learner <- farm    FarmOut      (SharedArrayBuffer env farm)
 *
 * Every message carries `v: PROTOCOL_VERSION`, except `ping` and `pong`, which stay unversioned so
 * a page can probe any worker. Build messages with `stamp()` and check what arrives with the
 * matching `parse*` function: it returns `{ ok: true, msg }` with a typed message, or
 * `{ ok: false, error }` naming the first bad field. Unknown types, missing fields, wrong types,
 * NaN, out-of-range sizes and a version mismatch are all rejected. Version 1 was the untyped ad hoc
 * messages of v1.0.0, which had no `v` field.
 */
import type { EpisodeConfig, GenStat, TrackSpec } from './types';

export const PROTOCOL_VERSION = 2 as const;
export type ProtocolVersion = typeof PROTOCOL_VERSION;

export type DRLevelName = 'off' | 'low' | 'high';

/** Snapshot of one car for the HUD and telemetry (`VecEnv.snapshot`). */
export interface HeroSnap {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  speed: number;
  steer: number;
  throttle: number;
  slip: number[];
  load: number[];
  lap: number;
  lapTime: number;
  s: number;
  yawRate: number;
  vx: number;
  vz: number;
  surfaceMu: number;
}

// ---------------------------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------------------------

export type Ping = { type: 'ping'; t: number };
export type Pong = { type: 'pong'; t: number };

/** Page to learner worker. */
export type LearnerIn =
  | Ping
  | { type: 'init'; n: number; seeds: number[]; stage: number }
  | { type: 'pause'; paused: boolean }
  | { type: 'baseline'; on: boolean }
  | { type: 'setStage'; stage: number }
  | { type: 'setDR'; grip: number; mass: number; level: DRLevelName }
  | { type: 'replayReq'; gen: number }
  | { type: 'setTrackPool'; points: number[]; halfWidth: number; mu: number; fineTune?: boolean }
  | { type: 'checkpointSave'; name: string }
  | { type: 'checkpointLoad'; name: string }
  | { type: 'ghostAck' };

/** The adaptation panel's payload: a FineTuneReport (src/rl/finetune.ts) spread into the message. */
export type AdaptMsg = { type: 'adapt'; phase: 'idle' | 'drain' | 'zeroShot' | 'tune' | 'done'; envTotal: number; steps: number } & Record<
  string,
  unknown
>;

/** Learner worker to page. */
export type LearnerOut =
  | Pong
  | { type: 'status'; text: string }
  | { type: 'ready'; hash: string; trackId: string }
  | {
      type: 'ghosts';
      poses: Float32Array;
      n: number;
      gen: number;
      trackId: string;
      ids: string[];
      hero: HeroSnap | null;
      bestLap: number | null;
    }
  | { type: 'metrics'; stat: GenStat; tape: number[] }
  | { type: 'replayRes'; gen: number; poses: number[]; n: number }
  | { type: 'swapStart'; trackId: string; hashBefore: string; ms: number }
  | AdaptMsg
  | {
      type: 'trackAck';
      trackId: string;
      envCount: number;
      envTotal: number;
      latencyMs: number | null;
      hash: string | null;
      hashBefore: string;
    }
  | { type: 'checkpointSaved'; name: string }
  | { type: 'error'; message: string };

/** Page to the standalone env worker (src/workers/env.worker.ts). */
export type EnvIn =
  | Ping
  | { type: 'init'; n: number; seeds: number[]; stage: number }
  | { type: 'step'; actions?: Float32Array }
  | { type: 'setTrackPool'; points: number[]; halfWidth: number; mu: number };

export type EnvOut =
  | Pong
  | { type: 'ready'; trackId: string }
  | { type: 'ghosts'; poses: Float32Array; n: number; gen: number; trackId: string; ids: string[] }
  | { type: 'trackAck'; trackId: string; envCount: number }
  | { type: 'error'; message: string };

/**
 * Learner to one env-farm worker. `farmInit` hands over the shared buffer once; after that each
 * step is signalled through control words in shared memory, not by message (src/workers/envFarmLayout.ts).
 * Observations, rewards, flags, poses and finished-episode records also travel through that buffer.
 */
export type FarmIn =
  | Ping
  | { type: 'farmInit'; shard: number; lo: number; hi: number; total: number; actDim: number; buffer: SharedArrayBuffer }
  | { type: 'farmReset'; epoch: number; tracks: TrackSpec[]; cfgs: EpisodeConfig[] }
  | { type: 'farmStop' };

export type FarmOut =
  | Pong
  | { type: 'farmReady'; shard: number }
  | { type: 'farmResetDone'; shard: number; epoch: number }
  | { type: 'farmStopped'; shard: number }
  | { type: 'error'; message: string; shard?: number };

export type Stamped<T> = T extends Ping | Pong ? T : T & { v: ProtocolVersion };

/** Adds the protocol version. Ping and pong pass through unchanged. */
export function stamp<T extends { type: string }>(msg: T): Stamped<T> {
  if (msg.type === 'ping' || msg.type === 'pong') return msg as unknown as Stamped<T>;
  return { ...msg, v: PROTOCOL_VERSION } as unknown as Stamped<T>;
}

export type Parsed<T> = { ok: true; msg: T } | { ok: false; error: string };

// ---------------------------------------------------------------------------------------------
// Runtime validation
// ---------------------------------------------------------------------------------------------

type Check = (v: unknown, path: string) => string | null;

const typeName = (v: unknown): string => {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  if (ArrayBuffer.isView(v)) return v.constructor.name;
  if (typeof SharedArrayBuffer !== 'undefined' && v instanceof SharedArrayBuffer) return 'SharedArrayBuffer';
  if (v instanceof ArrayBuffer) return 'ArrayBuffer';
  if (typeof v === 'number' && !Number.isFinite(v)) return String(v);
  return typeof v;
};
const fail = (path: string, want: string, v: unknown) => `${path}: expected ${want}, got ${typeName(v)}`;

const num: Check = (v, p) => (typeof v === 'number' && Number.isFinite(v) ? null : fail(p, 'a finite number', v));
const int: Check = (v, p) => (typeof v === 'number' && Number.isInteger(v) ? null : fail(p, 'an integer', v));
const nonNegInt: Check = (v, p) => int(v, p) ?? ((v as number) >= 0 ? null : `${p}: expected >= 0, got ${String(v)}`);
const bool: Check = (v, p) => (typeof v === 'boolean' ? null : fail(p, 'a boolean', v));
const str: Check = (v, p) => (typeof v === 'string' ? null : fail(p, 'a string', v));
const f32: Check = (v, p) => (v instanceof Float32Array ? null : fail(p, 'a Float32Array', v));
const sab: Check = (v, p) =>
  typeof SharedArrayBuffer !== 'undefined' && v instanceof SharedArrayBuffer ? null : fail(p, 'a SharedArrayBuffer', v);
const oneOf =
  (...values: string[]): Check =>
  (v, p) =>
    typeof v === 'string' && values.includes(v) ? null : `${p}: expected one of ${values.join(', ')}, got ${JSON.stringify(v)}`;
const nullable =
  (c: Check): Check =>
  (v, p) =>
    v === null ? null : c(v, p);
const arrayOf =
  (c: Check, max = Infinity): Check =>
  (v, p) => {
    if (!Array.isArray(v)) return fail(p, 'an array', v);
    if (v.length > max) return `${p}: expected at most ${max} items, got ${v.length}`;
    for (let i = 0; i < v.length; i++) {
      const e = c(v[i], `${p}[${i}]`);
      if (e) return e;
    }
    return null;
  };

/** Field spec. A key ending in `?` is optional (absent or undefined is fine). */
type Spec = Record<string, Check>;

const obj =
  (spec: Spec): Check =>
  (v, p) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v) || ArrayBuffer.isView(v)) return fail(p, 'an object', v);
    const rec = v as Record<string, unknown>;
    for (const [rawKey, check] of Object.entries(spec)) {
      const optional = rawKey.endsWith('?');
      const key = optional ? rawKey.slice(0, -1) : rawKey;
      const value = rec[key];
      if (value === undefined) {
        if (optional) continue;
        return `${p}.${key}: missing`;
      }
      const e = check(value, `${p}.${key}`);
      if (e) return e;
    }
    return null;
  };

const hero = obj({
  x: num,
  y: num,
  z: num,
  yaw: num,
  pitch: num,
  roll: num,
  speed: num,
  steer: num,
  throttle: num,
  slip: arrayOf(num, 4),
  load: arrayOf(num, 4),
  lap: int,
  lapTime: num,
  s: num,
  yawRate: num,
  vx: num,
  vz: num,
  surfaceMu: num,
});

const genStat = obj({
  gen: int,
  bestLap: nullable(num),
  meanLap: nullable(num),
  episodes: nonNegInt,
  completion: num,
  crashRate: num,
  topSpeed: num,
  meanReturn: num,
  trackId: str,
  'entropy?': num,
  'kl?': num,
  'valueLoss?': num,
  'stepsPerSec?': num,
});

const pacejka = obj({ B: num, C: num, E: num });
const carParams = obj({
  mass: num,
  cgHeight: num,
  wheelbase: num,
  track: num,
  yawInertia: num,
  pitchInertia: num,
  rollInertia: num,
  tire: obj({ front: pacejka, rear: pacejka, mu0: num, rearMuScale: num, loadSens: num }),
  susp: obj({ k: num, c: num, travel: num, arb: num }),
  motor: obj({ torque: num, maxRpm: num, tau: num }),
  steer: obj({ maxAngle: num, rate: num, offset: num }),
  drive: oneOf('4wd', 'rwd'),
  split: num,
});

const episodeConfig = obj({
  stage: int,
  car: carParams,
  noiseScale: num,
  latencyTicks: nonNegInt,
  trackId: str,
  'raysOnly?': bool,
  'lapsTarget?': int,
  'seed?': num,
  'drift?': obj({ 'handbrake?': bool, 'slipWeight?': num, 'slipTargetDeg?': num }),
});

/** Structural check of a TrackSpec: every field the env reads, with sample 0 spot-checked. */
const trackSpec: Check = (v, p) => {
  const e = obj({
    id: str,
    'seed?': num,
    source: oneOf('gen', 'user'),
    samples: (s, q) => (Array.isArray(s) && s.length >= 3 ? null : fail(q, 'an array of at least 3 samples', s)),
    length: num,
    leftEdge: f32,
    rightEdge: f32,
    walls: arrayOf(f32),
    startIndex: nonNegInt,
    tRef: num,
    gates: arrayOf(int),
  })(v, p);
  if (e) return e;
  const t = v as TrackSpec;
  const s0 = obj({ x: num, z: num, tx: num, tz: num, nx: num, nz: num, curvature: num, s: num, halfWidth: num, mu: num })(
    t.samples[0],
    `${p}.samples[0]`,
  );
  if (s0) return s0;
  return t.startIndex < t.samples.length ? null : `${p}.startIndex: ${t.startIndex} is past the last sample`;
};

/** Upper bounds that keep a bad message from allocating without limit. */
export const LIMITS = { learnerEnvs: 64, envs: 1024, seeds: 256, points: 20_000 } as const;

const LEARNER_IN: Record<LearnerIn['type'], Spec> = {
  ping: { t: num },
  init: { n: int, seeds: arrayOf(int, LIMITS.seeds), stage: int },
  pause: { paused: bool },
  baseline: { on: bool },
  setStage: { stage: int },
  setDR: { grip: num, mass: num, level: oneOf('off', 'low', 'high') },
  replayReq: { gen: nonNegInt },
  setTrackPool: { points: arrayOf(num, LIMITS.points), halfWidth: num, mu: num, 'fineTune?': bool },
  checkpointSave: { name: str },
  checkpointLoad: { name: str },
  ghostAck: {},
};

const LEARNER_OUT: Record<LearnerOut['type'], Spec> = {
  pong: { t: num },
  status: { text: str },
  ready: { hash: str, trackId: str },
  ghosts: {
    poses: f32,
    n: nonNegInt,
    gen: nonNegInt,
    trackId: str,
    ids: arrayOf(str, LIMITS.envs),
    hero: nullable(hero),
    bestLap: nullable(num),
  },
  metrics: { stat: genStat, tape: arrayOf(num) },
  replayRes: { gen: nonNegInt, poses: arrayOf(num), n: nonNegInt },
  swapStart: { trackId: str, hashBefore: str, ms: num },
  adapt: { phase: oneOf('idle', 'drain', 'zeroShot', 'tune', 'done'), envTotal: nonNegInt, steps: nonNegInt },
  trackAck: {
    trackId: str,
    envCount: nonNegInt,
    envTotal: nonNegInt,
    latencyMs: nullable(num),
    hash: nullable(str),
    hashBefore: str,
  },
  checkpointSaved: { name: str },
  error: { message: str },
};

const ENV_IN: Record<EnvIn['type'], Spec> = {
  ping: { t: num },
  init: { n: int, seeds: arrayOf(int, LIMITS.seeds), stage: int },
  step: { 'actions?': f32 },
  setTrackPool: { points: arrayOf(num, LIMITS.points), halfWidth: num, mu: num },
};

const ENV_OUT: Record<EnvOut['type'], Spec> = {
  pong: { t: num },
  ready: { trackId: str },
  ghosts: { poses: f32, n: nonNegInt, gen: nonNegInt, trackId: str, ids: arrayOf(str, LIMITS.envs) },
  trackAck: { trackId: str, envCount: nonNegInt },
  error: { message: str },
};

const FARM_IN: Record<FarmIn['type'], Spec> = {
  ping: { t: num },
  farmInit: { shard: nonNegInt, lo: nonNegInt, hi: nonNegInt, total: nonNegInt, actDim: int, buffer: sab },
  farmReset: { epoch: nonNegInt, tracks: arrayOf(trackSpec, LIMITS.envs), cfgs: arrayOf(episodeConfig, LIMITS.envs) },
  farmStop: {},
};

const FARM_OUT: Record<FarmOut['type'], Spec> = {
  pong: { t: num },
  farmReady: { shard: nonNegInt },
  farmResetDone: { shard: nonNegInt, epoch: nonNegInt },
  farmStopped: { shard: nonNegInt },
  error: { message: str, 'shard?': nonNegInt },
};

type Rec = Record<string, unknown>;
const len = (v: unknown) => (v as { length: number }).length;

/** Rules that span more than one field, keyed `channel:type`. */
const CROSS: Record<string, (m: Rec) => string | null> = {
  'learner:init': (m) => {
    const n = m.n as number;
    return n >= 1 && n <= LIMITS.learnerEnvs ? null : `init.n: expected 1 to ${LIMITS.learnerEnvs}, got ${n}`;
  },
  'learner:setTrackPool': (m) => (len(m.points) % 2 === 0 ? null : 'setTrackPool.points: expected x,z pairs (an even count)'),
  'env:init': (m) => {
    const n = m.n as number;
    return n >= 1 && n <= LIMITS.envs ? null : `init.n: expected 1 to ${LIMITS.envs}, got ${n}`;
  },
  'env:setTrackPool': (m) => (len(m.points) % 2 === 0 ? null : 'setTrackPool.points: expected x,z pairs (an even count)'),
  'learnerOut:ghosts': (m) => {
    const want = (m.n as number) * 7;
    return len(m.poses) === want ? null : `ghosts.poses: expected n * 7 = ${want} floats, got ${len(m.poses)}`;
  },
  'envOut:ghosts': (m) => {
    const want = (m.n as number) * 7;
    return len(m.poses) === want ? null : `ghosts.poses: expected n * 7 = ${want} floats, got ${len(m.poses)}`;
  },
  'farm:farmInit': (m) => {
    const { lo, hi, total, actDim } = m as { lo: number; hi: number; total: number; actDim: number };
    if (!(lo < hi && hi <= total && total <= LIMITS.envs)) return `farmInit: expected lo < hi <= total <= ${LIMITS.envs}, got ${lo}, ${hi}, ${total}`;
    return actDim === 2 || actDim === 3 ? null : `farmInit.actDim: expected 2 or 3, got ${actDim}`;
  },
  'farm:farmReset': (m) =>
    len(m.tracks) === len(m.cfgs) && len(m.tracks) > 0
      ? null
      : `farmReset: expected the same non-zero number of tracks and cfgs, got ${len(m.tracks)} and ${len(m.cfgs)}`,
};

function parseWith<T>(channel: string, table: Record<string, Spec>, data: unknown): Parsed<T> {
  if (typeof data !== 'object' || data === null || Array.isArray(data) || ArrayBuffer.isView(data)) {
    return { ok: false, error: `message: expected an object, got ${typeName(data)}` };
  }
  const rec = data as Rec;
  const type = rec.type;
  if (typeof type !== 'string') return { ok: false, error: `message.type: expected a string, got ${typeName(type)}` };
  const spec = Object.prototype.hasOwnProperty.call(table, type) ? table[type] : undefined;
  if (!spec) return { ok: false, error: `${channel}: unknown message type ${JSON.stringify(type)}` };
  if (type !== 'ping' && type !== 'pong') {
    if (rec.v === undefined) return { ok: false, error: `${type}.v: missing (protocol ${PROTOCOL_VERSION} expected; version 1 messages had no v)` };
    if (rec.v !== PROTOCOL_VERSION) return { ok: false, error: `${type}.v: protocol ${String(rec.v)} is not ${PROTOCOL_VERSION}` };
  }
  const e = obj(spec)(rec, type);
  if (e) return { ok: false, error: e };
  const cross = CROSS[`${channel}:${type}`];
  const ce = cross ? cross(rec) : null;
  if (ce) return { ok: false, error: ce };
  return { ok: true, msg: rec as T };
}

export const parseLearnerIn = (data: unknown): Parsed<LearnerIn> => parseWith('learner', LEARNER_IN, data);
export const parseLearnerOut = (data: unknown): Parsed<LearnerOut> => parseWith('learnerOut', LEARNER_OUT, data);
export const parseEnvIn = (data: unknown): Parsed<EnvIn> => parseWith('env', ENV_IN, data);
export const parseEnvOut = (data: unknown): Parsed<EnvOut> => parseWith('envOut', ENV_OUT, data);
export const parseFarmIn = (data: unknown): Parsed<FarmIn> => parseWith('farm', FARM_IN, data);
export const parseFarmOut = (data: unknown): Parsed<FarmOut> => parseWith('farmOut', FARM_OUT, data);

/** Message types per channel direction, for tests and docs. */
export const MESSAGE_TYPES = {
  learnerIn: Object.keys(LEARNER_IN) as LearnerIn['type'][],
  learnerOut: Object.keys(LEARNER_OUT) as LearnerOut['type'][],
  envIn: Object.keys(ENV_IN) as EnvIn['type'][],
  envOut: Object.keys(ENV_OUT) as EnvOut['type'][],
  farmIn: Object.keys(FARM_IN) as FarmIn['type'][],
  farmOut: Object.keys(FARM_OUT) as FarmOut['type'][],
};

/** Liveness probe, shared by every worker. Unversioned on purpose. */
export function handlePing(msg: { type: string; t?: number }): { type: 'pong'; t: number } | null {
  if (msg.type === 'ping') return { type: 'pong', t: msg.t ?? 0 };
  return null;
}
