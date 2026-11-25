/**
 * Shared-memory layout of the env farm (SPEC architecture: "EnvW -- obs/rew/done SAB --> LearnW").
 *
 * One SharedArrayBuffer holds everything a step exchanges, so a step costs no message and no copy
 * through the structured-clone path. Each region is a typed-array view at an 8-byte aligned offset.
 * Slot `i` of the farm owns row `i` of every region; shard k writes only the rows [lo_k, hi_k).
 *
 *   ctrl    Int32   CTRL_WORDS    [SEQ, STOP, DONE_0 .. DONE_{MAX_SHARDS-1}]
 *   actions Float32 total*actDim  written by the learner before it bumps SEQ
 *   obs     Float32 total*OBS_DIM written by the shards
 *   rew     Float32 total
 *   done    Uint8   total
 *   trunc   Uint8   total
 *   poses   Float32 total*7       x, y, z, qw, qx, qy, qz (VecEnv.poses)
 *   info    Float32 total*INFO_STRIDE  StepInfo fields, see INFO
 *   ep      Float32 total*EP_STRIDE    the episode that ended this step on that slot, see EP
 *
 * Step handshake: the learner writes `actions`, stores SEQ = s and notifies it. Shard k waits on SEQ,
 * steps its slots, writes its rows, stores DONE_k = s and notifies DONE_k. A shard that fails stores
 * DONE_k = -s so the learner fails fast instead of timing out.
 */
import { OBS_DIM } from '../shared/types';

export const MAX_SHARDS = 16;
export const CTRL = { SEQ: 0, STOP: 1, DONE0: 2 } as const;
export const CTRL_WORDS = 2 + MAX_SHARDS;

/** Per-slot StepInfo, flattened. Booleans are 0/1; a missing lapTime is 0. */
export const INFO = { progress: 0, lapDone: 1, lapTime: 2, offTrack: 3, wallHit: 4, speed: 5, wrongWay: 6, stuck: 7, rewardRaw: 8 } as const;
export const INFO_STRIDE = 9;

/** Per-slot record of an episode that ended this step (valid when done or trunc is set). */
export const EP = { lap: 0, lapTime: 1, crash: 2, ret: 3, tRef: 4, speed: 5 } as const;
export const EP_STRIDE = 6;

export interface FarmViews {
  readonly total: number;
  readonly actDim: number;
  ctrl: Int32Array;
  actions: Float32Array;
  obs: Float32Array;
  rew: Float32Array;
  done: Uint8Array;
  trunc: Uint8Array;
  poses: Float32Array;
  info: Float32Array;
  ep: Float32Array;
}

const align = (x: number) => Math.ceil(x / 8) * 8;

function regions(total: number, actDim: number): { name: keyof FarmViews; kind: 'i32' | 'f32' | 'u8'; count: number }[] {
  return [
    { name: 'ctrl', kind: 'i32', count: CTRL_WORDS },
    { name: 'actions', kind: 'f32', count: total * actDim },
    { name: 'obs', kind: 'f32', count: total * OBS_DIM },
    { name: 'rew', kind: 'f32', count: total },
    { name: 'done', kind: 'u8', count: total },
    { name: 'trunc', kind: 'u8', count: total },
    { name: 'poses', kind: 'f32', count: total * 7 },
    { name: 'info', kind: 'f32', count: total * INFO_STRIDE },
    { name: 'ep', kind: 'f32', count: total * EP_STRIDE },
  ];
}

const BYTES = { i32: 4, f32: 4, u8: 1 } as const;

/** Bytes of shared memory a farm of `total` slots needs. */
export function farmBytes(total: number, actDim: number): number {
  let off = 0;
  for (const r of regions(total, actDim)) off = align(off + r.count * BYTES[r.kind]);
  return off;
}

/** Typed views over a farm buffer. The learner and every shard build the same views from the same numbers. */
export function farmViews(buffer: SharedArrayBuffer, total: number, actDim: number): FarmViews {
  if (buffer.byteLength < farmBytes(total, actDim)) throw new Error(`farm buffer too small: ${buffer.byteLength} < ${farmBytes(total, actDim)}`);
  const out: Partial<Record<keyof FarmViews, unknown>> = { total, actDim };
  let off = 0;
  for (const r of regions(total, actDim)) {
    const view = r.kind === 'i32' ? new Int32Array(buffer, off, r.count) : r.kind === 'f32' ? new Float32Array(buffer, off, r.count) : new Uint8Array(buffer, off, r.count);
    out[r.name] = view;
    off = align(off + r.count * BYTES[r.kind]);
  }
  return out as FarmViews;
}

/** Contiguous shard ranges [lo, hi) covering 0..total, sizes differing by at most one. */
export function shardRanges(total: number, shards: number): [number, number][] {
  const k = Math.max(1, Math.min(shards, total, MAX_SHARDS));
  const out: [number, number][] = [];
  let lo = 0;
  for (let i = 0; i < k; i++) {
    const size = Math.floor(total / k) + (i < total % k ? 1 : 0);
    out.push([lo, lo + size]);
    lo += size;
  }
  return out;
}

/**
 * SharedArrayBuffer needs cross-origin isolation in browsers (the COOP/COEP headers that
 * vite.config.ts sets, or public/coi-serviceworker.js on static hosting). Node always has it.
 */
export function farmSupport(): { ok: true } | { ok: false; reason: string } {
  if (typeof SharedArrayBuffer === 'undefined') {
    return { ok: false, reason: 'SharedArrayBuffer is not available: serve the page with COOP same-origin and COEP require-corp' };
  }
  const iso = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated;
  if (iso === false) return { ok: false, reason: 'the page is not cross-origin isolated (COOP/COEP headers missing)' };
  if (typeof Atomics === 'undefined') return { ok: false, reason: 'Atomics is not available' };
  return { ok: true };
}
