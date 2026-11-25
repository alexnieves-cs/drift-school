/**
 * The env farm host: `total` env slots split into contiguous shards, each shard a VecEnv in its own
 * worker, all sharing one SharedArrayBuffer (layout and handshake in envFarmLayout.ts).
 *
 * The host runs in the learner (a worker in the browser, the main thread in Node). `step()` blocks
 * with Atomics.wait, which browsers only allow off the main thread; `stepAsync()` does not block.
 * Inline mode (`spawn` left out) runs the same FarmShard code in-process with no threads. That is
 * the deterministic reference path for tests: slot i of the farm matches slot i of one big VecEnv
 * over the same tracks and configs, whatever the shard count.
 *
 * Limits: no per-episode sampler (a slot restarts on its own track and config, like a VecEnv with no
 * sampler), and the page trainer does not use the farm yet.
 */
import { type EpisodeConfig, type StepInfo, type TrackSpec } from '../shared/types';
import { parseFarmOut, stamp, type FarmIn, type FarmOut } from '../shared/messages';
import { CTRL, EP, EP_STRIDE, INFO, INFO_STRIDE, farmBytes, farmSupport, farmViews, shardRanges, type FarmViews } from './envFarmLayout';
import { FarmShard } from './envFarmShard';

/** One worker as the host sees it. Browser Workers and Node worker_threads both fit. */
export interface FarmEndpoint {
  post(msg: FarmIn): void;
  onMessage(fn: (data: unknown) => void): void;
  terminate(): void | Promise<unknown>;
}

export type SpawnShard = (shard: number) => FarmEndpoint;

export interface EnvFarmOpts {
  total: number;
  /** Shard count, default 4, capped at `total` and at MAX_SHARDS. */
  shards?: number;
  actDim?: number;
  /** Start one worker per shard. Left out, shards run inline in this thread. */
  spawn?: SpawnShard;
  /** How long a step or a reset may take before the host gives up. Default 10 s. */
  timeoutMs?: number;
}

/** A finished episode, as VecEnv.finished records it, tagged with its slot. */
export interface FarmEpisodeRecord {
  slot: number;
  lap: boolean;
  lapTime?: number;
  crash: boolean;
  ret: number;
  trackId: string;
  tRef: number;
  speed: number;
}

/** Views into shared memory, valid until the next step or reset. Copy what you keep. */
export interface FarmStep {
  obs: Float32Array;
  rew: Float32Array;
  done: Uint8Array;
  trunc: Uint8Array;
  episodes: FarmEpisodeRecord[];
}

interface Remote {
  ep: FarmEndpoint;
  waiters: ((msg: FarmOut) => boolean)[];
}

export class EnvFarm {
  readonly total: number;
  readonly actDim: number;
  readonly ranges: [number, number][];
  readonly views: FarmViews;
  readonly mode: 'inline' | 'workers';
  /** Error messages workers reported, newest last. */
  readonly errors: string[] = [];
  private readonly inline: FarmShard[] = [];
  private readonly remotes: Remote[] = [];
  private readonly timeoutMs: number;
  private tracks: TrackSpec[] = [];
  private seq = 0;
  private epoch = 0;
  private closed = false;

  private constructor(opts: EnvFarmOpts, buffer: SharedArrayBuffer) {
    this.total = opts.total;
    this.actDim = opts.actDim ?? 2;
    this.ranges = shardRanges(opts.total, opts.shards ?? 4);
    this.views = farmViews(buffer, this.total, this.actDim);
    this.mode = opts.spawn ? 'workers' : 'inline';
    this.timeoutMs = opts.timeoutMs ?? 10_000;
  }

  static async open(opts: EnvFarmOpts): Promise<EnvFarm> {
    const support = farmSupport();
    if (!support.ok) throw new Error(`env farm unavailable: ${support.reason}`);
    if (!Number.isInteger(opts.total) || opts.total < 1) throw new Error(`env farm: total must be a positive integer, got ${opts.total}`);
    const actDim = opts.actDim ?? 2;
    const buffer = new SharedArrayBuffer(farmBytes(opts.total, actDim));
    const farm = new EnvFarm(opts, buffer);
    const inits = farm.ranges.map(([lo, hi], shard) => ({ type: 'farmInit' as const, shard, lo, hi, total: opts.total, actDim, buffer }));
    if (!opts.spawn) {
      for (const init of inits) farm.inline.push(new FarmShard(init));
      return farm;
    }
    for (let k = 0; k < inits.length; k++) {
      const ep = opts.spawn(k);
      const remote: Remote = { ep, waiters: [] };
      ep.onMessage((data) => farm.receive(remote, data));
      farm.remotes.push(remote);
    }
    try {
      await Promise.all(
        farm.remotes.map((r, k) => {
          const ready = farm.expect(r, (m) => m.type === 'farmReady' && m.shard === k, `shard ${k} init`);
          r.ep.post(stamp(inits[k]!));
          return ready;
        }),
      );
    } catch (err) {
      await farm.close();
      throw err;
    }
    return farm;
  }

  get shardCount(): number {
    return this.ranges.length;
  }

  /** Start fresh episodes everywhere and return the first observations (a shared view). */
  async reset(tracks: TrackSpec[], cfgs: EpisodeConfig[]): Promise<Float32Array> {
    this.assertOpen();
    if (tracks.length !== this.total || cfgs.length !== this.total) {
      throw new Error(`env farm: expected ${this.total} tracks and cfgs, got ${tracks.length} and ${cfgs.length}`);
    }
    this.tracks = tracks.slice();
    const epoch = ++this.epoch;
    if (this.mode === 'inline') {
      this.ranges.forEach(([lo, hi], k) => this.inline[k]!.reset(tracks.slice(lo, hi), cfgs.slice(lo, hi)));
    } else {
      await Promise.all(
        this.remotes.map((r, k) => {
          const [lo, hi] = this.ranges[k]!;
          const done = this.expect(r, (m) => m.type === 'farmResetDone' && m.shard === k && m.epoch === epoch, `shard ${k} reset`);
          r.ep.post(stamp({ type: 'farmReset', epoch, tracks: tracks.slice(lo, hi), cfgs: cfgs.slice(lo, hi) }));
          return done;
        }),
      );
    }
    return this.views.obs;
  }

  /** Step every slot. Blocks until all shards finish (Atomics.wait in worker mode). */
  step(actions: Float32Array): FarmStep {
    const seq = this.publish(actions);
    if (this.mode === 'inline') {
      for (const s of this.inline) s.step();
    } else {
      const deadline = Date.now() + this.timeoutMs;
      const ctrl = this.views.ctrl;
      for (let k = 0; k < this.ranges.length; k++) {
        const at = CTRL.DONE0 + k;
        for (;;) {
          const v = Atomics.load(ctrl, at);
          if (v === seq) break;
          if (v === -seq) throw this.shardFailed(k, seq);
          const left = deadline - Date.now();
          if (left <= 0) throw new Error(`env farm: shard ${k} did not finish step ${seq} within ${this.timeoutMs} ms`);
          Atomics.wait(ctrl, at, v, left);
        }
      }
    }
    return this.collect();
  }

  /** Same as step() without blocking the thread, for hosts on a browser main thread. */
  async stepAsync(actions: Float32Array): Promise<FarmStep> {
    if (this.mode === 'inline') return this.step(actions);
    const seq = this.publish(actions);
    const deadline = Date.now() + this.timeoutMs;
    const ctrl = this.views.ctrl;
    for (let k = 0; k < this.ranges.length; k++) {
      const at = CTRL.DONE0 + k;
      for (;;) {
        const v = Atomics.load(ctrl, at);
        if (v === seq) break;
        if (v === -seq) throw this.shardFailed(k, seq);
        const left = deadline - Date.now();
        if (left <= 0) throw new Error(`env farm: shard ${k} did not finish step ${seq} within ${this.timeoutMs} ms`);
        const waitAsync = (Atomics as unknown as { waitAsync?: (a: Int32Array, i: number, v: number, t: number) => { async: boolean; value: Promise<string> | string } }).waitAsync;
        if (typeof waitAsync === 'function') {
          const r = waitAsync(ctrl, at, v, left);
          if (r.async) await r.value;
        } else {
          await new Promise((res) => setTimeout(res, 0));
        }
      }
    }
    return this.collect();
  }

  /** Poses of every slot, 7 floats each (x, y, z, then a yaw quaternion), as a shared view. */
  poses(): Float32Array {
    return this.views.poses;
  }

  /** Decode one slot's step info from shared memory. */
  info(slot: number): StepInfo {
    const v = this.views.info;
    const b = slot * INFO_STRIDE;
    const lapDone = v[b + INFO.lapDone] === 1;
    return {
      progress: v[b + INFO.progress]!,
      lapDone,
      lapTime: lapDone ? v[b + INFO.lapTime] : undefined,
      offTrack: v[b + INFO.offTrack] === 1,
      wallHit: v[b + INFO.wallHit] === 1,
      speed: v[b + INFO.speed]!,
      wrongWay: v[b + INFO.wrongWay] === 1,
      stuck: v[b + INFO.stuck] === 1,
      rewardRaw: v[b + INFO.rewardRaw]!,
    };
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    Atomics.store(this.views.ctrl, CTRL.STOP, 1);
    Atomics.notify(this.views.ctrl, CTRL.SEQ);
    await Promise.all(
      this.remotes.map(async (r, k) => {
        const stopped = this.expect(r, (m) => m.type === 'farmStopped', `shard ${k} stop`, 2000).catch(() => undefined);
        r.ep.post(stamp({ type: 'farmStop' }));
        await stopped;
        await r.ep.terminate();
      }),
    );
  }

  private publish(actions: Float32Array): number {
    this.assertOpen();
    if (this.tracks.length === 0) throw new Error('env farm: step before reset');
    const want = this.total * this.actDim;
    if (actions.length !== want) throw new Error(`env farm: expected ${want} actions, got ${actions.length}`);
    this.views.actions.set(actions);
    const seq = ++this.seq;
    if (this.mode === 'workers') {
      Atomics.store(this.views.ctrl, CTRL.SEQ, seq);
      Atomics.notify(this.views.ctrl, CTRL.SEQ);
    }
    return seq;
  }

  private collect(): FarmStep {
    const v = this.views;
    const episodes: FarmEpisodeRecord[] = [];
    for (let i = 0; i < this.total; i++) {
      if (!v.done[i] && !v.trunc[i]) continue;
      const b = i * EP_STRIDE;
      const lap = v.ep[b + EP.lap] === 1;
      episodes.push({
        slot: i,
        lap,
        lapTime: lap ? v.ep[b + EP.lapTime] : undefined,
        crash: v.ep[b + EP.crash] === 1,
        ret: v.ep[b + EP.ret]!,
        trackId: this.tracks[i]!.id,
        tRef: v.ep[b + EP.tRef]!,
        speed: v.ep[b + EP.speed]!,
      });
    }
    return { obs: v.obs, rew: v.rew, done: v.done, trunc: v.trunc, episodes };
  }

  private shardFailed(k: number, seq: number): Error {
    const last = this.errors.at(-1);
    return new Error(`env farm: shard ${k} failed at step ${seq}${last ? ` (${last})` : ''}`);
  }

  private receive(r: Remote, data: unknown): void {
    const parsed = parseFarmOut(data);
    if (!parsed.ok) {
      this.errors.push(`unreadable message from a shard: ${parsed.error}`);
      return;
    }
    const msg = parsed.msg;
    if (msg.type === 'error') this.errors.push(msg.shard !== undefined ? `shard ${msg.shard}: ${msg.message}` : msg.message);
    r.waiters = r.waiters.filter((w) => !w(msg));
  }

  /** Resolve on the first message that matches, reject on a shard error or a timeout. */
  private expect(r: Remote, match: (m: FarmOut) => boolean, what: string, timeoutMs = this.timeoutMs): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        r.waiters = r.waiters.filter((w) => w !== waiter);
        reject(new Error(`env farm: ${what} timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      const waiter = (m: FarmOut): boolean => {
        if (match(m)) {
          clearTimeout(timer);
          resolve();
          return true;
        }
        if (m.type === 'error') {
          clearTimeout(timer);
          reject(new Error(`env farm: ${what} failed: ${m.message}`));
          return true;
        }
        return false;
      };
      r.waiters.push(waiter);
    });
  }

  private assertOpen(): void {
    if (this.closed) throw new Error('env farm: closed');
  }
}

