/**
 * One shard of the env farm: a VecEnv over slots [lo, hi) of the farm, reading actions from and
 * writing results to the shared buffer (src/workers/envFarmLayout.ts). The same class runs inside a
 * browser worker (envFarm.worker.ts), a Node worker thread (tests) and in-process (EnvFarm's inline
 * mode), so the deterministic test path exercises exactly the code the workers run.
 */
import { OBS_DIM, type EpisodeConfig, type TrackSpec } from '../shared/types';
import { handlePing, parseFarmIn, stamp, type FarmIn, type FarmOut } from '../shared/messages';
import { VecEnv } from '../sim/vecEnv';
import { CTRL, EP, EP_STRIDE, INFO, INFO_STRIDE, farmViews, type FarmViews } from './envFarmLayout';

export type FarmInit = Extract<FarmIn, { type: 'farmInit' }>;

export class FarmShard {
  readonly shard: number;
  readonly lo: number;
  readonly hi: number;
  readonly views: FarmViews;
  readonly env: VecEnv;
  private readonly local: Float32Array;
  private readonly actDim: number;

  constructor(init: FarmInit) {
    this.shard = init.shard;
    this.lo = init.lo;
    this.hi = init.hi;
    if (init.actDim !== 2 && init.actDim !== 3) throw new Error(`shard ${init.shard}: actDim must be 2 or 3, got ${init.actDim}`);
    this.actDim = init.actDim;
    this.views = farmViews(init.buffer, init.total, init.actDim);
    this.env = new VecEnv(init.hi - init.lo, { slotOffset: init.lo, actDim: init.actDim });
    this.local = new Float32Array((init.hi - init.lo) * init.actDim);
  }

  /** Start fresh episodes on this shard's slots and publish their first observations. */
  reset(tracks: TrackSpec[], cfgs: EpisodeConfig[]): void {
    const n = this.hi - this.lo;
    if (tracks.length !== n || cfgs.length !== n) throw new Error(`shard ${this.shard}: expected ${n} tracks and cfgs, got ${tracks.length} and ${cfgs.length}`);
    const obs = this.env.reset(tracks, cfgs);
    const v = this.views;
    v.obs.set(obs, this.lo * OBS_DIM);
    v.poses.set(this.env.poses(), this.lo * 7);
    v.rew.fill(0, this.lo, this.hi);
    v.done.fill(0, this.lo, this.hi);
    v.trunc.fill(0, this.lo, this.hi);
    v.info.fill(0, this.lo * INFO_STRIDE, this.hi * INFO_STRIDE);
    v.ep.fill(0, this.lo * EP_STRIDE, this.hi * EP_STRIDE);
  }

  /** Step every slot of this shard with the actions currently in shared memory. */
  step(): void {
    const v = this.views;
    const n = this.hi - this.lo;
    this.local.set(v.actions.subarray(this.lo * this.actDim, this.hi * this.actDim));
    const before = this.env.finished.length;
    const out = this.env.step(this.local);
    const lo = this.lo;
    v.obs.set(out.obs, lo * OBS_DIM);
    v.rew.set(out.rew, lo);
    v.done.set(out.done, lo);
    v.trunc.set(out.trunc, lo);
    v.poses.set(this.env.poses(), lo * 7);
    for (let i = 0; i < n; i++) {
      const info = out.infos[i]!;
      const b = (lo + i) * INFO_STRIDE;
      v.info[b + INFO.progress] = info.progress;
      v.info[b + INFO.lapDone] = info.lapDone ? 1 : 0;
      v.info[b + INFO.lapTime] = info.lapTime ?? 0;
      v.info[b + INFO.offTrack] = info.offTrack ? 1 : 0;
      v.info[b + INFO.wallHit] = info.wallHit ? 1 : 0;
      v.info[b + INFO.speed] = info.speed;
      v.info[b + INFO.wrongWay] = info.wrongWay ? 1 : 0;
      v.info[b + INFO.stuck] = info.stuck ? 1 : 0;
      v.info[b + INFO.rewardRaw] = info.rewardRaw;
    }
    // VecEnv appends one record per ended slot, in slot order, so the k-th new record belongs to the
    // k-th slot (ascending) whose done or trunc flag is set.
    let k = before;
    for (let i = 0; i < n; i++) {
      if (!out.done[i] && !out.trunc[i]) continue;
      const rec = this.env.finished[k++];
      if (!rec) throw new Error(`shard ${this.shard}: missing episode record for slot ${lo + i}`);
      const b = (lo + i) * EP_STRIDE;
      v.ep[b + EP.lap] = rec.lap ? 1 : 0;
      v.ep[b + EP.lapTime] = rec.lapTime ?? 0;
      v.ep[b + EP.crash] = rec.crash ? 1 : 0;
      v.ep[b + EP.ret] = rec.ret;
      v.ep[b + EP.tRef] = rec.tRef;
      v.ep[b + EP.speed] = rec.speed;
    }
    this.env.finished.length = 0;
  }
}

/** The two ends of a shard's message channel, independent of browser or Node worker APIs. */
export interface FarmPort {
  post(msg: FarmOut): void;
  onMessage(fn: (data: unknown) => void): void;
  close(): void;
}

type WaitAsync = (ta: Int32Array, index: number, value: number, timeout?: number) => { async: boolean; value: Promise<string> | string };

/** Resolve once ctrl[SEQ] differs from `seen` (or after a timeout), without blocking message handling. */
async function nextSeq(ctrl: Int32Array, seen: number): Promise<void> {
  const waitAsync = (Atomics as unknown as { waitAsync?: WaitAsync }).waitAsync;
  if (typeof waitAsync === 'function') {
    const r = waitAsync(ctrl, CTRL.SEQ, seen, 500);
    if (r.async) await r.value;
    return;
  }
  // No Atomics.waitAsync (some browsers): block briefly, then yield so farmReset/farmStop can arrive.
  Atomics.wait(ctrl, CTRL.SEQ, seen, 2);
  if (Atomics.load(ctrl, CTRL.SEQ) === seen) await new Promise((r) => setTimeout(r, 0));
}

/**
 * Serve one shard over a port: handle farmInit / farmReset / farmStop messages, and step whenever
 * the learner bumps SEQ. Steps never go through messages.
 */
export function serveShard(port: FarmPort): void {
  let shard: FarmShard | null = null;
  let running = false;
  let stopped = false;

  const fail = (err: unknown, seq?: number) => {
    const message = err instanceof Error ? err.message : String(err);
    if (shard && seq !== undefined) {
      Atomics.store(shard.views.ctrl, CTRL.DONE0 + shard.shard, -seq);
      Atomics.notify(shard.views.ctrl, CTRL.DONE0 + shard.shard);
    }
    port.post(stamp({ type: 'error', message, shard: shard?.shard }));
  };

  const run = async () => {
    if (!shard || running) return;
    running = true;
    const ctrl = shard.views.ctrl;
    let seen = Atomics.load(ctrl, CTRL.SEQ);
    while (!stopped && Atomics.load(ctrl, CTRL.STOP) === 0) {
      await nextSeq(ctrl, seen);
      const seq = Atomics.load(ctrl, CTRL.SEQ);
      if (seq === seen || stopped) continue;
      seen = seq;
      try {
        shard.step();
        Atomics.store(ctrl, CTRL.DONE0 + shard.shard, seq);
        Atomics.notify(ctrl, CTRL.DONE0 + shard.shard);
      } catch (err) {
        fail(err, seq);
      }
    }
    running = false;
  };

  port.onMessage((data) => {
    const parsed = parseFarmIn(data);
    if (!parsed.ok) {
      port.post(stamp({ type: 'error', message: `rejected message: ${parsed.error}`, shard: shard?.shard }));
      return;
    }
    const msg = parsed.msg;
    const pong = handlePing(msg);
    if (pong) {
      port.post(pong);
      return;
    }
    try {
      if (msg.type === 'farmInit') {
        shard = new FarmShard(msg);
        port.post(stamp({ type: 'farmReady', shard: msg.shard }));
        void run();
      } else if (msg.type === 'farmReset') {
        if (!shard) throw new Error('farmReset before farmInit');
        shard.reset(msg.tracks, msg.cfgs);
        port.post(stamp({ type: 'farmResetDone', shard: shard.shard, epoch: msg.epoch }));
      } else if (msg.type === 'farmStop') {
        stopped = true;
        port.post(stamp({ type: 'farmStopped', shard: shard?.shard ?? 0 }));
        port.close();
      }
    } catch (err) {
      fail(err);
    }
  });
}
