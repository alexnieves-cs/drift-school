import { ACT_DIM, OBS_DIM, type EpisodeConfig, type TrackSpec } from '../shared/types';
import { makeRng } from '../shared/rng';
import type { VecEnv } from '../sim/vecEnv';
import type { Trainer } from './trainer';

/**
 * Hot-swap and fine-tune on a user track (SPEC "Draw-a-track editor", steps 2-4).
 *
 * Timeline, driven by repeated `step()` calls (the learner worker and `npm run eval` both use it):
 *
 *  1. drain      Every env finishes its current episode, then restarts on the user track. An env
 *                that has not finished by `drainMs` is retargeted anyway, so the swap lands inside
 *                the 15 s budget even when a car is mid-lap. No gradient steps run, so the
 *                weights are byte-identical to the moment of the click.
 *  2. zeroShot   All envs are on the user track. The sampled policy drives the first
 *                `zeroShotEpisodes` episodes (time-capped) with no update: that completion rate
 *                is the zero-shot result.
 *  3. tune       The pool becomes 70% user track / 30% procedural replay (TrackMix.startFineTune:
 *                entropy 0.02 for 20 PPO updates, LR warm restart). Each finished episode draws
 *                its next track from that pool. One `step()` is `imitationPerPpo` DAgger batches
 *                followed by one PPO update.
 *
 * About the DAgger part, plainly: the adaptation is NOT pure reinforcement learning. A DAgger
 * batch rolls the current policy (the pure-pursuit baseline drives a `beta` fraction of the
 * steps), labels every visited state with the pure-pursuit action, and regresses the policy mean
 * onto those labels, on both the user track and the procedural replay tracks. SPEC lets the
 * baseline act as t_ref and demo car; here it is also a teacher. `mode` switches the parts off
 * (`ppo` = PPO only, `imitation` = DAgger only) so the eval can report what each contributes.
 */

export type FineTuneMode = 'mixed' | 'ppo' | 'imitation';
export type FineTunePhase = 'idle' | 'drain' | 'zeroShot' | 'tune' | 'done';

export interface FineTuneOpts {
  user: TrackSpec;
  /** Replay tracks for the 30% share. */
  procedural: TrackSpec[];
  mode?: FineTuneMode;
  /** Hard bound on the drain, in ms. */
  drainMs?: number;
  zeroShotEpisodes?: number;
  zeroShotMs?: number;
  /** DAgger batches per PPO update in `mixed` mode. */
  imitationPerPpo?: number;
  /** Fraction of rollout steps the expert drives in a DAgger batch. */
  beta?: number;
  /** Imitation learning rate (PPO keeps TrackMix's warm-restart rate). */
  imitationLr?: number;
  now?: () => number;
}

export interface FineTuneReport {
  phase: FineTunePhase;
  /** ms from start() to the last env landing on the user track; null while draining. */
  latencyMs: number | null;
  envsOnUser: number;
  envTotal: number;
  /** Envs on the user track the moment the drain finished (all of them, by construction). */
  envsOnUserAtSwap: number | null;
  weightHashStart: string;
  /** Hash the moment the last env landed on the user track (no update has run yet). */
  weightHashAtDrain: string | null;
  /** Hash at the end of the zero-shot phase, still before any update. */
  weightHashAtSwap: string | null;
  weightsUnchanged: boolean | null;
  zeroShot: { completion: number; episodes: number; ms: number } | null;
  /** Learner steps since the swap (one step is a DAgger burst plus a PPO update). */
  steps: number;
  ppoUpdates: number;
  imitationBatches: number;
  /** Completion over the last 16 policy-driven user-track episodes of the tune phase (sampled actions, no expert). */
  userCompletion: number | null;
  userBestLap: number | null;
  /** Wall ms spent inside tune-phase steps. */
  tuneMs: number;
}

type Retargetable = { retargetSlot?: (i: number, track: TrackSpec, cfg: EpisodeConfig) => void };
type SlotLike = { track: TrackSpec; cfg: EpisodeConfig; rays: { setTrack(t: TrackSpec): void } };
type EnvInternals = { slots?: SlotLike[]; place?: (s: SlotLike) => void; writeObs?: (i: number) => void };

/**
 * Point env slot `i` at `track` and restart its episode there. Uses `VecEnv.retargetSlot` when
 * the env has one; otherwise reaches the same three private members `VecEnv.reset` uses.
 * Throws if neither exists, so a VecEnv refactor fails loudly (tests/rl/finetune.test.ts).
 * The slot keeps its Vehicle, so car parameters drawn for its first episode carry over; track,
 * noise scale, latency and the ray field are the new ones.
 *
 * Proposed public form for src/sim/vecEnv.ts (physics worker's file, so not added here):
 *   retargetSlot(i, track, cfg) { const s = this.slots[i]; if (!s) return; s.track = track;
 *     s.cfg = { ...cfg, trackId: track.id }; s.rays.setTrack(track); this.place(s); this.writeObs(i); }
 */
export function retargetSlot(env: VecEnv, i: number, track: TrackSpec, cfg: EpisodeConfig): void {
  const pub = (env as unknown as Retargetable).retargetSlot;
  if (typeof pub === 'function') {
    pub.call(env, i, track, cfg);
    return;
  }
  const priv = env as unknown as EnvInternals;
  const slot = priv.slots?.[i];
  if (!slot || !priv.place || !priv.writeObs) throw new Error('VecEnv has no retargetSlot and no slots/place/writeObs');
  slot.track = track;
  slot.cfg = { ...cfg, trackId: track.id };
  slot.rays.setTrack(track);
  priv.place.call(env, slot);
  priv.writeObs.call(env, i);
}

/** Run `hook` after every `env.step`. Returns the function that removes it. */
export function tapEnvStep(env: VecEnv, hook: (done: Uint8Array, trunc: Uint8Array) => void): () => void {
  const original = env.step;
  env.step = function tapped(this: VecEnv, actions: Float32Array) {
    const out = original.call(env, actions);
    hook(out.done, out.trunc);
    return out;
  };
  return () => {
    env.step = original;
  };
}

/**
 * DAgger with an aggregated dataset. Each `batch` rolls `trainer.horizon` steps on the trainer's
 * envs (pure pursuit drives a `beta` share of the steps, the policy mean the rest), labels every
 * visited state with the pure-pursuit action, adds the pairs to a ring buffer, then takes a few
 * Adam steps on minibatches drawn from the whole buffer. Training on the aggregate, not on the
 * newest rollout alone, is what keeps the regression from oscillating (a latest-batch-only
 * version reached 100% on the user track at 20 s and was back at 0% by 60 s in testing).
 */
export class Imitator {
  readonly capacity: number;
  private readonly obs: Float32Array;
  private readonly act: Float32Array;
  private size = 0;
  private head = 0;
  batches = 0;
  lastLoss = 0;

  constructor(
    private readonly lr0 = 1e-3,
    private readonly lrMin = 1e-4,
    capacity = 20_000,
    private readonly minibatch = 256,
    private readonly stepsPerBatch = 6,
  ) {
    this.capacity = capacity;
    this.obs = new Float32Array(capacity * OBS_DIM);
    this.act = new Float32Array(capacity * ACT_DIM);
  }

  get samples(): number {
    return this.size;
  }

  private add(obs: Float32Array, act: Float32Array, count: number): void {
    for (let k = 0; k < count; k++) {
      this.obs.set(obs.subarray(k * OBS_DIM, (k + 1) * OBS_DIM), this.head * OBS_DIM);
      this.act.set(act.subarray(k * ACT_DIM, (k + 1) * ACT_DIM), this.head * ACT_DIM);
      this.head = (this.head + 1) % this.capacity;
      this.size = Math.min(this.capacity, this.size + 1);
    }
  }

  /** Roll one batch and fit. Returns the last minibatch loss. */
  batch(trainer: Trainer, beta: number): number {
    const env = trainer.env;
    const n = env.n;
    const T = trainer.horizon;
    const obsBuf = new Float32Array(T * n * OBS_DIM);
    const actBuf = new Float32Array(T * n * ACT_DIM);
    const act = new Float32Array(n * ACT_DIM);
    const rng = makeRng(this.batches * 7919 + 13, 'dagger');
    for (let t = 0; t < T; t++) {
      const expert = env.baselineActions();
      obsBuf.set(env.obs, t * n * OBS_DIM);
      actBuf.set(expert, t * n * ACT_DIM);
      const mine = trainer.policy.actMean(env.obs, n);
      for (let i = 0; i < n; i++) {
        const src = rng.next() < beta ? expert : mine;
        for (let k = 0; k < ACT_DIM; k++) act[i * ACT_DIM + k] = src[i * ACT_DIM + k]!;
      }
      env.step(act);
    }
    trainer.syncObs();
    this.add(obsBuf, actBuf, T * n);
    // `imitate` runs its own Adam at `policy.lr`; the PPO optimizer keeps the rate it was built with.
    const saved = trainer.policy.lr;
    trainer.policy.lr = this.lrMin + (this.lr0 - this.lrMin) * Math.exp(-this.batches / 25);
    const mb = Math.min(this.minibatch, this.size);
    const ob = new Float32Array(mb * OBS_DIM);
    const ab = new Float32Array(mb * ACT_DIM);
    for (let s = 0; s < this.stepsPerBatch; s++) {
      for (let k = 0; k < mb; k++) {
        // Half the minibatch is the newest rollout, half is drawn from everything seen.
        const idx =
          k < mb / 2 && this.size >= T * n
            ? (this.head - 1 - Math.floor(rng.next() * T * n) + this.capacity * 2) % this.capacity
            : Math.floor(rng.next() * this.size);
        ob.set(this.obs.subarray(idx * OBS_DIM, (idx + 1) * OBS_DIM), k * OBS_DIM);
        ab.set(this.act.subarray(idx * ACT_DIM, (idx + 1) * ACT_DIM), k * ACT_DIM);
      }
      this.lastLoss = trainer.policy.imitate(ob, ab, mb);
    }
    trainer.policy.lr = saved;
    this.batches++;
    return this.lastLoss;
  }
}

export class FineTune {
  phase: FineTunePhase = 'idle';
  readonly user: TrackSpec;
  private readonly procedural: TrackSpec[];
  private readonly trainer: Trainer;
  private readonly mode: FineTuneMode;
  private readonly drainMs: number;
  private readonly zeroShotEpisodes: number;
  private readonly zeroShotMs: number;
  private readonly imitationPerPpo: number;
  private readonly beta: number;
  private readonly imitator: Imitator;
  private readonly now: () => number;
  private untap: (() => void) | null = null;
  private t0 = 0;
  private zsT0 = 0;
  private zsFrom = 0;
  private tuneFrom = 0;
  private swapRng = makeRng(31337, 'swap');
  private recent: { lap: boolean; lapTime?: number }[] = [];
  private bestLap: number | null = null;
  private seen = 0;
  private r: FineTuneReport;

  constructor(trainer: Trainer, opts: FineTuneOpts) {
    this.trainer = trainer;
    this.user = opts.user;
    this.procedural = opts.procedural;
    this.mode = opts.mode ?? 'mixed';
    this.drainMs = opts.drainMs ?? 8_000;
    this.zeroShotEpisodes = opts.zeroShotEpisodes ?? 64;
    this.zeroShotMs = opts.zeroShotMs ?? 12_000;
    this.imitationPerPpo = opts.imitationPerPpo ?? 3;
    this.beta = opts.beta ?? 0.3;
    this.imitator = new Imitator(opts.imitationLr ?? 1e-3);
    this.now = opts.now ?? (() => performance.now());
    this.r = {
      phase: 'idle',
      latencyMs: null,
      envsOnUser: 0,
      envTotal: trainer.env.n,
      envsOnUserAtSwap: null,
      weightHashStart: '',
      weightHashAtDrain: null,
      weightHashAtSwap: null,
      weightsUnchanged: null,
      zeroShot: null,
      steps: 0,
      ppoUpdates: 0,
      imitationBatches: 0,
      userCompletion: null,
      userBestLap: null,
      tuneMs: 0,
    };
  }

  /** Begin the swap. Cheap: nothing is reset until the envs' episodes end. */
  start(): void {
    if (this.phase !== 'idle') return;
    this.t0 = this.now();
    this.r.weightHashStart = this.trainer.weightHash();
    this.phase = 'drain';
    this.untap = tapEnvStep(this.trainer.env, (done, trunc) => this.onEpisodeEnd(done, trunc));
    // An env that is already sitting at the start of an episode loses nothing by moving now.
    this.checkDrained();
  }

  report(): FineTuneReport {
    const ids = this.trainer.env.trackIds();
    return {
      ...this.r,
      phase: this.phase,
      envsOnUser: ids.filter((id) => id === this.user.id).length,
      envTotal: ids.length,
    };
  }

  stop(): void {
    this.untap?.();
    this.untap = null;
    if (this.phase !== 'idle') this.phase = 'done';
  }

  private retarget(i: number, track: TrackSpec): void {
    retargetSlot(this.trainer.env, i, track, this.trainer.configFor(track, i));
  }

  private onEpisodeEnd(done: Uint8Array, trunc: Uint8Array): void {
    const env = this.trainer.env;
    for (let i = 0; i < env.n; i++) {
      if (!done[i] && !trunc[i]) continue;
      if (this.phase === 'drain' || this.phase === 'zeroShot') {
        if (env.trackId(i) !== this.user.id) this.retarget(i, this.user);
      } else if (this.phase === 'tune') {
        const next = this.trainer.mix.sample(this.swapRng);
        if (next.id !== env.trackId(i)) this.retarget(i, next);
      }
    }
  }

  private checkDrained(): void {
    if (this.phase !== 'drain') return;
    const env = this.trainer.env;
    if (this.now() - this.t0 >= this.drainMs) {
      for (let i = 0; i < env.n; i++) if (env.trackId(i) !== this.user.id) this.retarget(i, this.user);
    }
    if (env.trackIds().every((id) => id === this.user.id)) {
      this.r.latencyMs = this.now() - this.t0;
      this.r.weightHashAtDrain = this.trainer.weightHash();
      this.r.envsOnUserAtSwap = env.trackIds().filter((id) => id === this.user.id).length;
      this.trainer.syncObs();
      this.zsFrom = env.finished.length;
      this.zsT0 = this.now();
      this.phase = 'zeroShot';
    }
  }

  private userEpisodesSince(from: number): { lap: boolean; lapTime?: number }[] {
    return this.trainer.env.finished.slice(from).filter((e) => e.trackId === this.user.id);
  }

  /** One unit of work. Non-learning phases take a few dozen env steps so callers can yield often. */
  async step(): Promise<FineTuneReport> {
    const trainer = this.trainer;
    if (this.phase === 'drain') {
      trainer.rolloutNoUpdate(16);
      this.checkDrained();
    } else if (this.phase === 'zeroShot') {
      trainer.rolloutNoUpdate(16);
      const eps = this.userEpisodesSince(this.zsFrom);
      const elapsed = this.now() - this.zsT0;
      if (eps.length >= this.zeroShotEpisodes || elapsed >= this.zeroShotMs) {
        const laps = eps.filter((e) => e.lap).length;
        this.r.zeroShot = { completion: eps.length ? laps / eps.length : 0, episodes: eps.length, ms: elapsed };
        this.r.weightHashAtSwap = trainer.weightHash();
        this.r.weightsUnchanged = this.r.weightHashAtSwap === this.r.weightHashStart;
        trainer.mix.startFineTune(this.user, this.procedural);
        trainer.policy.setLr(trainer.mix.lr);
        this.tuneFrom = trainer.env.finished.length;
        this.seen = this.tuneFrom;
        this.phase = 'tune';
      }
    } else if (this.phase === 'tune') {
      const t = this.now();
      const batches = this.mode === 'ppo' ? 0 : this.mode === 'imitation' ? 1 : this.imitationPerPpo;
      for (let b = 0; b < batches; b++) {
        this.imitator.batch(trainer, this.beta);
        this.r.imitationBatches++;
      }
      // Episodes that finished during a DAgger batch had the pure-pursuit car driving part of
      // the time, so they do not count toward the completion shown to the user. Only episodes
      // from the policy's own PPO rollouts do (imitation-only mode has none, so it counts all).
      if (this.mode !== 'imitation') this.seen = trainer.env.finished.length;
      if (this.mode !== 'imitation') {
        await trainer.update();
        this.r.ppoUpdates++;
      }
      this.r.steps++;
      this.collect();
      this.r.tuneMs += this.now() - t;
    }
    return this.report();
  }

  private collect(): void {
    const fresh = this.trainer.env.finished.slice(this.seen).filter((e) => e.trackId === this.user.id);
    this.seen = this.trainer.env.finished.length;
    for (const e of fresh) {
      this.recent.push({ lap: e.lap, lapTime: e.lapTime });
      if (e.lap && e.lapTime && (this.bestLap === null || e.lapTime < this.bestLap)) this.bestLap = e.lapTime;
    }
    if (this.recent.length > 16) this.recent.splice(0, this.recent.length - 16);
    this.r.userCompletion = this.recent.length ? this.recent.filter((e) => e.lap).length / this.recent.length : null;
    this.r.userBestLap = this.bestLap;
  }
}
