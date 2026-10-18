import { ACT_DIM, OBS_DIM, type EpisodeConfig, type GenStat, type TrackSpec } from '../shared/types';
import { hashBuffers } from '../shared/hash';
import { makeRng, type RNG } from '../shared/rng';
import { VecEnv } from '../sim/vecEnv';
import { defaultCarParams } from '../sim/car/params';
import { latencyFor, noiseFor, sampleCar, type DRLevel } from '../sim/dr';
import { generateTrack } from '../track/generate';
import { GaussianPolicy, ensureBackend } from './model';
import { Curriculum } from './curriculum';
import { TrackMix } from './trackMix';
import { aggregateGen, type EpRecord } from './genStats';
import { RewardScale } from './normalize';

export interface TrainerOpts {
  n?: number;
  horizon?: number;
  epochs?: number;
  minibatch?: number;
  clip?: number;
  /** Seeds the policy init, exploration noise, minibatch shuffling and per-episode sampling. */
  seed?: number;
}

const GAMMA = 0.99;
const LAMBDA = 0.95;
/** PPO updates after an imitation warm start that train only the critic, so a random value head cannot wreck the policy. */
const CRITIC_WARMUP_UPDATES = 3;

/** Progress on a drawn track since the last fine-tune swap. */
export interface AdaptStats {
  trackId: string;
  updatesSinceSwap: number;
  /** Lap completion over the first 64 episodes on the new track (updates keep running meanwhile). Null until 64 have finished. */
  first64: number | null;
  /** Completion over the most recent episodes on the new track. */
  completion: number | null;
  bestLap: number | null;
  /** Completion on procedural tracks before the swap and over the latest episodes since. */
  procBefore: number | null;
  procNow: number | null;
}

const FIRST_EPISODES = 64;
const WINDOW = 64;

const mean = (xs: boolean[]): number | null => (xs.length ? xs.filter(Boolean).length / xs.length : null);

export class Trainer {
  readonly env: VecEnv;
  readonly policy: GaussianPolicy;
  readonly curriculum = new Curriculum(24);
  readonly mix = new TrackMix();
  readonly rewardScale = new RewardScale();
  gen = 0;
  horizon: number;
  epochs: number;
  minibatch: number;
  clip: number;
  dr: DRLevel = 'off';
  grip = 1;
  massScale = 1;
  private obs: Float32Array;
  private ready = false;
  private readonly rng: RNG;
  private episodeCount = 0;
  /** Procedural seeds the pool was built from. Other stages regenerate from the same seeds. */
  private seeds: number[] = [];
  private stagePools = new Map<number, TrackSpec[]>();
  private criticWarmup = 0;
  private procWindow: boolean[] = [];
  private adapt: { trackId: string; updates: number; first: boolean[]; recent: boolean[]; bestLap: number | null; procBefore: number | null } | null = null;
  /** Called after every env step during a rollout, e.g. to stream poses. */
  onStep: (() => void) | null = null;
  /** Hand control back to the host event loop every this many rollout steps (0 = never). */
  yieldEvery = 0;
  bestLap: number | null = null;
  readonly history: GenStat[] = [];
  /** Best-lap pose tape per generation, for leaderboard replay. */
  readonly replays = new Map<number, Float32Array>();

  constructor(opts: TrainerOpts = {}) {
    const n = opts.n ?? 16;
    const seed = opts.seed ?? 1;
    this.env = new VecEnv(n);
    this.policy = new GaussianPolicy(3e-4, seed);
    this.rng = makeRng(seed, 'trainer');
    this.horizon = opts.horizon ?? 64;
    this.epochs = opts.epochs ?? 3;
    this.minibatch = opts.minibatch ?? 256;
    this.clip = opts.clip ?? 0.2;
    this.obs = new Float32Array(n * OBS_DIM);
  }

  async init(tracks: TrackSpec[]): Promise<void> {
    await ensureBackend();
    this.seeds = tracks.filter((t) => t.source === 'gen' && t.seed !== undefined).map((t) => t.seed!);
    this.stagePools.clear();
    this.stagePools.set(this.curriculum.stage, tracks);
    this.mix.setPool(tracks);
    this.resetEnvs();
    this.env.sampler = (i) => this.sampleEpisode(i);
    this.ready = true;
  }

  /** Procedural tracks for a curriculum stage, generated once from the pool's seeds. */
  private poolFor(stage: number): TrackSpec[] {
    let pool = this.stagePools.get(stage);
    if (!pool) {
      pool = this.seeds.map((seed) => generateTrack({ seed, stage }));
      this.stagePools.set(stage, pool);
    }
    return pool.length ? pool : this.mix.tracks;
  }

  /** Jump to a curriculum stage. Outside fine-tune mode the track pool becomes that stage's tracks. */
  setStage(stage: number): void {
    this.curriculum.stage = Math.max(0, Math.min(4, Math.round(stage)));
    this.curriculum.resetWindow();
    if (!this.mix.userId && this.seeds.length) this.mix.setPool(this.poolFor(this.curriculum.stage));
  }

  private cfg(track: TrackSpec, salt: number, stage = this.curriculum.stage): EpisodeConfig {
    const rng = makeRng(salt + (track.seed ?? 0) * 7919, `dr/${track.id}`);
    let car = sampleCar(defaultCarParams(), rng, this.dr, this.grip);
    car = { ...car, mass: car.mass * this.massScale };
    car.yawInertia *= this.massScale;
    return {
      stage,
      car,
      noiseScale: noiseFor(rng, this.dr),
      latencyTicks: latencyFor(rng, this.dr),
      trackId: track.id,
      lapsTarget: stage <= 1 ? 1 : 2,
      seed: salt + (track.seed ?? 0),
    };
  }

  /** Track and car for the next episode: 20% replay of earlier stages, the fine-tune mix, fresh DR. */
  private sampleEpisode(slot: number): { track: TrackSpec; cfg: EpisodeConfig } {
    const stage = this.curriculum.sampleStage(this.rng);
    const track = this.mix.userId || stage === this.curriculum.stage ? this.mix.sample(this.rng) : this.replayTrack(stage);
    return { track, cfg: this.cfg(track, ++this.episodeCount * 131 + slot, stage) };
  }

  private replayTrack(stage: number): TrackSpec {
    const pool = this.poolFor(stage);
    return pool[this.rng.int(pool.length)]!;
  }

  /** `all` puts every env on one track (the zero-shot start after a swap); otherwise each draws from the mix. */
  private resetEnvs(all?: TrackSpec): void {
    const tracks: TrackSpec[] = [];
    const cfgs: EpisodeConfig[] = [];
    for (let i = 0; i < this.env.n; i++) {
      const track = all ?? this.mix.sample(this.rng);
      tracks.push(track);
      cfgs.push(this.cfg(track, ++this.episodeCount * 131 + i));
    }
    const obs = this.env.reset(tracks, cfgs);
    this.obs.set(obs);
  }

  /**
   * Hot-swap. `forceSwap` resets every env onto the new pool right away, without touching
   * the weights. SPEC asks for envs to finish their episode first; resetting at once keeps
   * the swap well inside the 15 s budget.
   */
  pendingTracks: TrackSpec[] | null = null;
  pendingFine = false;

  queueTrackPool(tracks: TrackSpec[], fineTune: boolean): void {
    this.pendingTracks = tracks;
    this.pendingFine = fineTune;
  }

  forceSwap(): void {
    if (!this.pendingTracks?.length) return;
    const tracks = this.pendingTracks;
    const before = this.weightHash();
    if (this.pendingFine) {
      const user = tracks[0]!;
      const rest = tracks.slice(1);
      this.mix.startFineTune(user, rest.length ? rest : tracks);
      this.policy.setLr(this.mix.lr);
    } else {
      this.mix.userId = null;
      this.mix.setPool(tracks);
    }
    // After a fine-tune swap every env starts on the user track; later episodes follow the 70/30 mix.
    this.resetEnvs(this.pendingFine ? tracks[0] : undefined);
    if (this.pendingFine) {
      this.adapt = { trackId: tracks[0]!.id, updates: 0, first: [], recent: [], bestLap: null, procBefore: mean(this.procWindow) };
      this.procWindow = [];
    }
    const after = this.weightHash();
    if (before !== after) throw new Error('swap mutated weights');
    this.pendingTracks = null;
  }

  adaptStats(): AdaptStats | null {
    const a = this.adapt;
    if (!a) return null;
    return {
      trackId: a.trackId,
      updatesSinceSwap: a.updates,
      first64: a.first.length >= FIRST_EPISODES ? mean(a.first) : null,
      completion: mean(a.recent),
      bestLap: a.bestLap,
      procBefore: a.procBefore,
      procNow: mean(this.procWindow),
    };
  }

  weightHash(): string {
    return hashBuffers(this.policy.getWeights());
  }

  /**
   * Supervised warm start: regress the mean action toward pure-pursuit on the current pool.
   * Follows up with critic-only PPO updates (see `CRITIC_WARMUP_UPDATES`).
   */
  async imitate(batches = 8): Promise<number> {
    if (!this.ready) throw new Error('trainer not initialized');
    const n = this.env.n;
    const T = this.horizon;
    let last = 0;
    for (let b = 0; b < batches; b++) {
      const obsBuf = new Float32Array(T * n * OBS_DIM);
      const actBuf = new Float32Array(T * n * ACT_DIM);
      for (let t = 0; t < T; t++) {
        obsBuf.set(this.obs, t * n * OBS_DIM);
        const a = this.env.baselineActions();
        actBuf.set(a, t * n * ACT_DIM);
        const stepped = this.env.step(a);
        this.obs.set(stepped.obs);
      }
      for (let e = 0; e < 8; e++) last = this.policy.imitate(obsBuf, actBuf, T * n);
    }
    this.criticWarmup = CRITIC_WARMUP_UPDATES;
    return last;
  }

  /** Deterministic lap completion. Unfinished episodes count as failures. */
  evalCompletion(episodes = 8, maxSteps = 1000): number {
    const tracks: TrackSpec[] = [];
    const cfgs: EpisodeConfig[] = [];
    const rng = makeRng(4242, 'eval');
    for (let i = 0; i < this.env.n; i++) {
      const track = this.mix.sample(rng);
      tracks.push(track);
      cfgs.push({ ...this.cfg(track, 1000 + i), noiseScale: 0, latencyTicks: 0 });
    }
    const sampler = this.env.sampler;
    this.env.sampler = null;
    this.env.reset(tracks, cfgs);
    const obs = new Float32Array(this.env.obs);
    for (let t = 0; t < maxSteps && this.env.finished.length < episodes; t++) {
      const action = this.policy.actMean(obs, this.env.n);
      const stepped = this.env.step(action);
      obs.set(stepped.obs);
    }
    const laps = this.env.finished.slice(0, episodes).filter((e) => e.lap).length;
    this.env.sampler = sampler;
    this.resetEnvs();
    return laps / episodes;
  }

  async update(): Promise<GenStat> {
    if (!this.ready) throw new Error('trainer not initialized');
    const n = this.env.n;
    const T = this.horizon;
    const obsBuf = new Float32Array(T * n * OBS_DIM);
    const preBuf = new Float32Array(T * n * ACT_DIM);
    const logBuf = new Float32Array(T * n);
    const valBuf = new Float32Array(T * n);
    const rewBuf = new Float32Array(T * n);
    const doneBuf = new Uint8Array(T * n);
    const truncBuf = new Uint8Array(T * n);
    const t0 = performance.now();
    let steps = 0;
    const episodes: EpRecord[] = [];
    const replayFrames: number[] = [];
    this.env.finished.length = 0;

    for (let t = 0; t < T; t++) {
      const { action, pre, logp, value } = this.policy.act(this.obs, n);
      obsBuf.set(this.obs, t * n * OBS_DIM);
      preBuf.set(pre, t * n * ACT_DIM);
      logBuf.set(logp, t * n);
      valBuf.set(value, t * n);
      const stepped = this.env.step(action);
      for (let i = 0; i < n; i++) {
        const raw = stepped.rew[i]!;
        this.rewardScale.update(raw);
        rewBuf[t * n + i] = this.rewardScale.scale(raw);
        doneBuf[t * n + i] = stepped.done[i]!;
        truncBuf[t * n + i] = stepped.trunc[i]! & (stepped.done[i]! ^ 1);
      }
      this.obs.set(stepped.obs);
      steps += n;
      if (t % 2 === 0) {
        const pose = this.env.poses();
        for (let k = 0; k < 7; k++) replayFrames.push(pose[k]!);
      }
      this.onStep?.();
      if (this.yieldEvery > 0 && (t + 1) % this.yieldEvery === 0) await new Promise<void>((r) => setTimeout(r, 0));
    }
    for (const ep of this.env.finished) episodes.push(ep);
    this.env.finished.length = 0;

    // GAE. A true termination bootstraps from 0. A time-limit cut has no terminal observation
    // (the env already reset), so it bootstraps from its own value estimate. A rollout that ends
    // mid-episode bootstraps from V(obs after the last step).
    const lastValue = this.policy.valueOf(this.obs, n);
    const adv = new Float32Array(T * n);
    const ret = new Float32Array(T * n);
    for (let i = 0; i < n; i++) {
      let gae = 0;
      for (let t = T - 1; t >= 0; t--) {
        const idx = t * n + i;
        const ended = doneBuf[idx]! || truncBuf[idx]!;
        let nextV: number;
        if (doneBuf[idx]) nextV = 0;
        else if (truncBuf[idx]) nextV = valBuf[idx]!;
        else nextV = t === T - 1 ? lastValue[i]! : valBuf[(t + 1) * n + i]!;
        const delta = rewBuf[idx]! + GAMMA * nextV - valBuf[idx]!;
        gae = delta + GAMMA * LAMBDA * (ended ? 0 : gae);
        adv[idx] = gae;
        ret[idx] = gae + valBuf[idx]!;
      }
    }
    let mean = 0;
    for (let i = 0; i < adv.length; i++) mean += adv[i]!;
    mean /= adv.length;
    let varr = 0;
    for (let i = 0; i < adv.length; i++) varr += (adv[i]! - mean) ** 2;
    const std = Math.sqrt(varr / adv.length) + 1e-8;
    for (let i = 0; i < adv.length; i++) adv[i] = (adv[i]! - mean) / std;

    const total = T * n;
    const policyScale = this.criticWarmup > 0 ? 0 : 1;
    if (this.criticWarmup > 0) this.criticWarmup--;
    const order = new Uint32Array(total);
    const mObs = new Float32Array(this.minibatch * OBS_DIM);
    const mPre = new Float32Array(this.minibatch * ACT_DIM);
    const mLog = new Float32Array(this.minibatch);
    const mAdv = new Float32Array(this.minibatch);
    const mRet = new Float32Array(this.minibatch);
    let kl = 0;
    let ent = 0;
    let vloss = 0;
    let loss = 0;
    let updates = 0;
    for (let e = 0; e < this.epochs; e++) {
      for (let i = 0; i < total; i++) order[i] = i;
      for (let i = total - 1; i > 0; i--) {
        const j = this.rng.int(i + 1);
        const tmp = order[i]!;
        order[i] = order[j]!;
        order[j] = tmp;
      }
      for (let start = 0; start < total; start += this.minibatch) {
        const bs = Math.min(this.minibatch, total - start);
        for (let k = 0; k < bs; k++) {
          const src = order[start + k]!;
          mObs.set(obsBuf.subarray(src * OBS_DIM, (src + 1) * OBS_DIM), k * OBS_DIM);
          mPre.set(preBuf.subarray(src * ACT_DIM, (src + 1) * ACT_DIM), k * ACT_DIM);
          mLog[k] = logBuf[src]!;
          mAdv[k] = adv[src]!;
          mRet[k] = ret[src]!;
        }
        const stats = this.policy.ppoLoss(
          mObs.subarray(0, bs * OBS_DIM),
          mPre.subarray(0, bs * ACT_DIM),
          mLog.subarray(0, bs),
          mAdv.subarray(0, bs),
          mRet.subarray(0, bs),
          bs,
          this.clip,
          this.mix.entropy,
          policyScale,
        );
        kl += stats.kl;
        ent += stats.entropy;
        vloss += stats.valueLoss;
        loss += stats.loss;
        updates++;
      }
    }
    this.mix.onUpdate();
    if (this.adapt) this.adapt.updates++;
    for (const ep of episodes) {
      if (this.adapt && ep.trackId === this.adapt.trackId) {
        if (this.adapt.first.length < FIRST_EPISODES) this.adapt.first.push(ep.lap);
        this.adapt.recent.push(ep.lap);
        if (this.adapt.recent.length > WINDOW) this.adapt.recent.shift();
        if (ep.lapTime && (this.adapt.bestLap === null || ep.lapTime < this.adapt.bestLap)) this.adapt.bestLap = ep.lapTime;
      } else {
        this.procWindow.push(ep.lap);
        if (this.procWindow.length > WINDOW) this.procWindow.shift();
      }
      this.curriculum.note({ completed: ep.lap, lapTime: ep.lapTime, tRef: ep.tRef });
      if (ep.lapTime && (this.bestLap === null || ep.lapTime < this.bestLap)) this.bestLap = ep.lapTime;
    }
    if (this.curriculum.maybePromote() && !this.mix.userId) this.mix.setPool(this.poolFor(this.curriculum.stage));
    if (this.pendingTracks) this.forceSwap();
    this.replays.set(this.gen + 1, Float32Array.from(replayFrames));
    this.gen++;
    const stat = aggregateGen(this.gen, episodes, this.env.trackId(0));
    stat.entropy = updates ? ent / updates : 0;
    stat.kl = updates ? kl / updates : 0;
    stat.valueLoss = updates ? vloss / updates : 0;
    const dt = (performance.now() - t0) / 1000;
    stat.stepsPerSec = dt > 0 ? steps / dt : 0;
    void loss;
    this.history.push(stat);
    return stat;
  }
}
