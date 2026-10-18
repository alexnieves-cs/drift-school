import { ACT_DIM, OBS_DIM, type EpisodeConfig, type GenStat, type TrackSpec } from '../shared/types';
import { hashBuffers } from '../shared/hash';
import { makeRng } from '../shared/rng';
import { VecEnv } from '../sim/vecEnv';
import { defaultCarParams } from '../sim/car/params';
import { latencyFor, noiseFor, sampleCar, type DRLevel } from '../sim/dr';
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
}

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
  bestLap: number | null = null;
  readonly history: GenStat[] = [];
  /** Best-lap pose tape per generation, for leaderboard replay. */
  readonly replays = new Map<number, Float32Array>();

  constructor(opts: TrainerOpts = {}) {
    const n = opts.n ?? 16;
    this.env = new VecEnv(n);
    this.policy = new GaussianPolicy(3e-4);
    this.horizon = opts.horizon ?? 64;
    this.epochs = opts.epochs ?? 3;
    this.minibatch = opts.minibatch ?? 256;
    this.clip = opts.clip ?? 0.2;
    this.obs = new Float32Array(n * OBS_DIM);
  }

  async init(tracks: TrackSpec[]): Promise<void> {
    await ensureBackend();
    this.mix.setPool(tracks);
    this.resetEnvs();
    this.ready = true;
  }

  private cfg(track: TrackSpec, i: number): EpisodeConfig {
    const rng = makeRng((this.gen + 1) * 10007 + i * 13 + (track.seed ?? 0), `dr/${track.id}`);
    let car = sampleCar(defaultCarParams(), rng, this.dr, this.grip);
    car = { ...car, mass: car.mass * this.massScale };
    car.yawInertia *= this.massScale;
    return {
      stage: this.curriculum.stage,
      car,
      noiseScale: noiseFor(rng, this.dr),
      latencyTicks: latencyFor(rng, this.dr),
      trackId: track.id,
      lapsTarget: this.curriculum.stage <= 1 ? 1 : 2,
      seed: (track.seed ?? 0) + i + this.gen,
    };
  }

  private resetEnvs(): void {
    const tracks: TrackSpec[] = [];
    const cfgs: EpisodeConfig[] = [];
    const rng = makeRng(this.gen + 3, 'pool');
    for (let i = 0; i < this.env.n; i++) {
      const stage = this.curriculum.sampleStage(rng);
      void stage;
      const track = this.mix.sample(rng);
      tracks.push(track);
      cfgs.push(this.cfg(track, i));
    }
    const obs = this.env.reset(tracks, cfgs);
    this.obs.set(obs);
  }

  /**
   * Hot-swap. In-flight episodes keep rolling; callers should invoke
   * `applySwapWhenReady` after the current episodes terminate, or call
   * `forceSwap` which resets immediately without touching weights.
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
    if (this.pendingFine) {
      const user = tracks[0]!;
      const rest = tracks.slice(1);
      this.mix.startFineTune(user, rest.length ? rest : tracks);
      this.policy.setLr(this.mix.lr);
    } else {
      this.mix.setPool(tracks);
    }
    const before = this.weightHash();
    this.resetEnvs();
    const after = this.weightHash();
    if (before !== after) throw new Error('swap mutated weights');
    this.pendingTracks = null;
  }

  // --- Hot-swap support used by rl/finetune.ts (acc/swap). Additive; nothing above changed. ---

  /** Episode config for `track` in env slot `i`, same DR draw `resetEnvs` uses. */
  configFor(track: TrackSpec, i: number): EpisodeConfig {
    return this.cfg(track, i);
  }

  /** Rollout without learning: sampled-policy steps, no gradient update, weights untouched. */
  rolloutNoUpdate(steps: number): void {
    const n = this.env.n;
    for (let t = 0; t < steps; t++) {
      const { action } = this.policy.act(this.obs, n);
      const stepped = this.env.step(action);
      this.obs.set(stepped.obs);
    }
  }

  /** Re-read the env's current observation after something else stepped or retargeted it. */
  syncObs(): void {
    this.obs.set(this.env.obs);
  }

  weightHash(): string {
    return hashBuffers(this.policy.getWeights());
  }

  /** Supervised warm start: regress the mean action toward pure-pursuit on the current pool. */
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
      for (let e = 0; e < 3; e++) last = this.policy.imitate(obsBuf, actBuf, T * n);
    }
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
    this.env.reset(tracks, cfgs);
    const obs = new Float32Array(this.env.obs);
    for (let t = 0; t < maxSteps && this.env.finished.length < episodes; t++) {
      const action = this.policy.actMean(obs, this.env.n);
      const stepped = this.env.step(action);
      obs.set(stepped.obs);
    }
    const laps = this.env.finished.slice(0, episodes).filter((e) => e.lap).length;
    return laps / episodes;
  }

  async update(): Promise<GenStat> {
    if (!this.ready) throw new Error('trainer not initialized');
    const n = this.env.n;
    const T = this.horizon;
    const obsBuf = new Float32Array(T * n * OBS_DIM);
    const actBuf = new Float32Array(T * n * ACT_DIM);
    const logBuf = new Float32Array(T * n);
    const valBuf = new Float32Array(T * n);
    const rewBuf = new Float32Array(T * n);
    const doneBuf = new Uint8Array(T * n);
    const t0 = performance.now();
    let steps = 0;
    const episodes: EpRecord[] = [];
    const replayFrames: number[] = [];

    for (let t = 0; t < T; t++) {
      const { action, logp, value } = this.policy.act(this.obs, n);
      obsBuf.set(this.obs, t * n * OBS_DIM);
      actBuf.set(action, t * n * ACT_DIM);
      logBuf.set(logp, t * n);
      valBuf.set(value, t * n);
      const before = this.env.finished.length;
      const stepped = this.env.step(action);
      for (let k = before; k < this.env.finished.length; k++) episodes.push(this.env.finished[k]!);
      for (let i = 0; i < n; i++) {
        const raw = stepped.rew[i]!;
        this.rewardScale.update(raw);
        rewBuf[t * n + i] = this.rewardScale.scale(raw);
        doneBuf[t * n + i] = stepped.done[i]! | stepped.trunc[i]!;
      }
      this.obs.set(stepped.obs);
      steps += n;
      if (t % 2 === 0) {
        const pose = this.env.poses();
        const base = 0;
        for (let k = 0; k < 7; k++) replayFrames.push(pose[base + k]!);
      }
    }

    const adv = new Float32Array(T * n);
    const ret = new Float32Array(T * n);
    const gamma = 0.99;
    const lam = 0.95;
    for (let i = 0; i < n; i++) {
      let gae = 0;
      for (let t = T - 1; t >= 0; t--) {
        const idx = t * n + i;
        const nextV = t === T - 1 || doneBuf[idx] ? 0 : valBuf[(t + 1) * n + i]!;
        const delta = rewBuf[idx]! + gamma * nextV - valBuf[idx]!;
        gae = delta + gamma * lam * (doneBuf[idx] ? 0 : gae);
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
    let kl = 0;
    let ent = 0;
    let vloss = 0;
    let loss = 0;
    let updates = 0;
    for (let e = 0; e < this.epochs; e++) {
      for (let start = 0; start < total; start += this.minibatch) {
        const bs = Math.min(this.minibatch, total - start);
        const o = obsBuf.subarray(start * OBS_DIM, (start + bs) * OBS_DIM);
        const a = actBuf.subarray(start * ACT_DIM, (start + bs) * ACT_DIM);
        const lp = logBuf.subarray(start, start + bs);
        const ad = adv.subarray(start, start + bs);
        const rt = ret.subarray(start, start + bs);
        const stats = this.policy.ppoLoss(o, a, lp, ad, rt, bs, this.clip, this.mix.entropy);
        kl += stats.kl;
        ent += stats.entropy;
        vloss += stats.valueLoss;
        loss += stats.loss;
        updates++;
      }
    }
    this.mix.onUpdate();
    for (const ep of episodes) {
      this.curriculum.note({ completed: ep.lap, lapTime: ep.lapTime, tRef: this.mix.tracks[0]?.tRef ?? 10 });
      if (ep.lapTime && (this.bestLap === null || ep.lapTime < this.bestLap)) this.bestLap = ep.lapTime;
    }
    this.curriculum.maybePromote();
    if (this.pendingTracks) {
      const still = this.env.finished.length;
      void still;
      this.forceSwap();
    }
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
