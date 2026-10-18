import { ACT_DIM, OBS_DIM, type EpisodeConfig, type GenStat, type TrackSpec } from '../shared/types';
import { hashBuffers } from '../shared/hash';
import { makeRng } from '../shared/rng';
import { VecEnv } from '../sim/vecEnv';
import { defaultCarParams } from '../sim/car/params';
import { latencyFor, noiseFor, sampleCar, type DRLevel } from '../sim/dr';
import { GaussianPolicy, ensureBackend } from './model';
import { episodeConfig, type ScenarioSet } from './scenarios';
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
  /** Seeds weight init, exploration noise, episode DR and scenario sampling. */
  seed?: number;
  lr?: number;
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
  /** Multiplies the entropy bonus (TrackMix owns the base value). */
  entropyScale = 1;
  /** In scenario mode, redraw every slot's scenario at the start of each update. */
  freshEpisodesPerUpdate = true;
  /** false trains the value head only (critic warm-up before PPO touches an imitation policy). */
  trainPolicy = true;
  readonly seed: number;
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
    this.seed = opts.seed ?? 1;
    this.policy = new GaussianPolicy(opts.lr ?? 3e-4, this.seed);
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

  /**
   * Stage pools with per-stage DR and surface mixes. While set (and until a hot-swap replaces
   * the TrackMix pool), `resetEnvs` draws scenarios from here using `curriculum.stage`.
   */
  scenarios: ScenarioSet | null = null;
  private scenarioPool: TrackSpec[] | null = null;
  private resets = 0;

  setScenarios(set: ScenarioSet): void {
    this.scenarios = set;
    this.mix.setPool(set.tracks(0));
    this.scenarioPool = this.mix.tracks;
  }

  /** True while scenario sampling (not a swapped-in TrackMix pool) drives the env. */
  get scenarioActive(): boolean {
    return !!this.scenarios && this.scenarioPool === this.mix.tracks;
  }

  private cfg(track: TrackSpec, i: number): EpisodeConfig {
    const rng = makeRng((this.gen + 1) * 10007 + i * 13 + (track.seed ?? 0) + this.seed * 7919, `dr/${track.id}`);
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

  /** Fresh episodes on every slot. Public so the recipe can draw new scenarios per rollout. */
  resetEnvs(): void {
    const tracks: TrackSpec[] = [];
    const cfgs: EpisodeConfig[] = [];
    this.resets++;
    const scenario = this.scenarioActive;
    // The TrackMix path keeps the original draw (gen + 3) so swap behaviour is unchanged.
    const rng = makeRng(scenario ? this.seed * 977 + this.gen + this.resets * 31 + 3 : this.gen + 3, 'pool');
    for (let i = 0; i < this.env.n; i++) {
      if (scenario) {
        // A DR level chosen in the UI (setDR) overrides the stage profile.
        const sc = this.scenarios!.sample(rng, Math.min(3, this.curriculum.stage), this.dr !== 'off' ? { dr: this.dr } : undefined);
        tracks.push(sc.track);
        cfgs.push(
          episodeConfig(sc, this.seed * 7919 + this.resets * 101 + this.gen + 1, i, {
            grip: this.grip,
            massScale: this.massScale,
          }),
        );
      } else {
        const track = this.mix.sample(rng);
        tracks.push(track);
        cfgs.push(this.cfg(track, i));
      }
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
    // Scenario mode: every rollout starts from fresh tracks / DR / surfaces so DR is resampled per episode.
    if (this.scenarioActive && this.freshEpisodesPerUpdate) this.resetEnvs();

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
    // Bootstrap the rollout boundary with V(s_T) instead of treating it as terminal.
    const lastValue = this.policy.act(this.obs, n).value;
    for (let i = 0; i < n; i++) {
      let gae = 0;
      for (let t = T - 1; t >= 0; t--) {
        const idx = t * n + i;
        const nextV = doneBuf[idx] ? 0 : t === T - 1 ? lastValue[i]! : valBuf[(t + 1) * n + i]!;
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
    const shuffle = makeRng(this.seed * 313 + this.gen, 'ppo/shuffle');
    const perm = new Int32Array(total);
    const mo = new Float32Array(this.minibatch * OBS_DIM);
    const ma = new Float32Array(this.minibatch * ACT_DIM);
    const mlp = new Float32Array(this.minibatch);
    const mad = new Float32Array(this.minibatch);
    const mrt = new Float32Array(this.minibatch);
    for (let e = 0; e < this.epochs; e++) {
      for (let i = 0; i < total; i++) perm[i] = i;
      for (let i = total - 1; i > 0; i--) {
        const j = shuffle.int(i + 1);
        const tmp = perm[i]!;
        perm[i] = perm[j]!;
        perm[j] = tmp;
      }
      for (let start = 0; start < total; start += this.minibatch) {
        const bs = Math.min(this.minibatch, total - start);
        for (let k = 0; k < bs; k++) {
          const idx = perm[start + k]!;
          mo.set(obsBuf.subarray(idx * OBS_DIM, (idx + 1) * OBS_DIM), k * OBS_DIM);
          ma.set(actBuf.subarray(idx * ACT_DIM, (idx + 1) * ACT_DIM), k * ACT_DIM);
          mlp[k] = logBuf[idx]!;
          mad[k] = adv[idx]!;
          mrt[k] = ret[idx]!;
        }
        const stats = this.policy.ppoLoss(
          mo.subarray(0, bs * OBS_DIM),
          ma.subarray(0, bs * ACT_DIM),
          mlp.subarray(0, bs),
          mad.subarray(0, bs),
          mrt.subarray(0, bs),
          bs,
          this.clip,
          this.mix.entropy * this.entropyScale,
          this.trainPolicy,
        );
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
