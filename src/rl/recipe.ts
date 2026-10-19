import { makeRng } from '../shared/rng';
import { Trainer } from './trainer';
import { ScenarioSet } from './scenarios';
import { Dataset, collect, fit } from './dagger';
import { DEFAULT_TRAIN_SEEDS, VALIDATION_SEEDS, assertTrainSeeds } from './seeds';
import { buildSuite, validate, type ValResult, type ValSuite } from './validate';

/**
 * The training recipe the app ships and `npm run eval` measures. One function, used by both.
 *
 * WHAT IS IMITATION AND WHAT IS PPO (read this before quoting any number from it):
 *
 *  1. Phase "dagger": the policy is distilled from the pure-pursuit + speed-profile baseline
 *     (src/track/baseline.ts) with DAgger, stage by stage (SPEC stages 0..3: stage-appropriate
 *     track pools, DR off/low/high, mixed surfaces). Essentially ALL of the driving skill the
 *     eval measures comes from this phase. The baseline itself completes ~100% of stage 0-3
 *     tracks, so the distilled policy is bounded by it and is not "discovered" by RL.
 *  2. Phase "ppo": a short, conservative PPO fine-tune (critic warm-up first, small LR, tight clip,
 *     grad-norm clip, validation-gated: any update that scores worse on held-out validation
 *     seeds is reverted). It is real PPO on the SPEC reward, but its measured contribution is
 *     small (see `report.ppo`); it can improve lap time at equal completion, it does not
 *     create the lap-completing behaviour.
 *
 * Fully seeded: no Math.random, no TF.js RNG, no wall-clock-dependent branching. The same
 * `seed` + `seeds` give identical weights.
 */

export interface RecipeOpts {
  /** Training track seeds (must not overlap evaluation / validation seeds). */
  seeds?: number[];
  /** Parallel envs for data collection and PPO. */
  n?: number;
  /** Seeds init, sampling and scenario draws. */
  seed?: number;
  /** Highest SPEC stage to train (0..3). Stage 0 alone is the M2 recipe. */
  maxStage?: number;
  /** PPO fine-tune updates after imitation (0 disables PPO). */
  ppoUpdates?: number;
  /** Run the held-out validation suites (needed for PPO gating and the report). Default true. */
  validate?: boolean;
  onStatus?: (text: string) => void;
  /** Schedule sizes. Defaults are what ships; tests and experiments override them. */
  tune?: Partial<RecipeTune>;
}

export interface RecipeTune {
  /** Env steps per DAgger collection round (x n envs). */
  roundSteps: number;
  /** DAgger beta (expert-drive probability) per round, for stage 0 and for later stages. */
  beta0: number[];
  betaN: number[];
  /** Gradient steps after each round: first round of stage 0, other rounds. */
  fitFirst: number;
  fitRound: number;
  /** Final polish gradient steps over the whole aggregate. */
  fitFinal: number;
  batch: number;
  ppoUpdates: number;
  criticUpdates: number;
  ppoHorizon: number;
  /** Validate (and keep or revert) every this many PPO updates. */
  gateEvery: number;
}

export const DEFAULT_TUNE: RecipeTune = {
  roundSteps: 600,
  beta0: [1, 0.5, 0.2, 0],
  betaN: [0.3, 0],
  fitFirst: 200,
  fitRound: 80,
  fitFinal: 150,
  batch: 256,
  ppoUpdates: 4,
  criticUpdates: 2,
  ppoHorizon: 160,
  gateEvery: 2,
};

export interface PhaseReport {
  name: string;
  seconds: number;
  /** Validation after the phase (stages trained so far). */
  val?: ValResult;
  note?: string;
}

export interface RecipeReport {
  phases: PhaseReport[];
  /** Wall clock when the stage-0 phase finished and its validation completion on stage-0 tracks. */
  stage0?: { seconds: number; valRate: number; weights: ArrayBuffer[] };
  ppo?: { proposed: number; accepted: number; reverted: number; startVal?: ValResult; endVal?: ValResult };
  /** Final dataset size. */
  samples: number;
}

export interface RecipeResult {
  trainer: Trainer;
  seconds: number;
  report: RecipeReport;
}

const now = () => performance.now() / 1000;

export async function trainRecipe(opts: RecipeOpts = {}): Promise<RecipeResult> {
  const t0 = now();
  const seeds = opts.seeds ?? DEFAULT_TRAIN_SEEDS;
  assertTrainSeeds(seeds);
  const seed = opts.seed ?? 1;
  const maxStage = Math.max(0, Math.min(3, opts.maxStage ?? 3));
  const status = opts.onStatus ?? (() => undefined);
  const tune: RecipeTune = { ...DEFAULT_TUNE, ...opts.tune };
  const report: RecipeReport = { phases: [], samples: 0 };
  const doValidate = opts.validate ?? true;

  const trainer = new Trainer({ n: opts.n ?? 16, horizon: 400, epochs: 2, minibatch: 512, seed, lr: 3e-5 });
  const scenarios = new ScenarioSet(seeds);
  await trainer.init(scenarios.tracks(0));
  trainer.setScenarios(scenarios);

  const rng = makeRng(seed, 'recipe');
  const ds = new Dataset(160_000);
  const suiteStages = Array.from({ length: maxStage + 1 }, (_, i) => i);
  const suite = buildSuite(VALIDATION_SEEDS, suiteStages);
  const act = (o: Float32Array, n: number) => trainer.policy.actMean(o, n);

  for (let stage = 0; stage <= maxStage; stage++) {
    const ph0 = now();
    status(`imitating pure pursuit, stage ${stage}`);
    trainer.curriculum.stage = stage;
    const betas = stage === 0 ? tune.beta0 : tune.betaN;
    for (let r = 0; r < betas.length; r++) {
      collect(trainer, ds, {
        steps: tune.roundSteps,
        beta: betas[r]!,
        actionNoise: 0.15,
        noisyFrac: 0.5,
        chunk: 25,
        rng,
      });
      fit(trainer.policy, ds, {
        steps: stage === 0 && r === 0 ? tune.fitFirst : tune.fitRound,
        batch: tune.batch,
        lr0: stage === 0 ? 2e-3 : 1e-3,
        lr1: stage === 0 ? 3e-4 : 2e-4,
        rng,
      });
    }
    // Stage 0 gets its own validation so the M2 clock can be reported at the end of that phase.
    const stage0Val =
      stage === 0 && doValidate && maxStage > 0 ? validate(act, buildSuite(VALIDATION_SEEDS, [0])) : undefined;
    report.phases.push({
      name: `dagger stage ${stage}`,
      seconds: now() - ph0,
      val: stage0Val,
    });
    // Snapshot for the M2 check: the stage-0 curriculum phase alone, with the wall clock it took.
    if (stage === 0) report.stage0 = { seconds: now() - t0, valRate: stage0Val?.rate ?? -1, weights: trainer.policy.getWeights() };
  }
  fit(trainer.policy, ds, { steps: tune.fitFinal, batch: tune.batch, lr0: 5e-4, lr1: 5e-5, rng });
  report.samples = ds.size;
  const ppoUpdates = opts.ppoUpdates ?? tune.ppoUpdates;
  const afterImitation = doValidate || ppoUpdates > 0 ? validate(act, suite) : undefined;
  if (afterImitation) {
    report.phases.push({ name: 'dagger final polish', seconds: 0, val: afterImitation });
    if (maxStage === 0 && report.stage0) report.stage0.valRate = afterImitation.rate;
  }
  if (afterImitation && ppoUpdates > 0) {
    status('PPO fine-tune');
    const p0 = now();
    report.ppo = await ppoPhase(trainer, suite, afterImitation, ppoUpdates, tune, maxStage);
    report.phases.push({ name: 'ppo', seconds: now() - p0, val: report.ppo.endVal });
  }
  configureFinetune(trainer);
  return { trainer, seconds: now() - t0, report };
}

/**
 * Short schedule for the browser worker's init (the page must reach "racing" in seconds, not minutes):
 * stage 0 only, fewer rounds and gradient steps. Same code path as the eval recipe.
 */
export const BROWSER_TUNE: Partial<RecipeTune> = {
  roundSteps: 300,
  beta0: [1, 0.4, 0],
  fitFirst: 150,
  fitRound: 60,
  fitFinal: 60,
};

/** PPO settings for fine-tuning an imitation policy. Also what the browser's continuing loop uses. */
export function configureFinetune(trainer: Trainer): void {
  trainer.policy.setLr(PPO_LR);
  trainer.policy.setLogStd(PPO_LOGSTD, PPO_LOGSTD);
  trainer.clip = 0.1;
  trainer.entropyScale = 0;
  trainer.epochs = 1;
  trainer.minibatch = 512;
  trainer.trainPolicy = true;
}

const PPO_LR = 3e-5;
const PPO_LOGSTD = -2.3;

/** More completions, or the same completions with a mean lap at least 0.5% faster. Anything else is reverted. */
const better = (a: ValResult, b: ValResult) =>
  a.rate > b.rate || (a.rate === b.rate && a.meanLap > 0 && a.meanLap < b.meanLap * 0.995);

/**
 * Critic warm-up, then validation-gated PPO. An update that does not score at least as well as the
 * best weights so far on held-out validation seeds is reverted, so PPO can only keep or improve
 * what imitation produced.
 */
async function ppoPhase(
  trainer: Trainer,
  suite: ValSuite,
  start: ValResult,
  updates: number,
  tune: RecipeTune,
  maxStage: number,
): Promise<NonNullable<RecipeReport['ppo']>> {
  const act = (o: Float32Array, n: number) => trainer.policy.actMean(o, n);
  configureFinetune(trainer);
  trainer.curriculum.stage = maxStage;
  trainer.horizon = tune.ppoHorizon;
  // Critic first: with a random value head the first policy gradients are noise that wrecks the policy.
  trainer.trainPolicy = false;
  for (let i = 0; i < tune.criticUpdates; i++) await trainer.update();
  trainer.trainPolicy = true;
  let best = start;
  let bestW = trainer.policy.getWeights();
  let accepted = 0;
  let reverted = 0;
  for (let k = 0; k < updates; k++) {
    await trainer.update();
    if ((k + 1) % tune.gateEvery !== 0 && k + 1 !== updates) continue;
    const v = validate(act, suite);
    const group = Math.min(tune.gateEvery, k + 1 - (accepted + reverted));
    if (better(v, best)) {
      best = v;
      bestW = trainer.policy.getWeights();
      accepted += group;
    } else {
      trainer.policy.setWeights(bestW);
      reverted += group;
    }
  }
  return { proposed: updates, accepted, reverted, startVal: start, endVal: best };
}

/** Name kept for callers written against the first recipe. */
export const trainBaseRecipe = trainRecipe;
