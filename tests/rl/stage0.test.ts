import { describe, expect, it } from 'vitest';
import { Trainer } from '../../src/rl/trainer';
import { generateTrack } from '../../src/track/generate';
import { buildTrackFromPoints } from '../../src/track/build';
import { BROWSER_TUNE, configureFinetune, trainRecipe } from '../../src/rl/recipe';
import { STAGE0_HELD_OUT, DEFAULT_TRAIN_SEEDS, assertTrainSeeds } from '../../src/rl/seeds';
import { casesFor, evaluateCases } from '../../src/eval/harness';

describe('stage 0 training', () => {
  it('reaches 80% lap completion on held-out seeds within the 3 minute budget', async () => {
    const t0 = performance.now();
    // Stage-0 recipe: DAgger imitation of pure pursuit, then a short validation-gated PPO fine-tune.
    const { trainer, report } = await trainRecipe({ maxStage: 0, ppoUpdates: 2 });
    const seconds = (performance.now() - t0) / 1000;
    const tracks = STAGE0_HELD_OUT.map((seed) => generateTrack({ seed, stage: 0 }));
    const out = evaluateCases((o, n) => trainer.policy.actMean(o, n), casesFor(tracks));
    expect(seconds).toBeLessThan(180);
    expect(out.rate).toBeGreaterThanOrEqual(0.8);
    // PPO is gated: it may not leave the policy worse (on validation seeds) than imitation alone.
    const ppo = report.ppo!;
    expect(ppo.endVal!.rate).toBeGreaterThanOrEqual(ppo.startVal!.rate);
  }, 180_000);

  it('is deterministic: same seeds, same weights', async () => {
    const tune = { roundSteps: 60, beta0: [1, 0], fitFirst: 6, fitRound: 4, fitFinal: 4, batch: 32, ppoUpdates: 1, criticUpdates: 1, ppoHorizon: 40 };
    const a = await trainRecipe({ maxStage: 0, n: 4, tune });
    const b = await trainRecipe({ maxStage: 0, n: 4, tune });
    expect(a.trainer.weightHash()).toBe(b.trainer.weightHash());
    const c = await trainRecipe({ maxStage: 0, n: 4, tune, seed: 2 });
    expect(c.trainer.weightHash()).not.toBe(a.trainer.weightHash());
  }, 120_000);

  it('refuses training seeds that overlap evaluation seeds', () => {
    expect(() => assertTrainSeeds([1, 2, 101])).toThrow();
    expect(() => assertTrainSeeds(DEFAULT_TRAIN_SEEDS)).not.toThrow();
  });

  it('keeps the browser policy through further PPO updates', async () => {
    // The learner worker keeps running PPO after its imitation start (see learner.worker.ts);
    // those updates must not wreck the policy before the user draws anything.
    const { trainer } = await trainRecipe({ maxStage: 0, ppoUpdates: 0, validate: false, tune: BROWSER_TUNE });
    configureFinetune(trainer);
    for (let i = 0; i < 6; i++) await trainer.update();
    const tracks = STAGE0_HELD_OUT.slice(0, 4).map((seed) => generateTrack({ seed, stage: 0 }));
    const out = evaluateCases((obs, n) => trainer.policy.actMean(obs, n), casesFor(tracks, { copies: 2 }));
    expect(out.rate).toBeGreaterThanOrEqual(0.7);
  }, 180_000);

  it('swaps the track pool without changing weights, with every env starting on the new track', async () => {
    const tracks = [generateTrack({ seed: 2, stage: 0 })];
    const trainer = new Trainer({ n: 4, horizon: 16, epochs: 1, minibatch: 64 });
    await trainer.init(tracks);
    const before = trainer.weightHash();
    const custom = buildTrackFromPoints(
      Float32Array.from([0, 0, 8, 0, 12, 4, 8, 8, 0, 8, -4, 4, 0.2, 0.1]),
      { halfWidth: 1.2, mu: 1, source: 'user', id: 'user-s' },
    );
    trainer.queueTrackPool([custom, tracks[0]!], true);
    trainer.forceSwap();
    expect(trainer.weightHash()).toBe(before);
    expect(trainer.env.trackIds()).toEqual(['user-s', 'user-s', 'user-s', 'user-s']);
    expect(trainer.mix.entropy).toBe(0.02);
    expect(trainer.mix.weights[0]).toBeCloseTo(0.7, 2);
  });

  it('keeps using a swapped-in pool after scenarios were set', async () => {
    const trainer = new Trainer({ n: 4, horizon: 16, epochs: 1, minibatch: 64 });
    const { ScenarioSet } = await import('../../src/rl/scenarios');
    const set = new ScenarioSet([1, 2, 3]);
    await trainer.init(set.tracks(0));
    trainer.setScenarios(set);
    expect(trainer.scenarioActive).toBe(true);
    const custom = buildTrackFromPoints(
      Float32Array.from([0, 0, 8, 0, 12, 4, 8, 8, 0, 8, -4, 4, 0.2, 0.1]),
      { halfWidth: 1.2, mu: 1, source: 'user', id: 'user-s2' },
    );
    trainer.queueTrackPool([custom], false);
    trainer.forceSwap();
    expect(trainer.scenarioActive).toBe(false);
    expect(trainer.env.trackId(0)).toBe(custom.id);
  });
});
