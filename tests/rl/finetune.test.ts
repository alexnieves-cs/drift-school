import { describe, expect, it } from 'vitest';
import { FineTune, retargetSlot, tapEnvStep } from '../../src/rl/finetune';
import { Trainer } from '../../src/rl/trainer';
import { generateTrack } from '../../src/track/generate';
import { processStroke, sCurveStroke } from '../../src/editor/pipeline';
import { quantizePoints } from '../../src/editor/share';
import { toFloat } from '../../src/editor/stroke';
import { buildTrackFromPoints } from '../../src/track/build';

const procedural = [1, 3].map((seed) => generateTrack({ seed, stage: 0 }));
const user = buildTrackFromPoints(quantizePoints(toFloat(processStroke(sCurveStroke(), 1, 1.05).points)), {
  halfWidth: 1,
  mu: 1.05,
  source: 'user',
});

async function makeTrainer(n = 4): Promise<Trainer> {
  const t = new Trainer({ n, horizon: 16, epochs: 1, minibatch: 64 });
  await t.init(procedural);
  return t;
}

describe('retargetSlot', () => {
  it('moves one env onto a new track and restarts its episode there, leaving the others alone', async () => {
    const t = await makeTrainer();
    const before = t.env.trackIds();
    retargetSlot(t.env, 2, user, t.configFor(user, 2));
    const after = t.env.trackIds();
    expect(after[2]).toBe(user.id);
    for (const i of [0, 1, 3]) expect(after[i]).toBe(before[i]);
    // The restarted car sits at the start line of the new track.
    const snap = t.env.snapshot(2)!;
    const s0 = user.samples[user.startIndex]!;
    expect(Math.hypot(snap.x - s0.x, snap.z - s0.z)).toBeLessThan(0.5);
    // It still steps.
    t.env.step(new Float32Array(t.env.n * 2));
  });

  it('tapEnvStep calls the hook after every step and removes itself', async () => {
    const t = await makeTrainer(2);
    let calls = 0;
    const untap = tapEnvStep(t.env, () => calls++);
    t.env.step(new Float32Array(4));
    t.env.step(new Float32Array(4));
    untap();
    t.env.step(new Float32Array(4));
    expect(calls).toBe(2);
  });
});

describe('FineTune', () => {
  it('lets each env finish its episode, then lands all of them on the user track; weights stay byte-identical', async () => {
    const t = await makeTrainer();
    await t.update(); // some history, so the weights are not the initial ones
    const hash = t.weightHash();
    const ft = new FineTune(t, { user, procedural, drainMs: 60_000, zeroShotEpisodes: 4, zeroShotMs: 60_000 });
    ft.start();
    expect(ft.phase).toBe('drain');
    let steps = 0;
    const seen = new Set<number>();
    while (ft.phase === 'drain' && steps++ < 4000) {
      await ft.step();
      seen.add(ft.report().envsOnUser);
    }
    const rep = ft.report();
    expect(rep.latencyMs).not.toBeNull();
    expect(rep.envsOnUserAtSwap).toBe(t.env.n);
    expect(t.env.trackIds().every((id) => id === user.id)).toBe(true);
    expect(rep.weightHashAtDrain).toBe(hash);
    // Envs arrived one by one as their episodes ended, not all at once.
    expect(seen.size).toBeGreaterThan(1);
    ft.stop();
  }, 60_000);

  it('retargets stragglers at the deadline so the swap is bounded', async () => {
    const t = await makeTrainer();
    let clock = 0;
    const ft = new FineTune(t, { user, procedural, drainMs: 100, now: () => clock });
    ft.start();
    expect(ft.phase).toBe('drain');
    clock = 101;
    await ft.step();
    expect(ft.report().envsOnUserAtSwap).toBe(t.env.n);
    expect(ft.phase).toBe('zeroShot');
    ft.stop();
  });

  it('runs the zero-shot episodes without an update, then fine-tunes on a 70/30 pool', async () => {
    const t = await makeTrainer();
    const hash = t.weightHash();
    const ft = new FineTune(t, { user, procedural, drainMs: 0, zeroShotEpisodes: 6, zeroShotMs: 60_000, imitationPerPpo: 1 });
    ft.start();
    let guard = 0;
    while (ft.phase !== 'tune' && guard++ < 4000) await ft.step();
    const rep = ft.report();
    expect(ft.phase).toBe('tune');
    expect(rep.zeroShot?.episodes).toBeGreaterThanOrEqual(6);
    expect(rep.weightsUnchanged).toBe(true);
    expect(rep.weightHashAtSwap).toBe(hash);
    expect(t.mix.entropy).toBe(0.02);
    expect(t.mix.weights[0]).toBeCloseTo(0.7, 5);
    expect(t.mix.tracks[0]!.id).toBe(user.id);
    // One tune step is DAgger batches plus a PPO update, and it changes the weights.
    const after = await ft.step();
    expect(after.ppoUpdates).toBe(1);
    expect(after.imitationBatches).toBe(1);
    expect(t.weightHash()).not.toBe(hash);
    ft.stop();
  }, 60_000);

  it('PPO-only and imitation-only modes run only their own part', async () => {
    for (const mode of ['ppo', 'imitation'] as const) {
      const t = await makeTrainer(2);
      const ft = new FineTune(t, { user, procedural, drainMs: 0, zeroShotEpisodes: 1, zeroShotMs: 0, mode });
      ft.start();
      let guard = 0;
      while (ft.phase !== 'tune' && guard++ < 100) await ft.step();
      const rep = await ft.step();
      expect(rep.ppoUpdates).toBe(mode === 'ppo' ? 1 : 0);
      expect(rep.imitationBatches).toBe(mode === 'ppo' ? 0 : 1);
      ft.stop();
    }
  }, 60_000);
});
