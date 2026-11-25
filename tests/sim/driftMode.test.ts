import { describe, expect, it } from 'vitest';
import type { EpisodeConfig, TrackSpec } from '../../src/shared/types';
import { makeRng } from '../../src/shared/rng';
import { Vehicle } from '../../src/sim/car/vehicle';
import { defaultCarParams } from '../../src/sim/car/params';
import { EpisodeBook, DRIFT_MIN_SPEED, scoreStep, slipBonus, type RewardInput } from '../../src/sim/reward';
import { VecEnv } from '../../src/sim/vecEnv';
import { generateTrack } from '../../src/track/generate';
import { countersteer } from './driftScript';

/**
 * Drift mode: a handbrake action and a slip-angle reward term, both off by default. Off, nothing
 * changes (the golden tests in vehicle.test.ts and vecEnvGolden.test.ts still pass untouched).
 */

const DEG = 180 / Math.PI;
const DT = 0.02;

const beta = (car: Vehicle) => Math.atan2(car.vz, Math.max(0.3, Math.abs(car.vx)));

function cruise(v0: number): Vehicle {
  const car = new Vehicle(defaultCarParams());
  car.reset({ speed: v0 });
  for (let i = 0; i < 100; i++) car.step(0, Math.max(0, Math.min(1, 0.4 + (v0 - car.speed()) * 0.3)), DT);
  return car;
}

/** A 0.3-steer turn at about 6.4 m/s on the stock tune, with an optional handbrake tap. */
function turn(tapSeconds: number) {
  const car = cruise(6);
  let peak = 0;
  let minV = Infinity;
  for (let i = 0; i < 50; i++) {
    const hb = i >= 5 && i < 5 + Math.round(tapSeconds / DT) ? 1 : 0;
    // Scripted driver: hold the turn, and countersteer once the slide passes 20 deg.
    const steer = Math.abs(beta(car)) * DEG > 20 ? countersteer(car, 1.0, 0.0) : 0.3;
    car.step(steer, hb ? 0 : 0.6, DT, hb);
    peak = Math.max(peak, Math.abs(beta(car)) * DEG);
    minV = Math.min(minV, car.speed());
  }
  let calm = 0;
  let recover = -1;
  for (let i = 0; i < 150; i++) {
    car.step(countersteer(car), 0.15, DT);
    calm = Math.abs(car.meanSlip('rear')) * DEG < 5 && Math.abs(car.wYaw) < 1 && car.speed() > 1 ? calm + 1 : 0;
    if (calm >= 25) {
      recover = (i + 1) * DT;
      break;
    }
  }
  return { peak, minV, recover };
}

describe('handbrake', () => {
  it('a handbrake of 0 is exactly the step without one', () => {
    const a = cruise(5);
    const b = cruise(5);
    const rng = makeRng(3, 'hb-zero');
    for (let i = 0; i < 200; i++) {
      const steer = rng.range(-1, 1);
      const thr = rng.range(-0.5, 1);
      const sa = a.step(steer, thr, DT);
      const sb = b.step(steer, thr, DT, 0);
      expect(sb).toEqual(sa);
    }
    expect(Array.from(b.omega)).toEqual(Array.from(a.omega));
  });

  it('locks the rear wheels and leaves the fronts rolling', () => {
    const car = cruise(6);
    for (let i = 0; i < 10; i++) car.step(0, 0, DT, 1);
    expect(Math.abs(car.omega[2]!)).toBeLessThan(0.5);
    expect(Math.abs(car.omega[3]!)).toBeLessThan(0.5);
    expect(car.omega[0]!).toBeGreaterThan(50);
    expect(car.omega[1]!).toBeGreaterThan(50);
    expect(car.speed()).toBeLessThan(6);
  });

  it('never spins a wheel backwards', () => {
    const car = cruise(2);
    for (let i = 0; i < 150; i++) {
      car.step(0, 0, DT, 1);
      expect(car.omega[2]!).toBeGreaterThanOrEqual(0);
      expect(car.omega[3]!).toBeGreaterThanOrEqual(0);
    }
  });

  it('a short tap starts a slide on the stock tune that the same turn without it does not reach, and countersteer settles it', () => {
    const plain = turn(0);
    const tapped = turn(0.08);
    expect(plain.peak).toBeLessThan(15);
    expect(tapped.peak).toBeGreaterThan(25);
    expect(tapped.peak).toBeLessThan(75);
    expect(tapped.minV).toBeGreaterThan(4);
    expect(tapped.recover).toBeGreaterThan(0);
    expect(tapped.recover).toBeLessThan(1.5);
  });
});

function setup(n: number, drift?: EpisodeConfig['drift']): { tracks: TrackSpec[]; cfgs: EpisodeConfig[] } {
  const track = generateTrack({ seed: 21, stage: 0 });
  const tracks = Array.from({ length: n }, () => track);
  const cfgs = tracks.map((t, i) => ({ stage: 0, car: defaultCarParams(), noiseScale: 1, latencyTicks: i % 3, trackId: t.id, ...(drift ? { drift } : {}) }));
  return { tracks, cfgs };
}

function rollout(env: VecEnv, cfg: { tracks: TrackSpec[]; cfgs: EpisodeConfig[] }, steps: number, hb: (t: number, i: number) => number) {
  env.reset(cfg.tracks, cfg.cfgs);
  const obs: number[][] = [];
  const rew: number[][] = [];
  for (let t = 0; t < steps; t++) {
    const two = env.baselineActions();
    const a = new Float32Array(env.n * env.actDim);
    for (let i = 0; i < env.n; i++) {
      a[i * env.actDim] = two[i * env.actDim]!;
      a[i * env.actDim + 1] = two[i * env.actDim + 1]!;
      if (env.actDim === 3) a[i * 3 + 2] = hb(t, i);
    }
    const out = env.step(a);
    obs.push(Array.from(out.obs));
    rew.push(Array.from(out.rew));
  }
  return { obs, rew };
}

describe('drift mode in VecEnv', () => {
  it('actDim 3 with drift off is the same env as actDim 2, whatever the third float says', () => {
    const two = rollout(new VecEnv(4), setup(4), 200, () => 0);
    const three = rollout(new VecEnv(4, { actDim: 3 }), setup(4), 200, (t, i) => ((t + i) % 7) / 6);
    const threeOff = rollout(new VecEnv(4, { actDim: 3 }), setup(4, { handbrake: false, slipWeight: 0 }), 200, () => 1);
    expect(three).toEqual(two);
    expect(threeOff).toEqual(two);
  });

  it('the handbrake changes the run only when the slot enables it', () => {
    const off = rollout(new VecEnv(4, { actDim: 3 }), setup(4), 120, (t) => (t > 40 && t < 50 ? 1 : 0));
    const on = rollout(new VecEnv(4, { actDim: 3 }), setup(4, { handbrake: true }), 120, (t) => (t > 40 && t < 50 ? 1 : 0));
    expect(on.obs.slice(0, 41)).toEqual(off.obs.slice(0, 41));
    expect(on.obs.slice(45)).not.toEqual(off.obs.slice(45));
  });

  it('is deterministic with drift on', () => {
    const cfg = { handbrake: true, slipWeight: 0.5 };
    const a = rollout(new VecEnv(4, { actDim: 3 }), setup(4, cfg), 150, (t, i) => ((t * 3 + i) % 11 === 0 ? 1 : 0));
    const b = rollout(new VecEnv(4, { actDim: 3 }), setup(4, cfg), 150, (t, i) => ((t * 3 + i) % 11 === 0 ? 1 : 0));
    expect(b).toEqual(a);
  });

  it('rejects an action size other than 2 or 3', () => {
    expect(() => new VecEnv(1, { actDim: 4 as 3 })).toThrow(/actDim must be 2 or 3/);
  });
});

describe('slip-angle reward', () => {
  const base: RewardInput = {
    ds: 0.1,
    dt: DT,
    lapDone: false,
    lapTime: 3,
    tRef: 7,
    offTrack: false,
    wallHit: false,
    impact: 0,
    inContact: false,
    steer: 0.2,
    throttle: 0.6,
    prevSteer: 0.1,
    prevThrottle: 0.6,
    speed: 5,
    wrongWay: false,
  };
  const target = (25 * Math.PI) / 180;

  it('leaves the reward exactly as before at weight 0 or with no weight', () => {
    const plain = scoreStep(base, new EpisodeBook(), 1000, 1, 0);
    expect(scoreStep({ ...base, slip: 0.5, slipWeight: 0 }, new EpisodeBook(), 1000, 1, 0)).toEqual(plain);
    expect(scoreStep({ ...base, slip: 0.5 }, new EpisodeBook(), 1000, 1, 0)).toEqual(plain);
  });

  it('pays in proportion to progress, up to the target angle', () => {
    expect(slipBonus({ ...base, slip: target / 2, slipWeight: 0.5 })).toBeCloseTo(0.5 * 0.5 * 0.1, 12);
    expect(slipBonus({ ...base, slip: -target, slipWeight: 0.5 })).toBeCloseTo(0.05, 12);
    expect(slipBonus({ ...base, slip: 3 * target, slipWeight: 0.5 })).toBeCloseTo(0.05, 12);
    expect(slipBonus({ ...base, slip: 0.2, slipWeight: 0.5, slipTarget: 0.4 })).toBeCloseTo(0.025, 12);
    const plain = scoreStep(base, new EpisodeBook(), 1000, 1, 0).reward;
    const drifting = scoreStep({ ...base, slip: target, slipWeight: 0.5 }, new EpisodeBook(), 1000, 1, 0);
    expect(drifting.reward - plain).toBeCloseTo(0.05, 12);
    expect(drifting.info.rewardRaw).toBe(drifting.reward);
  });

  it('pays nothing without progress, off track, at a wall, the wrong way or when slow', () => {
    const on = { ...base, slip: target, slipWeight: 1 };
    expect(slipBonus(on)).toBeGreaterThan(0);
    for (const bad of [
      { ds: 0 },
      { ds: -0.02 },
      { ds: 0.6 },
      { offTrack: true },
      { wallHit: true },
      { inContact: true },
      { wrongWay: true },
      { speed: DRIFT_MIN_SPEED },
    ]) {
      expect(slipBonus({ ...on, ...bad }), JSON.stringify(bad)).toBe(0);
    }
  });

  it('adds reward in VecEnv when a slot drifts with the bonus on', () => {
    const off = rollout(new VecEnv(2, { actDim: 3 }), setup(2, { handbrake: true }), 200, (t) => (t > 60 && t < 64 ? 1 : 0));
    const on = rollout(new VecEnv(2, { actDim: 3 }), setup(2, { handbrake: true, slipWeight: 1 }), 200, (t) => (t > 60 && t < 64 ? 1 : 0));
    // Same trajectory (the bonus does not touch the physics), more reward.
    expect(on.obs).toEqual(off.obs);
    const sum = (r: number[][]) => r.flat().reduce((a, b) => a + b, 0);
    expect(sum(on.rew)).toBeGreaterThan(sum(off.rew));
  });
});
