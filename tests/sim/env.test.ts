import { describe, expect, it } from 'vitest';
import { ACT_DIM, OBS_DIM, type EpisodeConfig, type TrackSpec } from '../../src/shared/types';
import { OBS } from '../../src/shared/constants';
import { RayField } from '../../src/sim/sensors';
import { VecEnv, stepCapFor } from '../../src/sim/vecEnv';
import { defaultCarParams } from '../../src/sim/car/params';
import { generateTrack } from '../../src/track/generate';

function cfg(track: TrackSpec, extra: Partial<EpisodeConfig> = {}): EpisodeConfig {
  return { stage: 0, car: defaultCarParams(), noiseScale: 0, latencyTicks: 0, trackId: track.id, ...extra };
}

/** The env's slots are private; tests teleport a car with the same call `place` uses. */
function teleport(env: VecEnv, i: number, x: number, z: number, yaw: number, speed: number): void {
  const slot = (env as unknown as { slots: { car: { reset(p: object): void } }[] }).slots[i]!;
  slot.car.reset({ x, z, yaw, speed });
}

describe('ray field', () => {
  const mid = 10; // the ray along the car's heading (21 rays over a symmetric arc)

  it('reports the nearest wall whichever order the walls are listed in', () => {
    const near = new Float32Array([3, -5, 3, 5]);
    const far = new Float32Array([5, -5, 5, 5]);
    for (const walls of [[near, far], [far, near]]) {
      const field = new RayField();
      field.setTrack({ walls } as unknown as TrackSpec);
      const out = new Float32Array(21);
      field.cast(0, 0, 0, out, 0, null, 0);
      expect(out[mid]).toBeCloseTo(3 / 6, 5);
    }
  });

  it('reports max range when nothing is in reach', () => {
    const field = new RayField();
    field.setTrack({ walls: [new Float32Array([40, -5, 40, 5])] } as unknown as TrackSpec);
    const out = new Float32Array(21);
    field.cast(0, 0, 0, out, 0, null, 0);
    expect(out[mid]).toBe(1);
  });
});

describe('vec env laps', () => {
  const track = generateTrack({ seed: 3, stage: 0 });

  it('does not award a lap for reversing over the start line and driving forward again', () => {
    const env = new VecEnv(1);
    env.reset([track], [cfg(track, { lapsTarget: 2 })]);
    const s0 = track.samples[track.startIndex]!;
    // Start 1.5 m behind the line, as if the car had reversed across it.
    const back = track.samples[(track.startIndex - Math.round(1.5 / (track.length / track.samples.length)) + track.samples.length) % track.samples.length]!;
    teleport(env, 0, back.x, back.z, Math.atan2(back.tz, back.tx), 2);
    let total = 0;
    let laps = 0;
    for (let k = 0; k < 120; k++) {
      const out = env.step(env.baselineActions());
      total += out.rew[0]!;
      if (out.infos[0]!.lapDone) laps++;
    }
    expect(s0).toBeDefined();
    expect(laps).toBe(0);
    // A real lap pays 10+; 120 steps of honest progress pay a few.
    expect(total).toBeLessThan(8);
  });

  it('counts a real lap and times each lap separately', () => {
    const env = new VecEnv(1);
    env.reset([track], [cfg(track, { lapsTarget: 2 })]);
    const times: number[] = [];
    for (let k = 0; k < stepCapFor(track.length, 2) && times.length < 2; k++) {
      const out = env.step(env.baselineActions());
      const info = out.infos[0]!;
      if (info.lapDone) times.push(info.lapTime!);
      if (out.done[0] || out.trunc[0]) break;
    }
    expect(times).toHaveLength(2);
    // The second lap is its own lap: about as long as the first, not the running total.
    expect(times[1]!).toBeLessThan(times[0]! * 1.4);
    expect(times[1]!).toBeGreaterThan(times[0]! * 0.6);
    expect(env.finished.at(-1)?.lap).toBe(true);
  });
});

describe('vec env observations', () => {
  it('counts time_left down against the same step cap that truncates the episode', () => {
    const track = generateTrack({ seed: 4, stage: 0 });
    const env = new VecEnv(1);
    env.reset([track], [cfg(track, { lapsTarget: 1 })]);
    const cap = stepCapFor(track.length, 1);
    const hold = new Float32Array(ACT_DIM);
    const k = 25;
    for (let i = 0; i < k; i++) env.step(hold);
    expect(env.obs.length).toBe(OBS_DIM);
    expect(env.obs[OBS.TIME_LEFT]).toBeCloseTo(1 - k / cap, 5);
  });
});

describe('vec env episode records', () => {
  it('does not count a completed lap as a crash', () => {
    const track = generateTrack({ seed: 3, stage: 0 });
    const env = new VecEnv(1);
    env.reset([track], [cfg(track, { lapsTarget: 1 })]);
    for (let k = 0; k < stepCapFor(track.length, 1) && env.finished.length === 0; k++) env.step(env.baselineActions());
    const ep = env.finished[0]!;
    expect(ep.lap).toBe(true);
    expect(ep.crash).toBe(false);
  });
});
