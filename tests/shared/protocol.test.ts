import { describe, expect, it } from 'vitest';
import {
  LIMITS,
  MESSAGE_TYPES,
  PROTOCOL_VERSION,
  parseEnvIn,
  parseEnvOut,
  parseFarmIn,
  parseFarmOut,
  parseLearnerIn,
  parseLearnerOut,
  stamp,
  type Parsed,
} from '../../src/shared/messages';
import { generateTrack } from '../../src/track/generate';
import { episodeConfig } from '../../src/rl/scenarios';
import { Trainer } from '../../src/rl/trainer';
import { FineTune } from '../../src/rl/finetune';

const track = generateTrack({ seed: 3, stage: 0 });
const cfg = episodeConfig({ track, stage: 0, dr: 'high' }, 1, 0);
const hero = {
  x: 1,
  y: 0.05,
  z: -2,
  yaw: 0.3,
  pitch: 0,
  roll: 0,
  speed: 2,
  steer: 0.1,
  throttle: 0.5,
  slip: [0, 0, 0.1, 0.1],
  load: [4, 4, 4, 4],
  lap: 0,
  lapTime: 1.2,
  s: 3,
  yawRate: 0.2,
  vx: 2,
  vz: 0,
  surfaceMu: 1,
};
const stat = { gen: 1, bestLap: null, meanLap: null, episodes: 0, completion: 0, crashRate: 0, topSpeed: 0, meanReturn: 0, trackId: 'gen-3-s0' };

/** One valid example per message type; the tests below fail if a type has no example. */
const EXAMPLES = {
  learnerIn: {
    ping: { type: 'ping', t: 1 },
    init: { type: 'init', n: 8, seeds: [1, 2, 3, 4], stage: 0 },
    pause: { type: 'pause', paused: true },
    baseline: { type: 'baseline', on: false },
    setStage: { type: 'setStage', stage: 2 },
    setDR: { type: 'setDR', grip: 1, mass: 1.1, level: 'low' },
    replayReq: { type: 'replayReq', gen: 4 },
    setTrackPool: { type: 'setTrackPool', points: [0, 0, 10, 0, 10, 10], halfWidth: 1, mu: 1, fineTune: true },
    checkpointSave: { type: 'checkpointSave', name: 'manual' },
    checkpointLoad: { type: 'checkpointLoad', name: 'manual' },
    ghostAck: { type: 'ghostAck' },
  },
  learnerOut: {
    pong: { type: 'pong', t: 1 },
    status: { type: 'status', text: 'racing' },
    ready: { type: 'ready', hash: 'abcd1234', trackId: 'gen-1-s0' },
    ghosts: { type: 'ghosts', poses: new Float32Array(14), n: 2, gen: 0, trackId: 'gen-1-s0', ids: ['a', 'b'], hero, bestLap: null },
    metrics: { type: 'metrics', stat, tape: [0, 1, 2] },
    replayRes: { type: 'replayRes', gen: 1, poses: [], n: 0 },
    swapStart: { type: 'swapStart', trackId: 'user-1', hashBefore: 'abcd', ms: 3 },
    adapt: { type: 'adapt', phase: 'drain', envTotal: 8, steps: 0, latencyMs: null },
    trackAck: { type: 'trackAck', trackId: 'user-1', envCount: 8, envTotal: 8, latencyMs: 2400, hash: 'abcd', hashBefore: 'abcd' },
    checkpointSaved: { type: 'checkpointSaved', name: 'manual' },
    error: { type: 'error', message: 'nope' },
  },
  envIn: {
    ping: { type: 'ping', t: 2 },
    init: { type: 'init', n: 4, seeds: [1], stage: 0 },
    step: { type: 'step' },
    setTrackPool: { type: 'setTrackPool', points: [0, 0, 1, 1], halfWidth: 1, mu: 1 },
  },
  envOut: {
    pong: { type: 'pong', t: 2 },
    ready: { type: 'ready', trackId: 'gen-1-s0' },
    ghosts: { type: 'ghosts', poses: new Float32Array(7), n: 1, gen: 0, trackId: 'gen-1-s0', ids: ['gen-1-s0'] },
    trackAck: { type: 'trackAck', trackId: 'user-1', envCount: 3 },
    error: { type: 'error', message: 'nope' },
  },
  farmIn: {
    ping: { type: 'ping', t: 3 },
    farmInit: { type: 'farmInit', shard: 0, lo: 0, hi: 4, total: 16, actDim: 2, buffer: new SharedArrayBuffer(64) },
    farmReset: { type: 'farmReset', epoch: 1, tracks: [track], cfgs: [cfg] },
    farmStop: { type: 'farmStop' },
  },
  farmOut: {
    pong: { type: 'pong', t: 3 },
    farmReady: { type: 'farmReady', shard: 1 },
    farmResetDone: { type: 'farmResetDone', shard: 1, epoch: 1 },
    farmStopped: { type: 'farmStopped', shard: 1 },
    error: { type: 'error', message: 'boom', shard: 2 },
  },
} as const;

const PARSERS: Record<keyof typeof EXAMPLES, (d: unknown) => Parsed<unknown>> = {
  learnerIn: parseLearnerIn,
  learnerOut: parseLearnerOut,
  envIn: parseEnvIn,
  envOut: parseEnvOut,
  farmIn: parseFarmIn,
  farmOut: parseFarmOut,
};

const errorOf = (r: Parsed<unknown>) => (r.ok ? '' : r.error);

describe('worker protocol v2', () => {
  for (const channel of Object.keys(EXAMPLES) as (keyof typeof EXAMPLES)[]) {
    it(`${channel}: every message type has an example, and every example parses (also after structured clone)`, () => {
      const examples = EXAMPLES[channel] as Record<string, { type: string }>;
      expect(Object.keys(examples).sort()).toEqual([...MESSAGE_TYPES[channel]].sort());
      for (const ex of Object.values(examples)) {
        const wire = stamp(ex);
        const direct = PARSERS[channel](wire);
        expect(direct.ok, `${channel}/${ex.type}: ${errorOf(direct)}`).toBe(true);
        const cloned = PARSERS[channel](structuredClone(wire));
        expect(cloned.ok, `${channel}/${ex.type} after clone: ${errorOf(cloned)}`).toBe(true);
      }
    });
  }

  it('stamp adds the version to everything but ping and pong, without touching the input', () => {
    const msg = { type: 'pause' as const, paused: true };
    const wire = stamp(msg);
    expect(wire).toEqual({ type: 'pause', paused: true, v: PROTOCOL_VERSION });
    expect(msg).toEqual({ type: 'pause', paused: true });
    expect(stamp({ type: 'ping' as const, t: 4 })).toEqual({ type: 'ping', t: 4 });
    expect(stamp({ type: 'pong' as const, t: 4 })).toEqual({ type: 'pong', t: 4 });
  });

  it('rejects messages from version 1 (no v) and from another version, naming the problem', () => {
    expect(errorOf(parseLearnerIn({ type: 'pause', paused: true }))).toMatch(/pause\.v: missing/);
    expect(errorOf(parseLearnerIn({ type: 'pause', paused: true, v: 1 }))).toMatch(/protocol 1 is not 2/);
    expect(errorOf(parseFarmOut({ type: 'farmReady', shard: 0, v: 3 }))).toMatch(/protocol 3 is not 2/);
    // ping stays unversioned so any worker can be probed
    expect(parseLearnerIn({ type: 'ping', t: 0 }).ok).toBe(true);
  });

  it('rejects non-objects, unknown types and prototype keys', () => {
    for (const bad of [null, undefined, 3, 'ping', [], new Float32Array(2)]) expect(parseLearnerIn(bad).ok).toBe(false);
    expect(errorOf(parseLearnerIn({ v: 2 }))).toMatch(/message\.type/);
    expect(errorOf(parseLearnerIn(stamp({ type: 'startTrain' })))).toMatch(/unknown message type "startTrain"/);
    expect(errorOf(parseLearnerIn(stamp({ type: 'toString' })))).toMatch(/unknown message type/);
    expect(errorOf(parseLearnerIn(stamp({ type: '__proto__' })))).toMatch(/unknown message type/);
    // a farm message on the learner channel is unknown there
    expect(parseLearnerIn(stamp({ type: 'farmStop' })).ok).toBe(false);
  });

  it('rejects missing fields, wrong types and non-finite numbers with the field path', () => {
    expect(errorOf(parseLearnerIn(stamp({ type: 'setStage' })))).toBe('setStage.stage: missing');
    expect(errorOf(parseLearnerIn(stamp({ type: 'setStage', stage: '2' })))).toBe('setStage.stage: expected an integer, got string');
    expect(errorOf(parseLearnerIn(stamp({ type: 'setDR', grip: NaN, mass: 1, level: 'low' })))).toMatch(/setDR\.grip: expected a finite number, got NaN/);
    expect(errorOf(parseLearnerIn(stamp({ type: 'setDR', grip: 1, mass: 1, level: 'ice' })))).toMatch(/setDR\.level: expected one of off, low, high/);
    expect(errorOf(parseLearnerIn(stamp({ type: 'init', n: 8, seeds: [1, 'x'], stage: 0 })))).toMatch(/init\.seeds\[1\]/);
    expect(errorOf(parseLearnerOut(stamp({ ...EXAMPLES.learnerOut.ghosts, poses: [0, 0, 0] })))).toMatch(/ghosts\.poses: expected a Float32Array, got array/);
    expect(errorOf(parseLearnerOut(stamp({ ...EXAMPLES.learnerOut.ghosts, hero: { ...hero, speed: undefined } })))).toBe('ghosts.hero.speed: missing');
    expect(errorOf(parseLearnerOut(stamp({ ...EXAMPLES.learnerOut.metrics, stat: { ...stat, episodes: -1 } })))).toMatch(/metrics\.stat\.episodes: expected >= 0/);
  });

  it('enforces sizes and cross-field rules', () => {
    expect(errorOf(parseLearnerIn(stamp({ type: 'init', n: 65, seeds: [1], stage: 0 })))).toMatch(/init\.n: expected 1 to 64/);
    expect(errorOf(parseLearnerIn(stamp({ type: 'init', n: 0, seeds: [1], stage: 0 })))).toMatch(/init\.n: expected 1 to 64/);
    expect(errorOf(parseLearnerIn(stamp({ type: 'setTrackPool', points: [0, 0, 1], halfWidth: 1, mu: 1 })))).toMatch(/x,z pairs/);
    const huge = new Array<number>(LIMITS.points + 2).fill(0);
    expect(errorOf(parseLearnerIn(stamp({ type: 'setTrackPool', points: huge, halfWidth: 1, mu: 1 })))).toMatch(/at most 20000 items/);
    expect(errorOf(parseLearnerOut(stamp({ ...EXAMPLES.learnerOut.ghosts, n: 3 })))).toMatch(/expected n \* 7 = 21 floats, got 14/);
    const init = EXAMPLES.farmIn.farmInit;
    expect(errorOf(parseFarmIn(stamp({ ...init, lo: 4, hi: 4 })))).toMatch(/lo < hi <= total/);
    expect(errorOf(parseFarmIn(stamp({ ...init, hi: 17 })))).toMatch(/lo < hi <= total/);
    expect(errorOf(parseFarmIn(stamp({ ...init, actDim: 4 })))).toMatch(/actDim: expected 2 or 3/);
    expect(errorOf(parseFarmIn(stamp({ ...init, buffer: new ArrayBuffer(64) })))).toMatch(/buffer: expected a SharedArrayBuffer, got ArrayBuffer/);
    expect(errorOf(parseFarmIn(stamp({ type: 'farmReset', epoch: 1, tracks: [track, track], cfgs: [cfg] })))).toMatch(/same non-zero number/);
  });

  it('checks the structure of tracks and episode configs sent to the farm', () => {
    const reset = (t: unknown, c: unknown = cfg) => errorOf(parseFarmIn(stamp({ type: 'farmReset', epoch: 0, tracks: [t], cfgs: [c] })));
    expect(reset({ ...track, startIndex: track.samples.length })).toMatch(/startIndex: \d+ is past the last sample/);
    expect(reset({ ...track, walls: [[0, 1]] })).toMatch(/tracks\[0\]\.walls\[0\]: expected a Float32Array/);
    expect(reset({ ...track, samples: [{ ...track.samples[0], mu: 'x' }, ...track.samples.slice(1)] })).toMatch(/samples\[0\]\.mu/);
    expect(reset(track, { ...cfg, latencyTicks: -1 })).toMatch(/cfgs\[0\]\.latencyTicks: expected >= 0/);
    expect(reset(track, { ...cfg, car: { ...cfg.car, drive: 'awd' } })).toMatch(/cfgs\[0\]\.car\.drive/);
    expect(reset(track, { ...cfg, drift: { handbrake: 'yes' } })).toMatch(/cfgs\[0\]\.drift\.handbrake: expected a boolean/);
  });

  it('accepts the payloads the learner actually sends (GenStat, hero snapshot, poses, fine-tune report)', async () => {
    const trainer = new Trainer({ n: 2, horizon: 8, epochs: 1, minibatch: 16, seed: 5 });
    await trainer.init([track]);
    const s = await trainer.update();
    const wire = [
      { type: 'metrics', stat: s, tape: Array.from(trainer.replays.get(trainer.gen) ?? []) },
      {
        type: 'ghosts',
        poses: new Float32Array(trainer.env.poses()),
        n: trainer.env.n,
        gen: trainer.gen,
        trackId: trainer.env.trackId(0),
        ids: trainer.env.trackIds(),
        hero: trainer.env.snapshot(0),
        bestLap: trainer.bestLap,
      },
      { ...new FineTune(trainer, { user: track, procedural: [track] }).report(), type: 'adapt' },
      { type: 'ready', hash: trainer.weightHash(), trackId: trainer.env.trackId(0) },
    ];
    for (const m of wire) {
      const r = parseLearnerOut(structuredClone(stamp(m)));
      expect(r.ok, `${m.type}: ${errorOf(r)}`).toBe(true);
    }
    trainer.policy.dispose();
  });
});
