import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { ACT_DIM, OBS_DIM, type EpisodeConfig, type TrackSpec } from '../../src/shared/types';
import { makeRng } from '../../src/shared/rng';
import { defaultCarParams } from '../../src/sim/car/params';
import { latencyFor, noiseFor, sampleCar } from '../../src/sim/dr';
import { VecEnv } from '../../src/sim/vecEnv';
import { generateTrack } from '../../src/track/generate';
import { EnvFarm, type FarmEpisodeRecord, type SpawnShard } from '../../src/workers/envFarm';
import { CTRL, MAX_SHARDS, farmBytes, farmSupport, farmViews, shardRanges } from '../../src/workers/envFarmLayout';
import { serveShard, type FarmPort } from '../../src/workers/envFarmShard';
import { stamp, type FarmOut } from '../../src/shared/messages';
import { nodeSpawn } from '../helpers/nodeFarm';

const N = 16;
const STEPS = 700; // 14 s at 50 Hz, long enough for baseline laps to end episodes

/** 16 slots over 4 stage 0-1 tracks with high domain randomization: noise, latency and car jitter. */
function setup(): { tracks: TrackSpec[]; cfgs: EpisodeConfig[] } {
  const shapes = [11, 12, 13, 14].map((seed, k) => generateTrack({ seed, stage: k % 2 }));
  const rng = makeRng(77, 'farm-test');
  const tracks: TrackSpec[] = [];
  const cfgs: EpisodeConfig[] = [];
  for (let i = 0; i < N; i++) {
    const track = shapes[i % shapes.length]!;
    tracks.push(track);
    cfgs.push({
      stage: 0,
      car: sampleCar(defaultCarParams(), rng, 'high'),
      noiseScale: noiseFor(rng, 'high'),
      latencyTicks: latencyFor(rng, 'high'),
      trackId: track.id,
      lapsTarget: 1,
    });
  }
  return { tracks, cfgs };
}

/**
 * Actions for the run: even slots follow the pure-pursuit baseline (so laps finish and episodes
 * end inside the run), odd slots drive at random. The reference run records them and the farm runs
 * replay the recording, so every path sees the same actions.
 */
function noisyActions(env: VecEnv, rng: ReturnType<typeof makeRng>): Float32Array {
  const a = env.baselineActions().slice();
  for (let i = 1; i < N; i += 2) {
    a[i * ACT_DIM] = rng.range(-1, 1);
    a[i * ACT_DIM + 1] = rng.range(-0.2, 1);
  }
  return a;
}

interface Trace {
  actions: Float32Array[];
  obs: Float32Array[];
  rew: Float32Array[];
  done: Uint8Array[];
  trunc: Uint8Array[];
  poses: Float32Array[];
  infos: string[];
  episodes: FarmEpisodeRecord[];
}

function emptyTrace(first: Float32Array, poses: Float32Array): Trace {
  return { actions: [], obs: [first.slice()], rew: [], done: [], trunc: [], poses: [poses.slice()], infos: [], episodes: [] };
}

function referenceTrace(): Trace {
  const { tracks, cfgs } = setup();
  const env = new VecEnv(N);
  const trace = emptyTrace(env.reset(tracks, cfgs), env.poses());
  const rng = makeRng(5, 'farm-actions');
  for (let t = 0; t < STEPS; t++) {
    const a = noisyActions(env, rng);
    trace.actions.push(a);
    const out = env.step(a);
    trace.obs.push(out.obs.slice());
    trace.rew.push(out.rew.slice());
    trace.done.push(out.done.slice());
    trace.trunc.push(out.trunc.slice());
    trace.poses.push(env.poses().slice());
    trace.infos.push(JSON.stringify(out.infos.map((x) => [x.progress, x.lapDone, x.lapDone ? x.lapTime : 0, x.offTrack, x.wallHit, x.speed, x.wrongWay, x.stuck, x.rewardRaw].map(Number).map(Math.fround))));
    let k = 0;
    for (let i = 0; i < N; i++) {
      if (!out.done[i] && !out.trunc[i]) continue;
      const r = env.finished[k++]!;
      trace.episodes.push({ slot: i, lap: r.lap, lapTime: r.lap ? Math.fround(r.lapTime!) : undefined, crash: r.crash, ret: Math.fround(r.ret), trackId: r.trackId, tRef: Math.fround(r.tRef), speed: Math.fround(r.speed) });
    }
    env.finished.length = 0;
  }
  return trace;
}

async function farmTrace(farm: EnvFarm, actions: Float32Array[], asyncEvery = 0): Promise<Trace> {
  const { tracks, cfgs } = setup();
  const trace = emptyTrace(await farm.reset(tracks, cfgs), farm.poses());
  let t = 0;
  for (const a of actions) {
    trace.actions.push(a);
    const out = asyncEvery > 0 && t % asyncEvery === 0 ? await farm.stepAsync(a) : farm.step(a);
    t++;
    trace.obs.push(out.obs.slice());
    trace.rew.push(out.rew.slice());
    trace.done.push(out.done.slice());
    trace.trunc.push(out.trunc.slice());
    trace.poses.push(farm.poses().slice());
    const infos = [];
    for (let i = 0; i < N; i++) {
      const x = farm.info(i);
      infos.push([x.progress, x.lapDone, x.lapDone ? x.lapTime : 0, x.offTrack, x.wallHit, x.speed, x.wrongWay, x.stuck, x.rewardRaw].map(Number).map(Math.fround));
    }
    trace.infos.push(JSON.stringify(infos));
    trace.episodes.push(...out.episodes);
  }
  return trace;
}

function expectSame(a: Trace, b: Trace): void {
  expect(b.obs.length).toBe(a.obs.length);
  for (let t = 0; t < a.obs.length; t++) {
    expect(Buffer.from(b.obs[t]!.buffer).equals(Buffer.from(a.obs[t]!.buffer)), `obs at step ${t}`).toBe(true);
    expect(Buffer.from(b.poses[t]!.buffer).equals(Buffer.from(a.poses[t]!.buffer)), `poses at step ${t}`).toBe(true);
  }
  for (let t = 0; t < a.rew.length; t++) {
    expect(Buffer.from(b.rew[t]!.buffer).equals(Buffer.from(a.rew[t]!.buffer)), `rew at step ${t}`).toBe(true);
    expect(Array.from(b.done[t]!)).toEqual(Array.from(a.done[t]!));
    expect(Array.from(b.trunc[t]!)).toEqual(Array.from(a.trunc[t]!));
  }
  expect(b.infos).toEqual(a.infos);
  expect(b.episodes).toEqual(a.episodes);
}

describe('env farm layout', () => {
  it('splits slots into contiguous shards that differ by at most one', () => {
    expect(shardRanges(16, 4)).toEqual([[0, 4], [4, 8], [8, 12], [12, 16]]);
    expect(shardRanges(10, 4)).toEqual([[0, 3], [3, 6], [6, 8], [8, 10]]);
    expect(shardRanges(3, 4)).toEqual([[0, 1], [1, 2], [2, 3]]);
    expect(shardRanges(64, 99)).toHaveLength(MAX_SHARDS);
  });

  it('lays out aligned views and refuses a buffer that is too small', () => {
    const bytes = farmBytes(N, ACT_DIM);
    const v = farmViews(new SharedArrayBuffer(bytes), N, ACT_DIM);
    expect(v.obs.length).toBe(N * OBS_DIM);
    expect(v.actions.length).toBe(N * ACT_DIM);
    expect(v.poses.length).toBe(N * 7);
    for (const view of [v.ctrl, v.actions, v.obs, v.rew, v.poses, v.info, v.ep]) expect(view.byteOffset % 8).toBe(0);
    expect(() => farmViews(new SharedArrayBuffer(bytes - 8), N, ACT_DIM)).toThrow(/too small/);
  });

  it('reports support in Node', () => {
    expect(farmSupport()).toEqual({ ok: true });
  });
});

describe('env farm inline', () => {
  const reference = referenceTrace();

  it('episodes end during the reference run, so records are exercised', () => {
    expect(reference.episodes.length).toBeGreaterThan(3);
    expect(reference.episodes.some((e) => e.lap)).toBe(true);
  });

  it('matches one big VecEnv byte for byte with 4 shards', async () => {
    const farm = await EnvFarm.open({ total: N, shards: 4 });
    expect(farm.mode).toBe('inline');
    expectSame(reference, await farmTrace(farm, reference.actions));
    await farm.close();
  });

  it('does not depend on the shard count', async () => {
    for (const shards of [1, 3, 5]) {
      const farm = await EnvFarm.open({ total: N, shards });
      expectSame(reference, await farmTrace(farm, reference.actions));
      await farm.close();
    }
  });

  it('rejects misuse with plain errors', async () => {
    const farm = await EnvFarm.open({ total: 4, shards: 2 });
    expect(() => farm.step(new Float32Array(8))).toThrow(/step before reset/);
    const { tracks, cfgs } = setup();
    await expect(farm.reset(tracks, cfgs)).rejects.toThrow(/expected 4 tracks/);
    await farm.reset(tracks.slice(0, 4), cfgs.slice(0, 4));
    expect(() => farm.step(new Float32Array(7))).toThrow(/expected 8 actions/);
    await farm.close();
    expect(() => farm.step(new Float32Array(8))).toThrow(/closed/);
    await expect(EnvFarm.open({ total: 0 })).rejects.toThrow(/positive integer/);
  });
});

describe('env farm in worker threads', () => {
  let spawn: SpawnShard;
  const reference = referenceTrace();

  beforeAll(async () => {
    spawn = await nodeSpawn();
  }, 60_000);

  it('matches one big VecEnv byte for byte with 4 workers, blocking and async steps', async () => {
    const farm = await EnvFarm.open({ total: N, shards: 4, spawn });
    try {
      expect(farm.mode).toBe('workers');
      expect(farm.shardCount).toBe(4);
      expectSame(reference, await farmTrace(farm, reference.actions));
      expectSame(reference, await farmTrace(farm, reference.actions, 3));
      expect(farm.errors).toEqual([]);
    } finally {
      await farm.close();
    }
  }, 60_000);

  it('fails fast with the shard error when a shard cannot reset', async () => {
    const farm = await EnvFarm.open({ total: 4, shards: 2, spawn });
    try {
      const { tracks, cfgs } = setup();
      const bad = tracks.slice(0, 4).map((t, i) => (i === 3 ? { ...t, samples: [] } : t));
      await expect(farm.reset(bad, cfgs.slice(0, 4))).rejects.toThrow(/shard 1 reset failed/);
    } finally {
      await farm.close();
    }
  }, 60_000);

  afterAll(() => undefined);
});

describe('env farm shard server', () => {
  function harness(): { port: FarmPort; sent: FarmOut[]; deliver: (d: unknown) => void } {
    const sent: FarmOut[] = [];
    let handler: (d: unknown) => void = () => undefined;
    const port: FarmPort = { post: (m) => sent.push(m), onMessage: (fn) => (handler = fn), close: () => undefined };
    return { port, sent, deliver: (d) => handler(d) };
  }

  it('answers ping, rejects bad messages and resets before init', () => {
    const h = harness();
    serveShard(h.port);
    h.deliver({ type: 'ping', t: 4 });
    h.deliver({ type: 'farmStep' });
    h.deliver(stamp({ type: 'farmReset', epoch: 1, tracks: [setup().tracks[0]!], cfgs: [setup().cfgs[0]!] }));
    expect(h.sent[0]).toEqual({ type: 'pong', t: 4 });
    expect(h.sent[1]).toMatchObject({ type: 'error', message: expect.stringMatching(/rejected message/) });
    expect(h.sent[2]).toMatchObject({ type: 'error', message: 'farmReset before farmInit' });
  });

  it('steps when SEQ moves and stops on farmStop', async () => {
    const h = harness();
    serveShard(h.port);
    const buffer = new SharedArrayBuffer(farmBytes(2, ACT_DIM));
    h.deliver(stamp({ type: 'farmInit', shard: 0, lo: 0, hi: 2, total: 2, actDim: ACT_DIM, buffer }));
    const { tracks, cfgs } = setup();
    h.deliver(stamp({ type: 'farmReset', epoch: 1, tracks: tracks.slice(0, 2), cfgs: cfgs.slice(0, 2) }));
    expect(h.sent.map((m) => m.type)).toEqual(['farmReady', 'farmResetDone']);
    const v = farmViews(buffer, 2, ACT_DIM);
    v.actions.set([0, 1, 0, 1]);
    Atomics.store(v.ctrl, CTRL.SEQ, 1);
    Atomics.notify(v.ctrl, CTRL.SEQ);
    const t0 = Date.now();
    while (Atomics.load(v.ctrl, CTRL.DONE0) !== 1 && Date.now() - t0 < 5000) await new Promise((r) => setTimeout(r, 5));
    expect(Atomics.load(v.ctrl, CTRL.DONE0)).toBe(1);
    expect(v.rew.some((x) => x !== 0)).toBe(true);
    h.deliver(stamp({ type: 'farmStop' }));
    expect(h.sent.at(-1)).toEqual({ type: 'farmStopped', shard: 0, v: 2 });
    Atomics.store(v.ctrl, CTRL.STOP, 1);
    Atomics.notify(v.ctrl, CTRL.SEQ);
  });
});
